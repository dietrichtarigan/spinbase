"""Layer 1: hybrid dense + BM25 retrieval with priority-tier boosting.

* Dense: local embeddings (bge-m3 default, SPECTER2 optional) stored as float16 BLOBs
  in SQLite and held in memory as a normalised float32 matrix (500 papers ≈ 40k chunks
  ≈ 160 MB at 1024-d — no vector DB server needed).
* Lexical: SQLite FTS5 BM25 over formula-normalised text (Fe$_3$GaTe$_2$ == Fe3GaTe2).
* Fusion: weighted Reciprocal Rank Fusion, then × priority weight
  (core 1.0 > supporting 0.8 > background 0.6 by default).
* Embedding never leaves the machine; private chunks are embedded locally too, and are
  returned only when ``include_private`` (for local display, never for LLM prompts).
"""
from __future__ import annotations

import hashlib
import logging
import re
import threading
from dataclasses import dataclass, field

import numpy as np

from config.settings import settings
from src.core.db import fts_normalize
from src.core.models import Chunk

log = logging.getLogger(__name__)


# ================================================================ embedders
class Embedder:
    name: str
    dim: int

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        raise NotImplementedError


class HashEmbedder(Embedder):
    """Deterministic feature-hashing embedder (no model download). For tests/offline only."""

    def __init__(self, dim: int = 512):
        self.name, self.dim = f"hash-{dim}", dim

    def encode(self, texts, is_query=False):
        M = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            toks = re.findall(r"[a-z0-9]+", fts_normalize(t).lower())
            for tok in toks + [a + "_" + b for a, b in zip(toks, toks[1:])]:
                h = int.from_bytes(hashlib.md5(tok.encode()).digest()[:4], "little")
                M[i, h % self.dim] += 1.0 if (h >> 31) & 1 else -1.0
        n = np.linalg.norm(M, axis=1, keepdims=True)
        return M / np.maximum(n, 1e-9)


class STEmbedder(Embedder):
    """sentence-transformers backend (bge-m3 dense, SPECTER2 base, any ST model)."""

    def __init__(self, model_name: str, query_prefix: str = "", doc_prefix: str = ""):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device=settings.embedding_device)
        self.name = model_name
        self.dim = self.model.get_sentence_embedding_dimension()
        self.qp, self.dp = query_prefix, doc_prefix

    def encode(self, texts, is_query=False):
        p = self.qp if is_query else self.dp
        v = self.model.encode([p + t for t in texts], batch_size=settings.embedding_batch_size,
                              normalize_embeddings=True, show_progress_bar=len(texts) > 64,
                              convert_to_numpy=True)
        return v.astype(np.float32)


_embedder: Embedder | None = None


def get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        be = settings.embedding_backend
        if be == "hash":
            _embedder = HashEmbedder()
        elif be == "specter2":
            # SPECTER2 base via ST (the adapter variants need the `adapters` lib; base is a solid default)
            _embedder = STEmbedder(settings.embedding_model if "specter" in settings.embedding_model
                                   else "allenai/specter2_base")
        else:
            _embedder = STEmbedder(settings.embedding_model)
    return _embedder


# ============================================================== the store
@dataclass
class Hit:
    chunk: Chunk
    score: float
    dense_rank: int | None = None
    bm25_rank: int | None = None
    rerank_score: float | None = None
    rerank_summary: str | None = None
    sources: set[str] = field(default_factory=set)


_STOP = set("""a an the of in on for to and or with by from is are was were be been as at that this these those
what which who how why when where does do did can could would should between across all any each report reported
compare comparison value values paper papers corpus show list give""".split())


def fts_query(q: str) -> str:
    toks = [t for t in re.findall(r"[A-Za-z0-9_]+", fts_normalize(q)) if t.lower() not in _STOP and len(t) > 1]
    return " OR ".join(f'"{t}"' for t in dict.fromkeys(toks))


class HybridStore:
    def __init__(self, kb, embedder: Embedder | None = None):
        self.kb = kb
        self._emb = embedder
        self._lock = threading.Lock()
        self._ids: list[str] = []
        self._M: np.ndarray | None = None
        self._loaded_n = -1

    @property
    def emb(self) -> Embedder:
        if self._emb is None:
            self._emb = get_embedder()
        return self._emb

    # -------------------------------------------------------------- index
    def index_missing(self, batch: int = 64, progress=None) -> int:
        model = self.emb.name
        rows = self.kb.conn.execute(
            "SELECT c.* FROM chunks c LEFT JOIN chunk_vectors v ON v.chunk_id=c.chunk_id AND v.model=? "
            "WHERE v.chunk_id IS NULL", (model,)).fetchall()
        chunks = [self.kb.row_to_chunk(r) for r in rows]
        for i in range(0, len(chunks), batch):
            part = chunks[i:i + batch]
            V = self.emb.encode([c.embed_text() for c in part]).astype(np.float16)
            with self.kb.tx() as c:
                c.executemany("INSERT OR REPLACE INTO chunk_vectors (chunk_id, model, dim, vec) VALUES (?,?,?,?)",
                              [(ch.chunk_id, model, V.shape[1], V[j].tobytes()) for j, ch in enumerate(part)])
            if progress:
                progress(min(i + batch, len(chunks)), len(chunks), "")
        self._loaded_n = -1
        return len(chunks)

    def _load(self) -> None:
        model = self.emb.name
        n = self.kb.conn.execute("SELECT COUNT(*) FROM chunk_vectors WHERE model=?", (model,)).fetchone()[0]
        if n == self._loaded_n:
            return
        with self._lock:
            rows = self.kb.conn.execute("SELECT chunk_id, dim, vec FROM chunk_vectors WHERE model=?", (model,)).fetchall()
            self._ids = [r[0] for r in rows]
            self._M = (np.stack([np.frombuffer(r[2], dtype=np.float16) for r in rows]).astype(np.float32)
                       if rows else np.zeros((0, self.emb.dim), np.float32))
            self._loaded_n = n

    # ------------------------------------------------------------- search
    def dense(self, query: str, k: int) -> list[tuple[str, float]]:
        self._load()
        if self._M is None or not len(self._ids):
            return []
        q = self.emb.encode([query], is_query=True)[0]
        s = self._M @ q
        k = min(k, len(s))
        idx = np.argpartition(-s, k - 1)[:k]
        idx = idx[np.argsort(-s[idx])]
        return [(self._ids[i], float(s[i])) for i in idx]

    def bm25(self, query: str, k: int) -> list[tuple[str, float]]:
        fq = fts_query(query)
        if not fq:
            return []
        try:
            rows = self.kb.conn.execute(
                "SELECT chunk_id, bm25(chunks_fts, 1.0, 0.5, 0.3) AS s FROM chunks_fts WHERE chunks_fts MATCH ? "
                "ORDER BY s LIMIT ?", (fq, k)).fetchall()
        except Exception as e:
            log.warning("FTS query failed (%s): %s", fq, e)
            return []
        return [(r[0], -float(r[1])) for r in rows]

    def search(self, query: str, k: int | None = None, *, nodes: list[str] | None = None,
               priorities: list[str] | None = None, include_private: bool = False,
               year_min: int | None = None, year_max: int | None = None,
               paper_ids: list[str] | None = None, kinds: list[str] | None = None) -> list[Hit]:
        k = k or settings.retrieve_k_per_query
        pool = max(k * 4, 80)
        try:
            d = self.dense(query, pool)
        except Exception as e:  # embedding model unavailable -> BM25 only
            log.warning("dense search unavailable: %s", e)
            d = []
        b = self.bm25(query, pool)
        fused: dict[str, Hit] = {}
        ids = list(dict.fromkeys([i for i, _ in d] + [i for i, _ in b]))
        chunks = self.kb.chunks_by_ids(ids)
        for rank, (cid, _) in enumerate(d):
            if cid in chunks:
                h = fused.setdefault(cid, Hit(chunks[cid], 0.0))
                h.dense_rank = rank
                h.score += settings.dense_weight / (settings.rrf_k + rank + 1)
                h.sources.add("dense")
        for rank, (cid, _) in enumerate(b):
            if cid in chunks:
                h = fused.setdefault(cid, Hit(chunks[cid], 0.0))
                h.bm25_rank = rank
                h.score += settings.bm25_weight / (settings.rrf_k + rank + 1)
                h.sources.add("bm25")
        out = []
        for h in fused.values():
            c = h.chunk
            if c.is_private and not include_private:
                continue
            if nodes and c.node not in nodes:
                continue
            if priorities and c.priority not in priorities:
                continue
            if year_min and (c.year or 0) < year_min:
                continue
            if year_max and (c.year or 9999) > year_max:
                continue
            if paper_ids and c.paper_id not in paper_ids:
                continue
            if kinds and c.kind not in kinds:
                continue
            h.score *= settings.priority_weights.get(c.priority, 0.6)
            if c.kind in ("table", "abstract"):
                h.score *= 1.05  # dense quantitative content
            out.append(h)
        out.sort(key=lambda h: -h.score)
        return out[:k]
