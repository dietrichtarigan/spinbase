"""Single-file SQLite store: papers, chunks (+FTS5/BM25), vectors, physics parameters.

One file keeps the KB portable and fully local. DuckDB can ``ATTACH`` it read-only
for heavier analytics (``INSTALL sqlite; ATTACH 'kb.sqlite' (TYPE sqlite)``).
"""
from __future__ import annotations

import json
import re
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator

from config.settings import settings
from src.core.models import Chunk, Paper

SCHEMA = r"""
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS papers (
  paper_id TEXT PRIMARY KEY,
  doi TEXT, arxiv_id TEXT, openalex_id TEXT, s2_id TEXT,
  title TEXT NOT NULL, title_norm TEXT,
  authors TEXT, year INTEGER, venue TEXT, abstract TEXT,
  node TEXT, node_id TEXT, sub_node TEXT, priority TEXT DEFAULT 'background',
  matched_keywords TEXT, relevance_score REAL DEFAULT 0,
  source TEXT, is_private INTEGER NOT NULL DEFAULT 0,
  pdf_path TEXT, latex_path TEXT, status TEXT DEFAULT 'candidate',
  is_review INTEGER DEFAULT 0, mvp INTEGER DEFAULT 0,
  extra TEXT, added_at TEXT DEFAULT (datetime('now'))
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_papers_doi ON papers(doi) WHERE doi IS NOT NULL;
CREATE INDEX IF NOT EXISTS ix_papers_arxiv ON papers(arxiv_id);
CREATE INDEX IF NOT EXISTS ix_papers_status ON papers(status);

-- citation graph from snowballing (paper_id may be a candidate not in corpus)
CREATE TABLE IF NOT EXISTS citations (
  src_paper_id TEXT, dst_doi TEXT, dst_openalex TEXT, direction TEXT,
  PRIMARY KEY (src_paper_id, dst_doi, dst_openalex, direction)
);

CREATE TABLE IF NOT EXISTS chunks (
  chunk_id TEXT PRIMARY KEY,
  paper_id TEXT NOT NULL REFERENCES papers(paper_id) ON DELETE CASCADE,
  ord INTEGER, section TEXT, kind TEXT, text TEXT NOT NULL,
  doi TEXT, title TEXT, authors TEXT, year INTEGER, venue TEXT,
  node TEXT, sub_node TEXT, priority TEXT, is_private INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS ix_chunks_paper ON chunks(paper_id);

-- standalone FTS5 (BM25) over *normalised* text: Fe$_3$GaTe$_2$ is indexed as fe3gate2
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
  text, section, title, chunk_id UNINDEXED,
  tokenize = "unicode61 remove_diacritics 2 tokenchars '_'"
);

CREATE TABLE IF NOT EXISTS chunk_vectors (
  chunk_id TEXT NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE,
  model TEXT NOT NULL, dim INTEGER NOT NULL, vec BLOB NOT NULL,
  PRIMARY KEY (chunk_id, model)
);

-- private-text fingerprints (hashed word shingles) used by the egress guard
CREATE TABLE IF NOT EXISTS private_fingerprints (
  h INTEGER PRIMARY KEY, paper_id TEXT
) WITHOUT ROWID;

-------------------------------------------------------------------- Layer 2
CREATE TABLE IF NOT EXISTS materials (
  material_id INTEGER PRIMARY KEY AUTOINCREMENT,
  formula_norm TEXT NOT NULL, composition_key TEXT NOT NULL,
  UNIQUE(formula_norm)
);
CREATE TABLE IF NOT EXISTS methods (
  method_id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT UNIQUE NOT NULL
);
CREATE TABLE IF NOT EXISTS measurements (
  measurement_id INTEGER PRIMARY KEY AUTOINCREMENT,
  paper_id TEXT NOT NULL REFERENCES papers(paper_id) ON DELETE CASCADE,
  material_id INTEGER NOT NULL REFERENCES materials(material_id),
  method_id INTEGER REFERENCES methods(method_id),
  formula_raw TEXT, stack TEXT, doping TEXT, sample_form TEXT, thickness_nm REAL,
  layers INTEGER, substrate TEXT,
  quantity TEXT NOT NULL, quantity_label TEXT,
  value REAL, value_min REAL, value_max REAL, uncertainty REAL, unit TEXT,
  value_norm REAL, value_min_norm REAL, value_max_norm REAL, unit_norm TEXT,
  temperature_K REAL, field_T REAL, method_detail TEXT, is_theoretical INTEGER,
  evidence_quote TEXT, quote_verified INTEGER, section TEXT, confidence REAL,
  is_private INTEGER NOT NULL DEFAULT 0, extractor TEXT,
  created_at TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS ix_meas_q ON measurements(quantity, material_id);

CREATE TABLE IF NOT EXISTS paper_findings (
  paper_id TEXT REFERENCES papers(paper_id) ON DELETE CASCADE, finding TEXT
);

-- LLM usage ledger (cost tracking)
CREATE TABLE IF NOT EXISTS llm_usage (
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT DEFAULT (datetime('now')),
  purpose TEXT, model TEXT, prompt_tokens INTEGER, cached_tokens INTEGER,
  completion_tokens INTEGER, cost_usd REAL
);

-- Phase-2 Knowledge Graph: Material -[HAS_PROPERTY]-> Property -[MEASURED_BY]-> Method -[REPORTED_IN]-> Paper
CREATE VIEW IF NOT EXISTS kg_edges AS
  SELECT 'material:'||m.formula_norm AS src, 'HAS_PROPERTY' AS rel, 'property:'||ms.quantity AS dst,
         ms.measurement_id, ms.paper_id FROM measurements ms JOIN materials m USING(material_id)
  UNION ALL
  SELECT 'property:'||ms.quantity, 'MEASURED_BY', 'method:'||COALESCE(me.name,'other'),
         ms.measurement_id, ms.paper_id FROM measurements ms LEFT JOIN methods me USING(method_id)
  UNION ALL
  SELECT 'measurement:'||ms.measurement_id, 'REPORTED_IN', 'paper:'||ms.paper_id,
         ms.measurement_id, ms.paper_id FROM measurements ms;
"""

_MATH = re.compile(r"\$([^$]{1,120})\$")


def _demath(m: re.Match) -> str:
    t = re.sub(r"\\(?:text|mathrm|rm|mathit|textrm)\s*\{([^}]*)\}", r"\1", m.group(1))
    t = re.sub(r"[_^]\{?([^{}\s]*)\}?", r"\1", t)
    t = re.sub(r"[{}\\\s]", "", t)
    # only collapse "formula-like" math; leave real equations readable
    return t if re.fullmatch(r"[A-Za-z0-9.]+", t) else " " + m.group(1) + " "


def fts_normalize(text: str) -> str:
    """Make LaTeX-y formulas lexically searchable: 'Fe$_3$GaTe$_2$' -> 'Fe3GaTe2'."""
    t = _MATH.sub(_demath, text or "")
    t = re.sub(r"\b([A-Z][a-z]?)_\{?(\d+(?:\.\d+)?)\}?", r"\1\2", t)  # Fe_3GaTe_2 outside $...$
    # PDF text often splits subscripts: "Fe 3 GaTe 2" -> "Fe3GaTe2" (only element-symbol runs)
    t = _SPLIT_FORMULA.sub(lambda m: re.sub(r"\s+", "", m.group(0)), t)
    return t


_EL = r"(?:[A-Z][a-z]?)+"
_SPLIT_FORMULA = re.compile(rf"\b{_EL}\s\d{{1,2}}(?:\.\d+)?(?:\s{_EL}(?:\s\d{{1,2}}(?:\.\d+)?)?)+\b")


_PAPER_COLS = ["paper_id", "doi", "arxiv_id", "openalex_id", "s2_id", "title", "title_norm", "authors",
               "year", "venue", "abstract", "node", "node_id", "sub_node", "priority",
               "matched_keywords", "relevance_score", "source", "is_private", "pdf_path",
               "latex_path", "status", "is_review", "mvp"]


class KB:
    """Thin DAO around the SQLite file. Thread-safe via per-thread connections."""

    def __init__(self, path: Path | None = None):
        self.path = Path(path or settings.db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        with self.tx() as c:
            c.executescript(SCHEMA)

    @property
    def conn(self) -> sqlite3.Connection:
        c = getattr(self._local, "conn", None)
        if c is None:
            c = sqlite3.connect(self.path, timeout=30)
            c.row_factory = sqlite3.Row
            c.execute("PRAGMA foreign_keys=ON")
            self._local.conn = c
        return c

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        c = self.conn
        try:
            yield c
            c.commit()
        except Exception:
            c.rollback()
            raise

    # ------------------------------------------------------------ papers
    @staticmethod
    def _paper_row(p: Paper) -> dict:
        from src.ingestion.deduplicator import normalize_title
        d = p.model_dump()
        d["authors"] = json.dumps(p.authors, ensure_ascii=False)
        d["matched_keywords"] = json.dumps(p.matched_keywords, ensure_ascii=False)
        d["title_norm"] = normalize_title(p.title)
        d["is_private"] = int(p.is_private)
        d["is_review"] = int(p.is_review)
        d["mvp"] = int(p.mvp)
        return {k: d.get(k) for k in _PAPER_COLS}

    def upsert_paper(self, p: Paper) -> None:
        row = self._paper_row(p)
        cols = ",".join(row)
        qs = ",".join("?" for _ in row)
        upd = ",".join(f"{k}=excluded.{k}" for k in row if k != "paper_id")
        with self.tx() as c:
            c.execute(f"INSERT INTO papers ({cols}) VALUES ({qs}) ON CONFLICT(paper_id) DO UPDATE SET {upd}",
                      list(row.values()))

    @staticmethod
    def row_to_paper(r: sqlite3.Row) -> Paper:
        d = dict(r)
        d["authors"] = json.loads(d.get("authors") or "[]")
        d["matched_keywords"] = json.loads(d.get("matched_keywords") or "[]")
        for b in ("is_private", "is_review", "mvp"):
            d[b] = bool(d.get(b))
        d = {k: v for k, v in d.items() if k in Paper.model_fields}
        return Paper(**d)

    def get_paper(self, paper_id: str) -> Paper | None:
        r = self.conn.execute("SELECT * FROM papers WHERE paper_id=?", (paper_id,)).fetchone()
        return self.row_to_paper(r) if r else None

    def papers(self, where: str = "1=1", params: Iterable = ()) -> list[Paper]:
        rows = self.conn.execute(f"SELECT * FROM papers WHERE {where}", tuple(params)).fetchall()
        return [self.row_to_paper(r) for r in rows]

    def set_status(self, paper_id: str, status: str, **fields) -> None:
        sets = ", ".join(["status=?"] + [f"{k}=?" for k in fields])
        with self.tx() as c:
            c.execute(f"UPDATE papers SET {sets} WHERE paper_id=?", (status, *fields.values(), paper_id))

    # ------------------------------------------------------------ chunks
    def replace_chunks(self, paper_id: str, chunks: list[Chunk]) -> None:
        with self.tx() as c:
            c.execute("DELETE FROM chunks_fts WHERE chunk_id IN (SELECT chunk_id FROM chunks WHERE paper_id=?)",
                      (paper_id,))
            c.execute("DELETE FROM chunks WHERE paper_id=?", (paper_id,))
            c.executemany("INSERT INTO chunks_fts (text, section, title, chunk_id) VALUES (?,?,?,?)",
                          [(fts_normalize(k.text), k.section, fts_normalize(k.title), k.chunk_id) for k in chunks])
            c.executemany(
                "INSERT INTO chunks (chunk_id,paper_id,ord,section,kind,text,doi,title,authors,year,venue,"
                "node,sub_node,priority,is_private) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                [(k.chunk_id, k.paper_id, k.ord, k.section, k.kind, k.text, k.doi, k.title,
                  json.dumps(k.authors, ensure_ascii=False), k.year, k.venue, k.node, k.sub_node,
                  k.priority, int(k.is_private)) for k in chunks])

    @staticmethod
    def row_to_chunk(r: sqlite3.Row) -> Chunk:
        d = dict(r)
        d["authors"] = json.loads(d.get("authors") or "[]")
        d["is_private"] = bool(d["is_private"])
        return Chunk(**{k: v for k, v in d.items() if k in Chunk.model_fields})

    def chunks_by_ids(self, ids: list[str]) -> dict[str, Chunk]:
        if not ids:
            return {}
        out: dict[str, Chunk] = {}
        for i in range(0, len(ids), 500):
            part = ids[i:i + 500]
            q = ",".join("?" for _ in part)
            for r in self.conn.execute(f"SELECT * FROM chunks WHERE chunk_id IN ({q})", part):
                out[r["chunk_id"]] = self.row_to_chunk(r)
        return out

    def chunks_for_paper(self, paper_id: str) -> list[Chunk]:
        rows = self.conn.execute("SELECT * FROM chunks WHERE paper_id=? ORDER BY ord", (paper_id,))
        return [self.row_to_chunk(r) for r in rows]

    def stats(self) -> dict:
        c = self.conn
        q = lambda s: c.execute(s).fetchone()[0]
        return {
            "papers_total": q("SELECT COUNT(*) FROM papers"),
            "by_status": dict(c.execute("SELECT status, COUNT(*) FROM papers GROUP BY status").fetchall()),
            "by_priority_in_corpus": dict(c.execute(
                "SELECT priority, COUNT(*) FROM papers WHERE status IN ('selected','downloaded','parsed','indexed','extracted') GROUP BY priority").fetchall()),
            "private_docs": q("SELECT COUNT(*) FROM papers WHERE is_private=1"),
            "chunks": q("SELECT COUNT(*) FROM chunks"),
            "vectors": q("SELECT COUNT(*) FROM chunk_vectors"),
            "measurements": q("SELECT COUNT(*) FROM measurements"),
            "llm_cost_usd": round(q("SELECT COALESCE(SUM(cost_usd),0) FROM llm_usage"), 4),
        }


_kb: KB | None = None


def get_kb() -> KB:
    global _kb
    if _kb is None:
        _kb = KB()
    return _kb
