"""arXiv API harvester + downloader.

Strictly follows the arXiv API Terms of Use: **one request every 3 seconds, single
connection** (shared persisted limiter covers API queries *and* PDF/e-print downloads).

Improvisation: for arXiv papers we also fetch the LaTeX source (``/e-print``), which
gives perfect equations and tables without any PDF-to-LaTeX guesswork.
"""
from __future__ import annotations

import gzip
import io
import logging
import tarfile
import xml.etree.ElementTree as ET
from pathlib import Path

from config.settings import settings
from src.acquisition.csv_loader import Keyword, Taxonomy
from src.core.http import GuardedClient, RateLimiter
from src.core.models import Paper, norm_arxiv
from src.core.privacy import assert_public

log = logging.getLogger(__name__)

API = "https://export.arxiv.org/api/query"
NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom",
      "os": "http://a9.com/-/spec/opensearch/1.1/"}

_limiter = RateLimiter("arxiv", settings.arxiv_min_interval_s)
_client: GuardedClient | None = None


def client() -> GuardedClient:
    global _client
    if _client is None:
        _client = GuardedClient(_limiter, timeout=90)
    return _client


def build_query(kw: Keyword, categories: list[str] | None = None) -> str:
    term = kw.keyword.strip().replace('"', "")
    q = f'all:"{term}"' if " " in term or "-" in term else f"all:{term}"
    if categories:
        q = f"({q}) AND (" + " OR ".join(f"cat:{c}" for c in categories) + ")"
    return q


def parse_feed(xml_text: str) -> tuple[list[Paper], int]:
    root = ET.fromstring(xml_text)
    total = int(root.findtext("os:totalResults", "0", NS) or 0)
    out = []
    for e in root.findall("a:entry", NS):
        aid = norm_arxiv((e.findtext("a:id", "", NS) or "").rsplit("/abs/", 1)[-1])
        if not aid:
            continue
        title = " ".join((e.findtext("a:title", "", NS) or "").split())
        if title.lower() == "error":
            continue
        authors = [" ".join((a.findtext("a:name", "", NS) or "").split()) for a in e.findall("a:author", NS)]
        published = e.findtext("a:published", "", NS) or ""
        out.append(Paper(
            arxiv_id=aid, title=title, authors=authors,
            year=int(published[:4]) if published[:4].isdigit() else None,
            abstract=" ".join((e.findtext("a:summary", "", NS) or "").split()),
            doi=e.findtext("arxiv:doi", None, NS),
            venue=(e.findtext("arxiv:journal_ref", None, NS) or "arXiv").strip(),
            source="arxiv",
        ))
    return out, total


def search(query: str, max_results: int, sort_by: str = "relevance") -> list[Paper]:
    assert_public(text=query)
    results: list[Paper] = []
    start = 0
    while start < max_results:
        n = min(settings.arxiv_page_size, max_results - start)
        r = client().get(API, params={"search_query": query, "start": start, "max_results": n,
                                      "sortBy": sort_by, "sortOrder": "descending"})
        r.raise_for_status()
        page, total = parse_feed(r.text)
        results += page
        start += n
        if not page or start >= total:
            break
    return results


def harvest(tax: Taxonomy, kb, priorities: tuple[str, ...] = ("core", "supporting", "background"),
            categories: list[str] | None = None, nodes: list[str] | None = None,
            progress=None, refresh: bool = False) -> dict:
    """Query arXiv for every keyword; score hits against the full taxonomy; register candidates."""
    from src.ingestion.deduplicator import register_candidate

    stats = {"queries": 0, "hits": 0, "new": 0, "merged": 0, "skipped_done": 0, "failed": 0}
    with kb.tx() as c:
        c.execute("CREATE TABLE IF NOT EXISTS harvest_log (keyword TEXT PRIMARY KEY, priority TEXT, "
                  "n_results INTEGER, ts TEXT DEFAULT (datetime('now')))")
    done = set() if refresh else {r[0] for r in kb.conn.execute("SELECT keyword FROM harvest_log")}
    kws = [k for k in tax.keywords if k.priority in priorities and (not nodes or k.node in nodes)]
    for i, kw in enumerate(kws):
        if progress:
            progress(i, len(kws), kw.keyword)
        if kw.keyword in done:
            stats["skipped_done"] += 1
            continue
        n = settings.arxiv_max_results_per_keyword.get(kw.priority, 5)
        try:
            papers = search(build_query(kw, categories), n)
        except Exception as e:  # keep harvesting other keywords; rerun later retries only failures
            log.warning("arXiv query failed for %s: %s", kw.keyword, e)
            stats["failed"] += 1
            continue
        with kb.tx() as c:
            c.execute("INSERT OR REPLACE INTO harvest_log (keyword, priority, n_results) VALUES (?,?,?)",
                      (kw.keyword, kw.priority, len(papers)))
        stats["queries"] += 1
        for p in papers:
            score_and_tag(p, tax, seed=kw)
            stats["hits"] += 1
            stats["new" if register_candidate(kb, p) == "new" else "merged"] += 1
    return stats


def score_and_tag(p: Paper, tax: Taxonomy, seed: Keyword | None = None) -> Paper:
    """Assign node/sub_node/priority from taxonomy hits (title weighted x2)."""
    m_title = tax.score_text(p.title)
    m_all = tax.score_text(f"{p.title}. {p.abstract or ''}")
    p.relevance_score = round(m_all.score + m_title.score, 3)
    hits = {k.keyword for k in m_all.hits}
    if seed:
        hits.add(seed.keyword)
    p.matched_keywords = sorted(hits)
    best = m_all.best or seed
    if best:
        p.node, p.node_id, p.sub_node = best.node, best.node_id, best.sub_node
    p.priority = m_all.priority if m_all.hits else (seed.priority if seed else "background")
    low = (p.title + " " + (p.abstract or "")).lower()
    p.is_review = any(w in low for w in ("review", "perspective", "progress in", "roadmap"))
    return p


# ---------------------------------------------------------------- downloads
def download_pdf(p: Paper, dest_dir: Path | None = None) -> Path | None:
    if not p.arxiv_id:
        return None
    dest = (dest_dir or settings.raw_pdfs_dir) / f"{p.paper_id}.pdf"
    if dest.exists() and dest.stat().st_size > 10_000:
        return dest
    r = client().get(f"https://arxiv.org/pdf/{p.arxiv_id}")
    if r.status_code != 200 or not r.content.startswith(b"%PDF"):
        log.warning("PDF download failed for %s (%s)", p.arxiv_id, r.status_code)
        return None
    dest.write_bytes(r.content)
    return dest


_MAX_SRC_BYTES = 60_000_000
_KEEP_SUFFIXES = {".tex", ".bbl", ".ltx"}


def download_latex(p: Paper) -> Path | None:
    """Fetch the e-print and safely extract only .tex/.bbl text files."""
    if not p.arxiv_id:
        return None
    out = settings.latex_src_dir / p.paper_id
    if out.exists() and any(out.rglob("*.tex")):
        return out
    r = client().get(f"https://arxiv.org/e-print/{p.arxiv_id}")
    if r.status_code != 200 or len(r.content) > _MAX_SRC_BYTES:
        return None
    data = r.content
    out.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as tf:
            members = [m for m in tf.getmembers()
                       if m.isfile() and Path(m.name).suffix.lower() in _KEEP_SUFFIXES and m.size < 20_000_000]
            tf.extractall(out, members=members, filter="data")
    except tarfile.TarError:
        try:  # single gzipped .tex file
            txt = gzip.decompress(data) if data[:2] == b"\x1f\x8b" else data
            if b"\\documentclass" in txt[:200_000] or b"\\begin{document}" in txt:
                (out / "main.tex").write_bytes(txt)
        except OSError:
            pass
    if not any(out.rglob("*.tex")):
        return None  # PDF-only submission
    return out
