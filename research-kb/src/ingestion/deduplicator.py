"""Deduplication of preprint vs. published versions.

Match order: DOI -> arXiv id -> OpenAlex/S2 id -> fuzzy (normalised title >= 93 and
first-author surname match and |year diff| <= 2). When an arXiv preprint and its
journal version meet, the record keeps the journal DOI/venue/year (citable version)
and the arXiv id (open full text), and the higher priority / union of keywords.
"""
from __future__ import annotations

import re
import unicodedata

from rapidfuzz import fuzz

from config.settings import PRIORITIES
from src.core.models import Paper

_STOP = re.compile(r"[^a-z0-9 ]+")


def normalize_title(t: str) -> str:
    t = unicodedata.normalize("NFKD", t or "").encode("ascii", "ignore").decode()
    t = re.sub(r"\$[^$]*\$", lambda m: re.sub(r"[_^{}\\$]", "", m.group()), t)  # Fe$_3$GaTe$_2$ -> Fe3GaTe2
    t = _STOP.sub(" ", t.lower())
    return " ".join(t.split())


def _surname(p: Paper) -> str:
    return normalize_title(p.first_author_surname)


def find_duplicate(kb, p: Paper) -> Paper | None:
    c = kb.conn
    priv = int(p.is_private)
    if p.doi:
        r = c.execute("SELECT * FROM papers WHERE doi=? AND is_private=?", (p.doi, priv)).fetchone()
        if r:
            return kb.row_to_paper(r)
    for col, val in (("arxiv_id", p.arxiv_id), ("openalex_id", p.openalex_id), ("s2_id", p.s2_id)):
        if val:
            r = c.execute(f"SELECT * FROM papers WHERE {col}=? AND is_private=?", (val, priv)).fetchone()
            if r:
                return kb.row_to_paper(r)
    tn = normalize_title(p.title)
    if len(tn) < 12:
        return None
    # candidate pre-filter on first significant title word to keep it cheap
    first = next((w for w in tn.split() if len(w) > 3), tn.split()[0])
    rows = c.execute("SELECT * FROM papers WHERE is_private=? AND title_norm LIKE ?",
                     (priv, f"%{first}%")).fetchall()
    best, best_s = None, 0.0
    for r in rows:
        s = fuzz.token_sort_ratio(tn, r["title_norm"] or "")
        if s > best_s:
            best, best_s = r, s
    if best is not None and best_s >= 93:
        q = kb.row_to_paper(best)
        same_author = not p.authors or not q.authors or _surname(p) == _surname(q)
        year_ok = not p.year or not q.year or abs(p.year - q.year) <= 2
        if same_author and year_ok:
            return q
    return None


def merge(old: Paper, new: Paper) -> Paper:
    m = old.model_copy()
    # prefer published (journal) metadata over arXiv
    published_new = bool(new.doi) and (new.venue or "").lower() not in ("", "arxiv")
    for f in ("doi", "arxiv_id", "openalex_id", "s2_id", "abstract", "venue", "year"):
        ov, nv = getattr(m, f), getattr(new, f)
        if nv and (not ov or (published_new and f in ("doi", "venue", "year"))):
            setattr(m, f, nv)
    if len(new.authors) > len(m.authors):
        m.authors = new.authors
    if PRIORITIES.index(new.priority) < PRIORITIES.index(m.priority):
        m.priority, m.node, m.node_id, m.sub_node = new.priority, new.node, new.node_id, new.sub_node
    m.node = m.node or new.node
    m.matched_keywords = sorted(set(m.matched_keywords) | set(new.matched_keywords))
    m.relevance_score = max(m.relevance_score, new.relevance_score)
    m.is_review = m.is_review or new.is_review
    m.mvp = m.mvp or new.mvp
    m.pdf_path = m.pdf_path or new.pdf_path
    if new.source == "zotero":
        m.source = "zotero+" + m.source if "zotero" not in m.source else m.source
    return m


def register_candidate(kb, p: Paper, force_status: str | None = None) -> str:
    """Insert or merge. Returns 'new' or 'merged'; sets ``p.paper_id`` to the stored id."""
    dup = find_duplicate(kb, p)
    if dup is None:
        if force_status:
            p.status = force_status
        # guard the unique DOI index (different id, same DOI across private/public)
        kb.upsert_paper(p)
        return "new"
    merged = merge(dup, p)
    if force_status and merged.status == "candidate":
        merged.status = force_status
    p.paper_id = merged.paper_id
    kb.upsert_paper(merged)
    return "merged"


def dedupe_existing(kb) -> int:
    """Second pass over the whole table (e.g. after snowballing added DOIs). Returns merges."""
    merges = 0
    papers = kb.papers("is_private=0 ORDER BY relevance_score DESC")
    alive = {p.paper_id for p in papers}
    for p in papers:
        if p.paper_id not in alive:
            continue
        with kb.tx() as c:
            c.execute("DELETE FROM papers WHERE paper_id=? AND status IN ('candidate','selected')", (p.paper_id,))
        dup = find_duplicate(kb, p)
        if dup and dup.paper_id != p.paper_id:
            kb.upsert_paper(merge(dup, p))
            alive.discard(p.paper_id)
            merges += 1
        else:
            kb.upsert_paper(p)
    return merges
