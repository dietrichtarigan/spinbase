"""Citation snowballing via OpenAlex (primary) and Semantic Scholar (fallback).

Backward = references of a seed (what a review builds on); forward = papers citing it.
Every discovered work is scored against the taxonomy and only kept if it is on-topic,
so the corpus stays high-signal instead of exploding along citation edges.
"""
from __future__ import annotations

import logging
from typing import Iterable

from config.settings import settings
from src.acquisition.arxiv_scraper import score_and_tag
from src.acquisition.csv_loader import Taxonomy
from src.core.http import GuardedClient, RateLimiter
from src.core.models import Paper, norm_arxiv, norm_doi
from src.core.privacy import assert_public

log = logging.getLogger(__name__)
OA = "https://api.openalex.org"
S2 = "https://api.semanticscholar.org/graph/v1"
S2_FIELDS = "title,authors,year,venue,externalIds,abstract,publicationTypes"

_oa = GuardedClient(RateLimiter("openalex", settings.openalex_min_interval_s), timeout=60)
_s2 = GuardedClient(
    RateLimiter("s2", settings.s2_min_interval_s), timeout=60,
    headers={"x-api-key": settings.semantic_scholar_api_key.get_secret_value()}
    if settings.semantic_scholar_api_key else None)


def _oa_params(**kw) -> dict:
    if settings.openalex_mailto:
        kw["mailto"] = settings.openalex_mailto
    return kw


def _abstract_from_inverted(inv: dict | None) -> str | None:
    if not inv:
        return None
    pos = {}
    for word, idxs in inv.items():
        for i in idxs:
            pos[i] = word
    return " ".join(pos[i] for i in sorted(pos))


def oa_work_to_paper(w: dict) -> Paper:
    arxiv_id = None
    for loc in w.get("locations") or []:
        url = (loc or {}).get("landing_page_url") or ""
        if "arxiv.org/abs/" in url:
            arxiv_id = norm_arxiv(url.split("/abs/")[-1])
            break
    doi = norm_doi(w.get("doi"))
    if doi and doi.startswith("10.48550/arxiv."):
        arxiv_id = arxiv_id or doi.split("arxiv.", 1)[1]
        doi = None  # arXiv DataCite DOI is not the published-version DOI
    src = ((w.get("primary_location") or {}).get("source") or {})
    return Paper(
        doi=doi, arxiv_id=arxiv_id, openalex_id=(w.get("id") or "").rsplit("/", 1)[-1] or None,
        title=w.get("title") or w.get("display_name") or "(untitled)",
        authors=[(a.get("author") or {}).get("display_name", "") for a in w.get("authorships") or []],
        year=w.get("publication_year"), venue=src.get("display_name"),
        abstract=_abstract_from_inverted(w.get("abstract_inverted_index")),
        source="openalex", is_review=(w.get("type") == "review"),
    )


def oa_resolve(p: Paper) -> dict | None:
    """Find the OpenAlex work for a paper by DOI, arXiv DOI, or title search."""
    assert_public([p])
    tries = []
    if p.openalex_id:
        tries.append(f"{OA}/works/{p.openalex_id}")
    if p.doi:
        tries.append(f"{OA}/works/doi:{p.doi}")
    if p.arxiv_id:
        tries.append(f"{OA}/works/doi:10.48550/arxiv.{p.arxiv_id}")
    for url in tries:
        r = _oa.get(url, params=_oa_params())
        if r.status_code == 200:
            return r.json()
    r = _oa.get(f"{OA}/works", params=_oa_params(search=p.title[:200], **{"per-page": 1}))
    if r.status_code == 200 and r.json().get("results"):
        from rapidfuzz import fuzz
        w = r.json()["results"][0]
        if fuzz.token_sort_ratio((w.get("title") or "").lower(), p.title.lower()) >= 90:
            return w
    return None


def oa_backward(work: dict) -> list[Paper]:
    ids = [i.rsplit("/", 1)[-1] for i in work.get("referenced_works") or []]
    out = []
    for i in range(0, len(ids), 50):
        r = _oa.get(f"{OA}/works", params=_oa_params(filter="openalex_id:" + "|".join(ids[i:i + 50]),
                                                     **{"per-page": 50}))
        if r.status_code == 200:
            out += [oa_work_to_paper(w) for w in r.json().get("results", [])]
    return out


def oa_forward(work: dict, max_results: int = 400) -> list[Paper]:
    wid = (work.get("id") or "").rsplit("/", 1)[-1]
    out, cursor = [], "*"
    while cursor and len(out) < max_results:
        r = _oa.get(f"{OA}/works", params=_oa_params(filter=f"cites:{wid}", cursor=cursor,
                                                     sort="cited_by_count:desc", **{"per-page": 100}))
        if r.status_code != 200:
            break
        j = r.json()
        out += [oa_work_to_paper(w) for w in j.get("results", [])]
        cursor = (j.get("meta") or {}).get("next_cursor")
        if not j.get("results"):
            break
    return out[:max_results]


# ------------------------------------------------------- Semantic Scholar
def _s2_to_paper(d: dict) -> Paper | None:
    if not d or not d.get("title"):
        return None
    ext = d.get("externalIds") or {}
    return Paper(doi=ext.get("DOI"), arxiv_id=ext.get("ArXiv"), s2_id=d.get("paperId"),
                 title=d["title"], authors=[a.get("name", "") for a in d.get("authors") or []],
                 year=d.get("year"), venue=d.get("venue") or None, abstract=d.get("abstract"),
                 source="s2", is_review="Review" in (d.get("publicationTypes") or []))


def s2_neighbors(p: Paper, direction: str, limit: int = 500) -> list[Paper]:
    assert_public([p])
    key = f"DOI:{p.doi}" if p.doi else (f"ARXIV:{p.arxiv_id}" if p.arxiv_id else None)
    if not key:
        return []
    edge = "references" if direction == "backward" else "citations"
    node = "citedPaper" if direction == "backward" else "citingPaper"
    r = _s2.get(f"{S2}/paper/{key}/{edge}", params={"fields": S2_FIELDS, "limit": min(limit, 1000)})
    if r.status_code != 200:
        return []
    return [x for x in (_s2_to_paper(e.get(node)) for e in r.json().get("data", [])) if x]


# ------------------------------------------------------------------ main
def snowball(kb, tax: Taxonomy, seeds: Iterable[Paper], directions=("backward", "forward"),
             min_score: float = 1.0, require_core_hit: bool = False, forward_limit: int = 300,
             progress=None) -> dict:
    """Expand from seed papers; register on-topic neighbours as candidates."""
    from src.ingestion.deduplicator import register_candidate

    stats = {"seeds": 0, "seen": 0, "kept": 0, "new": 0}
    seeds = list(seeds)
    for si, seed in enumerate(seeds):
        if seed.is_private:
            continue
        if progress:
            progress(si, len(seeds), seed.title[:60])
        neighbours: list[tuple[str, Paper]] = []
        try:
            work = oa_resolve(seed)
        except Exception as e:
            log.warning("OpenAlex resolve failed: %s", e)
            work = None
        if work:
            if not seed.openalex_id:
                seed.openalex_id = work["id"].rsplit("/", 1)[-1]
                kb.upsert_paper(seed)
            if "backward" in directions:
                neighbours += [("backward", x) for x in oa_backward(work)]
            if "forward" in directions:
                neighbours += [("forward", x) for x in oa_forward(work, forward_limit)]
        else:  # fallback
            for d in directions:
                neighbours += [(d, x) for x in s2_neighbors(seed, d)]
        stats["seeds"] += 1
        with kb.tx() as c:
            c.executemany("INSERT OR IGNORE INTO citations VALUES (?,?,?,?)",
                          [(seed.paper_id, n.doi or "", n.openalex_id or "", d) for d, n in neighbours])
        for d, n in neighbours:
            stats["seen"] += 1
            score_and_tag(n, tax)
            if n.relevance_score < min_score or (require_core_hit and n.priority != "core"):
                continue
            n.source = f"snowball-{d}"
            stats["kept"] += 1
            if register_candidate(kb, n) == "new":
                stats["new"] += 1
    return stats


def oa_find_arxiv(p: Paper) -> str | None:
    """For a published paper with only a DOI, find its arXiv preprint (to download open text)."""
    w = oa_resolve(p)
    return oa_work_to_paper(w).arxiv_id if w else None
