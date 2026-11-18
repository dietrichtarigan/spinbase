"""Import a Zotero library export (BibTeX with files, or CSL-JSON) into the KB.

How to export from Zotero
-------------------------
Right-click collection -> *Export Collection...* -> **BibTeX** (tick *Export Files*,
and *Export Notes* if you want) -> produces ``Library.bib`` + ``files/`` folder.
Better BibTeX's *Better CSL JSON* also works (tags come through as ``keyword``).

Tag -> taxonomy mapping (first match wins)
------------------------------------------
* ``node:<node name | node_id>`` / ``sub:<sub_node>`` / ``priority:<tier>`` explicit tags
* bare tags equal to a tier (``core``, ``supporting``, ``background``)
* bare tags matching a node / node_id / sub_node / keyword of the CSV
* ``private`` / ``unpublished`` / ``lab`` -> ``is_private=True`` (PDF copied to private_local)
* optional YAML ``data/keywords/zotero_tag_map.yaml``: ``{tag: {node: ..., priority: ...}}``
* otherwise inferred by scoring title+abstract against the taxonomy
"""
from __future__ import annotations

import json
import logging
import re
import shutil
from pathlib import Path

import yaml

from config.settings import PRIORITIES, settings
from src.acquisition.arxiv_scraper import score_and_tag
from src.acquisition.csv_loader import Taxonomy
from src.core.models import Paper

log = logging.getLogger(__name__)
PRIVATE_TAGS = {"private", "unpublished", "lab", "internal", "confidential"}


def _load_tag_map() -> dict:
    p = settings.keywords_dir / "zotero_tag_map.yaml"
    if p.exists():
        return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return {}


def _bib_files(field: str, base: Path) -> list[Path]:
    """Parse Zotero's BibTeX ``file`` field: 'label:path:mime;label2:path2:mime'."""
    if not field:
        return []
    s = field.replace("\\:", "\x00").replace("\\\\", "\\")
    out = []
    for part in s.split(";"):
        for seg in part.split(":"):
            seg = seg.replace("\x00", ":").strip()
            if seg.lower().endswith(".pdf"):
                p = Path(seg)
                out.append(p if p.is_absolute() else (base / p))
    return out


def _clean_tex(s: str | None) -> str | None:
    if s is None:
        return None
    s = re.sub(r"[{}]", "", s)
    s = s.replace("\\&", "&").replace("~", " ").replace("\\%", "%")
    return " ".join(s.split())


def parse_bibtex(path: Path) -> list[dict]:
    import bibtexparser
    from bibtexparser.bparser import BibTexParser
    parser = BibTexParser(common_strings=True)
    parser.ignore_nonstandard_types = False
    with open(path, encoding="utf-8") as f:
        db = bibtexparser.load(f, parser=parser)
    items = []
    for e in db.entries:
        authors = [_clean_tex(a) for a in re.split(r"\s+and\s+", e.get("author", "")) if a.strip()]
        yr = re.search(r"\d{4}", e.get("year", "") or e.get("date", ""))
        items.append({
            "title": _clean_tex(e.get("title")) or "(untitled)",
            "authors": authors, "year": int(yr.group()) if yr else None,
            "doi": e.get("doi"), "venue": _clean_tex(e.get("journal") or e.get("booktitle")),
            "abstract": _clean_tex(e.get("abstract")),
            "arxiv_id": e.get("eprint") if "arxiv" in (e.get("archiveprefix", "") + e.get("eprinttype", "")).lower() else None,
            "tags": [t.strip() for t in re.split(r"[,;]", e.get("keywords", "")) if t.strip()],
            "files": _bib_files(e.get("file", ""), path.parent),
        })
    return items


def parse_csl_json(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    items = []
    for e in data if isinstance(data, list) else data.get("items", []):
        authors = []
        for a in e.get("author", []):
            authors.append(", ".join(x for x in (a.get("family"), a.get("given")) if x) or a.get("literal", ""))
        dp = ((e.get("issued") or {}).get("date-parts") or [[None]])[0]
        kw = e.get("keyword") or ""
        tags = kw if isinstance(kw, list) else [t.strip() for t in re.split(r"[,;]", kw) if t.strip()]
        tags += [t.get("tag", t) if isinstance(t, dict) else t for t in e.get("tags", [])]
        files = [Path(f) for f in (e.get("file") or "").split(";") if f.lower().endswith(".pdf")]
        files = [f if f.is_absolute() else path.parent / f for f in files]
        items.append({
            "title": e.get("title") or "(untitled)", "authors": authors,
            "year": int(dp[0]) if dp and dp[0] else None, "doi": e.get("DOI"),
            "venue": e.get("container-title"), "abstract": e.get("abstract"),
            "arxiv_id": None, "tags": tags, "files": files,
        })
    return items


def map_tags(p: Paper, tags: list[str], tax: Taxonomy, tag_map: dict) -> Paper:
    node_set = prio_set = False
    for raw in tags:
        t = raw.strip()
        tl = t.lower()
        if tl in PRIVATE_TAGS:
            p.is_private = True
            continue
        if t in tag_map:
            m = tag_map[t] or {}
            if m.get("node"):
                k = tax.find_node(m["node"])
                p.node = k.node if k else m["node"]
                p.node_id = k.node_id if k else p.node_id
                node_set = True
            if m.get("priority") in PRIORITIES:
                p.priority, prio_set = m["priority"], True
            continue
        if ":" in tl:
            key, val = [x.strip() for x in t.split(":", 1)]
            key = key.lower()
            if key == "priority" and val.lower() in PRIORITIES:
                p.priority, prio_set = val.lower(), True
            elif key in ("node", "sub", "sub_node"):
                k = tax.find_node(val)
                if k:
                    p.node, p.node_id, node_set = k.node, k.node_id, True
                    if key != "node" or not p.sub_node:
                        p.sub_node = k.sub_node
            continue
        if tl in PRIORITIES:
            p.priority, prio_set = tl, True
            continue
        k = tax.find_node(t)
        if k and not node_set:
            p.node, p.node_id, p.sub_node, node_set = k.node, k.node_id, k.sub_node, True
            if not prio_set and k.keyword.lower() == tl:
                p.priority = k.priority
    if not node_set or not prio_set:
        node, nid, sub, prio = p.node, p.node_id, p.sub_node, p.priority
        score_and_tag(p, tax)
        if node_set:
            p.node, p.node_id, p.sub_node = node, nid, sub
        if prio_set:
            p.priority = prio
        elif p.relevance_score == 0:
            # manually curated -> at least 'supporting' unless the user says otherwise
            p.priority = "supporting"
    p.matched_keywords = sorted(set(p.matched_keywords) | {f"zotero:{t}" for t in tags})
    return p


def import_zotero(kb, tax: Taxonomy, export_path: str | Path, mvp: bool = False) -> dict:
    from src.ingestion.deduplicator import register_candidate

    export_path = Path(export_path)
    items = parse_csl_json(export_path) if export_path.suffix.lower() == ".json" else parse_bibtex(export_path)
    tag_map = _load_tag_map()
    stats = {"items": len(items), "new": 0, "merged": 0, "with_pdf": 0, "private": 0}
    for it in items:
        p = Paper(title=it["title"], authors=it["authors"], year=it["year"], doi=it["doi"],
                  venue=it["venue"], abstract=it["abstract"], arxiv_id=it["arxiv_id"], source="zotero",
                  mvp=mvp)
        map_tags(p, it["tags"], tax, tag_map)
        if p.is_private:
            p.paper_id = ""
            p.model_post_init(None)  # regenerate id with private prefix
            stats["private"] += 1
        pdf = next((f for f in it["files"] if f.exists()), None)
        res = register_candidate(kb, p, force_status="selected")
        stats["new" if res == "new" else "merged"] += 1
        stored = kb.get_paper(p.paper_id) or p
        if pdf:
            dest_dir = settings.private_dir / "zotero" if p.is_private else settings.raw_pdfs_dir
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / f"{stored.paper_id}.pdf"
            shutil.copy2(pdf, dest)
            kb.set_status(stored.paper_id, "downloaded", pdf_path=str(dest))
            stats["with_pdf"] += 1
    return stats
