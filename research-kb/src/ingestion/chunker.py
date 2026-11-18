"""Structure-aware chunking with full metadata inheritance.

Rules
-----
* Never cross a section boundary; the section heading travels with every chunk.
* Equations stay attached to the paragraph that introduces them (they are
  meaningless alone) and are never split.
* Tables (and figure captions) become their own chunks: they carry most of the
  quantitative parameters and are prime targets for both BM25 and extraction.
* Paragraph packing to ~``chunk_target_tokens`` with a 1-sentence overlap.
"""
from __future__ import annotations

import hashlib
import re

from config.settings import settings
from src.core.models import Chunk, Paper
from src.ingestion.pdf_parser import ParsedDocument

_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")
_SKIP_SECTIONS = re.compile(r"^(acknowledg|author contributions|competing interests|data availability|"
                            r"funding|references|bibliography|supplementary)", re.I)


def n_tokens(text: str) -> int:
    # cheap approximation good enough for packing (≈ 1.3 tokens / word for scientific English)
    return int(len(text.split()) * 1.3) + text.count("\\") // 2


def _split_long(text: str, max_tok: int) -> list[str]:
    if n_tokens(text) <= max_tok:
        return [text]
    sents, out, cur = _SENT.split(text), [], []
    for s in sents:
        if cur and n_tokens(" ".join(cur + [s])) > max_tok:
            out.append(" ".join(cur))
            cur = []
        cur.append(s)
    if cur:
        out.append(" ".join(cur))
    return out


def chunk_document(doc: ParsedDocument, paper: Paper) -> list[Chunk]:
    target, mx = settings.chunk_target_tokens, settings.chunk_max_tokens
    meta = dict(doi=paper.doi, title=paper.title, authors=paper.authors, year=paper.year,
                venue=paper.venue, node=paper.node, sub_node=paper.sub_node,
                priority=paper.priority, is_private=paper.is_private)
    chunks: list[Chunk] = []

    def emit(section: str, kind: str, text: str):
        text = text.strip()
        if len(text) < 40:
            return
        cid = hashlib.sha1(f"{paper.paper_id}|{len(chunks)}|{text[:200]}".encode()).hexdigest()[:16]
        chunks.append(Chunk(chunk_id=f"{paper.paper_id}:{cid}", paper_id=paper.paper_id, ord=len(chunks),
                            section=section, kind=kind, text=text, **meta))

    if doc.abstract:
        emit("Abstract", "abstract", doc.abstract)

    for sec in doc.sections:
        if _SKIP_SECTIONS.match(sec.heading.strip()):
            continue
        # group paragraph + following equations into units
        units: list[str] = []
        for b in sec.blocks:
            if b.kind in ("table", "caption"):
                emit(sec.heading, "table" if b.kind == "table" else "caption", b.text)
                continue
            if b.kind == "equation":
                eq = f"$$ {b.text} $$"
                if units:
                    units[-1] = f"{units[-1]}\n{eq}"
                else:
                    units.append(eq)
                continue
            units.extend(_split_long(b.text, mx))
        buf: list[str] = []
        for u in units:
            if buf and n_tokens("\n".join(buf + [u])) > target:
                emit(sec.heading, "text", "\n".join(buf))
                tail = _SENT.split(buf[-1])[-settings.chunk_overlap_sentences:] if settings.chunk_overlap_sentences else []
                tail = [t for t in tail if "$$" not in t]
                buf = tail if n_tokens(" ".join(tail)) < target // 3 else []
            buf.append(u)
        if buf:
            emit(sec.heading, "text", "\n".join(buf))
    return chunks
