"""Core data models shared across modules."""
from __future__ import annotations

import hashlib
import re
from typing import Literal, Optional

from pydantic import BaseModel, Field

Status = Literal["candidate", "selected", "downloaded", "parsed", "indexed", "extracted", "rejected"]


def norm_doi(doi: str | None) -> str | None:
    if not doi:
        return None
    d = doi.strip().lower()
    d = re.sub(r"^(https?://(dx\.)?doi\.org/|doi:)", "", d)
    return d or None


def norm_arxiv(aid: str | None) -> str | None:
    if not aid:
        return None
    a = aid.strip()
    a = re.sub(r"^(https?://arxiv\.org/(abs|pdf)/|arxiv:)", "", a, flags=re.I)
    a = re.sub(r"v\d+$", "", a.removesuffix(".pdf"))
    return a or None


class Paper(BaseModel):
    paper_id: str = ""
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    openalex_id: Optional[str] = None
    s2_id: Optional[str] = None
    title: str
    authors: list[str] = Field(default_factory=list)
    year: Optional[int] = None
    venue: Optional[str] = None
    abstract: Optional[str] = None
    node: Optional[str] = None
    node_id: Optional[str] = None
    sub_node: Optional[str] = None
    priority: str = "background"
    matched_keywords: list[str] = Field(default_factory=list)
    relevance_score: float = 0.0
    source: str = "arxiv"                 # arxiv | openalex | s2 | zotero | private
    is_private: bool = False
    pdf_path: Optional[str] = None
    latex_path: Optional[str] = None
    status: Status = "candidate"
    is_review: bool = False
    mvp: bool = False

    def model_post_init(self, __ctx) -> None:
        self.doi = norm_doi(self.doi)
        self.arxiv_id = norm_arxiv(self.arxiv_id)
        if not self.paper_id:
            self.paper_id = make_paper_id(self)

    @property
    def first_author_surname(self) -> str:
        if not self.authors:
            return "Unknown"
        a = self.authors[0]
        if "," in a:
            return a.split(",")[0].strip()
        return a.split()[-1] if a.split() else a

    def citation_label(self) -> str:
        etal = " et al." if len(self.authors) > 1 else ""
        doi = self.doi or (f"arXiv:{self.arxiv_id}" if self.arxiv_id else "no-DOI")
        return f"{self.first_author_surname}{etal}, {self.year or 'n.d.'}, {doi}"


def make_paper_id(p: Paper) -> str:
    key = p.doi or (f"arxiv:{p.arxiv_id}" if p.arxiv_id else None) or (p.title.lower() + str(p.year))
    prefix = "priv" if p.is_private else "p"
    return f"{prefix}_{hashlib.sha1(key.encode('utf-8')).hexdigest()[:12]}"


class Chunk(BaseModel):
    chunk_id: str
    paper_id: str
    ord: int
    section: str = ""
    kind: Literal["text", "table", "equation", "abstract", "caption", "data"] = "text"
    text: str
    # inherited bibliographic metadata (denormalised for filtering & citation)
    doi: Optional[str] = None
    title: str = ""
    authors: list[str] = Field(default_factory=list)
    year: Optional[int] = None
    venue: Optional[str] = None
    node: Optional[str] = None
    sub_node: Optional[str] = None
    priority: str = "background"
    is_private: bool = False

    def embed_text(self) -> str:
        """Contextualised text for embedding (title + section header prepended)."""
        return f"{self.title} | {self.section}\n{self.text}"
