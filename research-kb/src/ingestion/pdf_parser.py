"""Local, equation-preserving document parsing. Nothing here touches the network
except a *local* GROBID server.

Backends (tried in order, configurable via ``PARSER_ORDER``):

* ``latex``   – arXiv e-print source. Best possible fidelity: real LaTeX equations/tables.
* ``grobid``  – GROBID TEI (structure, header, references; formulas as text). Run with
                ``docker compose up grobid`` (see docker-compose.yml).
* ``marker``  – Marker PDF->Markdown with LaTeX math (pip install research-kb[marker]).
* ``pymupdf`` – always-available fallback (font-size heading detection, no LaTeX math).

All produce a :class:`ParsedDocument`, persisted as JSON + Markdown in ``data/parsed_md``.
"""
from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field

from config.settings import settings

log = logging.getLogger(__name__)

BlockKind = Literal["paragraph", "equation", "table", "caption", "list"]


class Block(BaseModel):
    kind: BlockKind = "paragraph"
    text: str


class Section(BaseModel):
    heading: str
    level: int = 1
    blocks: list[Block] = Field(default_factory=list)


class Reference(BaseModel):
    raw: str = ""
    title: Optional[str] = None
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    year: Optional[int] = None


class ParsedDocument(BaseModel):
    paper_id: str
    parser: str
    title: Optional[str] = None
    abstract: Optional[str] = None
    sections: list[Section] = Field(default_factory=list)
    references: list[Reference] = Field(default_factory=list)
    is_private: bool = False

    def to_markdown(self) -> str:
        out = [f"# {self.title or self.paper_id}", ""]
        if self.abstract:
            out += ["## Abstract", "", self.abstract, ""]
        for s in self.sections:
            out += ["#" * min(s.level + 1, 6) + " " + s.heading, ""]
            for b in s.blocks:
                if b.kind == "equation":
                    out += ["$$", b.text.strip(), "$$", ""]
                elif b.kind == "table":
                    out += [b.text, ""]
                elif b.kind == "caption":
                    out += [f"*{b.text}*", ""]
                else:
                    out += [b.text, ""]
        return "\n".join(out)

    def full_text(self) -> str:
        return "\n".join([self.abstract or ""] + [b.text for s in self.sections for b in s.blocks])

    def save(self, base: Path | None = None) -> Path:
        base = base or (settings.private_dir / "parsed" if self.is_private else settings.parsed_dir)
        base.mkdir(parents=True, exist_ok=True)
        (base / f"{self.paper_id}.json").write_text(self.model_dump_json(indent=1), encoding="utf-8")
        p = base / f"{self.paper_id}.md"
        p.write_text(self.to_markdown(), encoding="utf-8")
        return p

    @classmethod
    def load(cls, paper_id: str, private: bool = False) -> "ParsedDocument | None":
        base = settings.private_dir / "parsed" if private else settings.parsed_dir
        f = base / f"{paper_id}.json"
        return cls.model_validate_json(f.read_text(encoding="utf-8")) if f.exists() else None


# ===================================================================== LaTeX
_SEC = re.compile(r"\\(section|subsection|subsubsection)\*?\s*(?:\[[^\]]*\])?\s*\{", re.S)
_EQ_ENVS = "equation|equation\\*|align|align\\*|eqnarray|eqnarray\\*|gather|gather\\*|multline|multline\\*|displaymath"
_ENV_BLOCK = re.compile(
    rf"\\begin\{{(?P<env>{_EQ_ENVS}|table\*?|figure\*?|tabular)\}}(?P<body>.*?)\\end\{{(?P=env)\}}", re.S)
_DISPLAY = re.compile(r"\\\[(.*?)\\\]|\$\$(.*?)\$\$", re.S)


def _balanced(s: str, start: int) -> tuple[str, int]:
    """Return content of {...} starting right after an opening brace at s[start-1]."""
    depth, i = 1, start
    while i < len(s) and depth:
        if s[i] == "\\":
            i += 2
            continue
        depth += {"{": 1, "}": -1}.get(s[i], 0)
        i += 1
    return s[start:i - 1], i


def _strip_comments(s: str) -> str:
    return re.sub(r"(?<!\\)%.*", "", s)


def _inline_inputs(text: str, root: Path, depth: int = 0) -> str:
    if depth > 4:
        return text

    def rep(m):
        name = m.group(2).strip()
        for cand in (root / name, root / f"{name}.tex"):
            try:
                cand.resolve().relative_to(root.resolve())
            except ValueError:
                continue
            if cand.is_file():
                return _inline_inputs(_strip_comments(cand.read_text("utf-8", "ignore")), root, depth + 1)
        return ""
    return re.sub(r"\\(input|include)\s*\{([^}]+)\}", rep, text)


_TEXT_CMDS = re.compile(r"\\(?:emph|textbf|textit|textrm|mathrm|text|underline)\s*\{([^{}]*)\}")
_DROP_CMDS = re.compile(r"\\(?:label|ref|eqref|cite[tp]?\*?|citep|citet|footnote|vspace|hspace|"
                        r"includegraphics|bibliographystyle|bibliography|thanks)\*?(?:\[[^\]]*\])*\{[^{}]*\}")


def _clean_prose(s: str) -> str:
    s = _DROP_CMDS.sub(lambda m: "[ref]" if "ref" in m.group()[:5] else ("[cite]" if "cite" in m.group()[:6] else ""), s)
    for _ in range(2):
        s = _TEXT_CMDS.sub(r"\1", s)
    s = re.sub(r"\\(?:noindent|newpage|clearpage|centering|small|footnotesize|maketitle|par)\b", " ", s)
    s = s.replace("~", " ").replace("\\%", "%").replace("``", '"').replace("''", '"')
    return re.sub(r"[ \t]+", " ", s).strip()


def _latex_blocks(body: str) -> list[Block]:
    blocks: list[Block] = []
    pos = 0
    matches = sorted([(m.start(), m.end(), "env", m) for m in _ENV_BLOCK.finditer(body)] +
                     [(m.start(), m.end(), "disp", m) for m in _DISPLAY.finditer(body)])
    last_end = 0
    for st, en, typ, m in matches:
        if st < last_end:
            continue
        for para in re.split(r"\n\s*\n", body[pos:st]):
            if para.strip():
                blocks.append(Block(kind="paragraph", text=_clean_prose(para)))
        if typ == "disp":
            blocks.append(Block(kind="equation", text=(m.group(1) or m.group(2)).strip()))
        else:
            env, inner = m.group("env"), m.group("body")
            if env.startswith(("table", "tabular")):
                cap = re.search(r"\\caption\s*\{", inner)
                caption = _clean_prose(_balanced(inner, cap.end())[0]) if cap else ""
                inner_clean = re.sub(r"\\label\{[^}]*\}", "", inner).strip()
                blocks.append(Block(kind="table", text=(f"Table: {caption}\n" if caption else "") + inner_clean))
            elif env.startswith("figure"):
                cap = re.search(r"\\caption\s*\{", inner)
                if cap:
                    blocks.append(Block(kind="caption", text="Figure: " + _clean_prose(_balanced(inner, cap.end())[0])))
            else:
                eq = re.sub(r"\\label\{[^}]*\}", "", inner).strip()
                blocks.append(Block(kind="equation", text=eq))
        pos = last_end = en
    for para in re.split(r"\n\s*\n", body[pos:]):
        if para.strip():
            blocks.append(Block(kind="paragraph", text=_clean_prose(para)))
    return [b for b in blocks if len(b.text) > 2]


def parse_latex(paper_id: str, src_dir: Path) -> ParsedDocument:
    texs = sorted(src_dir.rglob("*.tex"), key=lambda p: -p.stat().st_size)
    main = next((t for t in texs if "\\documentclass" in t.read_text("utf-8", "ignore")[:20000]), texs[0])
    raw = _inline_inputs(_strip_comments(main.read_text("utf-8", "ignore")), main.parent)
    title = None
    tm = re.search(r"\\title\s*(?:\[[^\]]*\])?\s*\{", raw)
    if tm:
        title = _clean_prose(_balanced(raw, tm.end())[0]).replace("\\\\", " ")
    body_m = re.search(r"\\begin\{document\}(.*?)(\\end\{document\}|$)", raw, re.S)
    body = body_m.group(1) if body_m else raw
    abstract = None
    am = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", body, re.S)
    if am:
        abstract = _clean_prose(am.group(1))
        body = body[:am.start()] + body[am.end():]
    body = re.split(r"\\begin\{thebibliography\}|\\bibliography\{|\\section\*?\{\s*(?:Acknowledg|References)", body)[0]
    sections: list[Section] = []
    heads = list(_SEC.finditer(body))
    pre = body[:heads[0].start()] if heads else body
    pre = re.sub(r"\\maketitle|\\(author|affiliation|email|date|keywords)\s*(\[[^\]]*\])?\{[^{}]*\}", "", pre)
    if _latex_blocks(pre):
        sections.append(Section(heading="Front matter", blocks=_latex_blocks(pre)))
    for i, h in enumerate(heads):
        heading, after = _balanced(body, h.end())
        end = heads[i + 1].start() if i + 1 < len(heads) else len(body)
        level = {"section": 1, "subsection": 2, "subsubsection": 3}[h.group(1)]
        sections.append(Section(heading=_clean_prose(heading), level=level, blocks=_latex_blocks(body[after:end])))
    refs = []
    for bbl in src_dir.rglob("*.bbl"):
        for item in re.split(r"\\bibitem", bbl.read_text("utf-8", "ignore"))[1:]:
            doi = re.search(r"10\.\d{4,9}/[^\s}{,]+", item)
            refs.append(Reference(raw=_clean_prose(item)[:500], doi=doi.group().rstrip(".") if doi else None))
    return ParsedDocument(paper_id=paper_id, parser="latex", title=title, abstract=abstract,
                          sections=sections, references=refs)


# ==================================================================== GROBID
TEI = {"t": "http://www.tei-c.org/ns/1.0"}


def grobid_alive() -> bool:
    import httpx
    try:
        return httpx.get(f"{settings.grobid_url}/api/isalive", timeout=3).text.strip().lower() == "true"
    except Exception:
        return False


def _txt(el) -> str:
    return " ".join("".join(el.itertext()).split()) if el is not None else ""


def parse_grobid_tei(paper_id: str, tei_xml: str) -> ParsedDocument:
    root = ET.fromstring(tei_xml)
    title = _txt(root.find(".//t:titleStmt/t:title", TEI)) or None
    abstract = _txt(root.find(".//t:profileDesc/t:abstract", TEI)) or None
    sections: list[Section] = []
    body = root.find(".//t:text/t:body", TEI)
    if body is not None:
        for div in body.findall("t:div", TEI):
            head = div.find("t:head", TEI)
            sec = Section(heading=_txt(head) or "Body", level=1 if not (head is not None and "." in (head.get("n") or "").strip(".")) else 2)
            for child in div:
                tag = child.tag.split("}")[-1]
                if tag == "p":
                    sec.blocks.append(Block(kind="paragraph", text=_txt(child)))
                elif tag == "formula":
                    sec.blocks.append(Block(kind="equation", text=_txt(child)))
            sections.append(sec)
        for fig in body.iter(f"{{{TEI['t']}}}figure"):
            cap = _txt(fig.find("t:figDesc", TEI))
            if fig.get("type") == "table":
                rows = []
                for row in fig.findall(".//t:row", TEI):
                    rows.append("| " + " | ".join(_txt(c) for c in row.findall("t:cell", TEI)) + " |")
                if rows:
                    sep = "|" + "---|" * (rows[0].count("|") - 1)
                    rows.insert(1, sep)
                text = (f"Table: {_txt(fig.find('t:head', TEI))} {cap}".strip() + "\n" + "\n".join(rows)).strip()
                (sections[-1] if sections else sections.append(Section(heading="Tables")) or sections[-1]
                 ).blocks.append(Block(kind="table", text=text))
            elif cap:
                (sections[-1] if sections else sections.append(Section(heading="Figures")) or sections[-1]
                 ).blocks.append(Block(kind="caption", text=f"Figure: {cap}"))
    refs = []
    for b in root.findall(".//t:listBibl/t:biblStruct", TEI):
        doi = b.find(".//t:idno[@type='DOI']", TEI)
        arx = b.find(".//t:idno[@type='arXiv']", TEI)
        yr = b.find(".//t:date[@type='published']", TEI)
        refs.append(Reference(
            raw=_txt(b)[:500], title=_txt(b.find(".//t:title[@level='a']", TEI)) or None,
            doi=_txt(doi).lower() or None if doi is not None else None,
            arxiv_id=_txt(arx).replace("arXiv:", "") or None if arx is not None else None,
            year=int(yr.get("when")[:4]) if yr is not None and (yr.get("when") or "")[:4].isdigit() else None))
    return ParsedDocument(paper_id=paper_id, parser="grobid", title=title, abstract=abstract,
                          sections=sections, references=refs)


def parse_grobid(paper_id: str, pdf: Path) -> ParsedDocument:
    import httpx  # local server only; not routed through GuardedClient on purpose
    if not settings.grobid_url.startswith(("http://localhost", "http://127.0.0.1", "http://grobid")):
        raise RuntimeError("GROBID_URL must point to a local GROBID instance (privacy air-gap).")
    with open(pdf, "rb") as f:
        r = httpx.post(f"{settings.grobid_url}/api/processFulltextDocument",
                       files={"input": (pdf.name, f, "application/pdf")},
                       data={"consolidateHeader": "0", "consolidateCitations": "0",
                             "includeRawCitations": "1", "segmentSentences": "0"}, timeout=300)
    r.raise_for_status()
    return parse_grobid_tei(paper_id, r.text)


# ==================================================================== Marker
def marker_available() -> bool:
    return shutil.which("marker_single") is not None


def _md_to_doc(paper_id: str, md: str, parser: str) -> ParsedDocument:
    title, abstract, sections = None, None, []
    cur: Section | None = None
    lines = md.splitlines()
    buf: list[str] = []

    def flush():
        nonlocal buf
        txt = "\n".join(buf).strip()
        buf = []
        if not txt:
            return
        kind: BlockKind = "paragraph"
        if txt.startswith("$$") and txt.endswith("$$"):
            kind, txt = "equation", txt.strip("$").strip()
        elif txt.startswith("|"):
            kind = "table"
        if cur is not None:
            cur.blocks.append(Block(kind=kind, text=txt))

    for ln in lines:
        h = re.match(r"^(#{1,6})\s+(.*)", ln)
        if h:
            flush()
            text = re.sub(r"[*_]", "", h.group(2)).strip()
            if title is None and len(h.group(1)) == 1:
                title = text
                continue
            cur = Section(heading=text, level=max(1, len(h.group(1)) - 1))
            sections.append(cur)
        elif not ln.strip():
            flush()
        else:
            if cur is None:
                cur = Section(heading="Front matter")
                sections.append(cur)
            buf.append(ln)
    flush()
    for s in sections:
        if s.heading.lower().startswith("abstract"):
            abstract = " ".join(b.text for b in s.blocks)
    sections = [s for s in sections if not s.heading.lower().startswith(("abstract", "reference", "bibliograph"))]
    return ParsedDocument(paper_id=paper_id, parser=parser, title=title, abstract=abstract, sections=sections)


def parse_marker(paper_id: str, pdf: Path) -> ParsedDocument:
    with tempfile.TemporaryDirectory() as td:
        subprocess.run(["marker_single", str(pdf), "--output_dir", td, "--output_format", "markdown"],
                       check=True, capture_output=True, timeout=3600)
        md_files = list(Path(td).rglob("*.md"))
        if not md_files:
            raise RuntimeError("marker produced no markdown")
        return _md_to_doc(paper_id, md_files[0].read_text("utf-8"), "marker")


# =================================================================== PyMuPDF
def parse_pymupdf(paper_id: str, pdf: Path) -> ParsedDocument:
    import pymupdf as fitz
    doc = fitz.open(pdf)
    spans = []
    for page in doc:
        d = page.get_text("dict")
        for b in d["blocks"]:
            if b.get("type") != 0:
                continue
            txt, sizes, bold = [], [], False
            for line in b["lines"]:
                for sp in line["spans"]:
                    txt.append(sp["text"])
                    sizes.append(sp["size"])
                    bold |= bool(sp["flags"] & 16)
            t = " ".join(" ".join(txt).split())
            if t:
                spans.append((t, max(sizes) if sizes else 0, bold))
    if not spans:
        raise RuntimeError("no extractable text (scanned PDF? run OCR first)")
    from collections import Counter
    body_size = Counter(round(s) for _, s, _ in spans).most_common(1)[0][0]
    title = max(spans[:30], key=lambda x: x[1])[0] if spans else None
    sections, cur, abstract = [], Section(heading="Front matter"), None
    sections.append(cur)
    head_re = re.compile(r"^((\d+(\.\d+)*\.?)|[IVX]+\.)\s+[A-Z]|^(Abstract|Introduction|Methods?|Results|Discussion|Conclusions?|Acknowledg|References)\b", re.I)
    for t, size, bold in spans:
        is_head = len(t) < 120 and (size > body_size + 0.8 or (bold and head_re.match(t)))
        if is_head and t != title:
            cur = Section(heading=t)
            sections.append(cur)
            continue
        kind: BlockKind = "caption" if re.match(r"^(Fig\.?|Figure|Table)\s*\d", t) else "paragraph"
        cur.blocks.append(Block(kind=kind, text=t))
    for s in sections:
        if s.heading.lower().startswith("abstract"):
            abstract = " ".join(b.text for b in s.blocks)
    cut = next((i for i, s in enumerate(sections) if re.match(r"^(References|Bibliography)\b", s.heading, re.I)), None)
    sections = sections[:cut] if cut else sections
    sections = [s for s in sections if not s.heading.lower().startswith("abstract") and s.blocks]
    if abstract is None and sections and sections[0].blocks:
        # first long paragraph is usually the abstract in physics journals
        long = next((b for b in sections[0].blocks if len(b.text) > 400), None)
        abstract = long.text if long else None
    return ParsedDocument(paper_id=paper_id, parser="pymupdf", title=title, abstract=abstract, sections=sections)


# ================================================================ dispatcher
def parse_paper(paper, prefer: str | None = None) -> ParsedDocument:
    """Parse a Paper using the best available local backend."""
    order: list[str] = []
    if paper.latex_path and Path(paper.latex_path).exists():
        order.append("latex")
    choice = prefer or settings.pdf_parser
    order += settings.parser_order if choice == "auto" else [choice]
    pdf = Path(paper.pdf_path) if paper.pdf_path else None
    errors = []
    for be in order:
        try:
            if be == "latex":
                doc = parse_latex(paper.paper_id, Path(paper.latex_path))
            elif pdf is None or not pdf.exists():
                continue
            elif be == "grobid":
                if not grobid_alive():
                    continue
                doc = parse_grobid(paper.paper_id, pdf)
            elif be == "marker":
                if not marker_available():
                    continue
                doc = parse_marker(paper.paper_id, pdf)
            elif be == "pymupdf":
                doc = parse_pymupdf(paper.paper_id, pdf)
            else:
                continue
            if sum(len(b.text) for s in doc.sections for b in s.blocks) < 1500:
                errors.append(f"{be}: too little text")
                continue
            # GROBID is the best reference extractor; enrich other parsers' output with it
            if be != "grobid" and settings.grobid_for_references and pdf and pdf.exists() \
                    and not doc.references and grobid_alive():
                try:
                    doc.references = parse_grobid(paper.paper_id, pdf).references
                except Exception:
                    pass
            doc.title = paper.title or doc.title  # bibliographic title is authoritative
            doc.abstract = doc.abstract or paper.abstract
            doc.is_private = paper.is_private
            return doc
        except Exception as e:  # try next backend
            log.warning("parser %s failed on %s: %s", be, paper.paper_id, e)
            errors.append(f"{be}: {e}")
    raise RuntimeError(f"All parsers failed for {paper.paper_id}: {errors}")


def available_backends() -> dict[str, bool]:
    try:
        import pymupdf  # noqa
        pm = True
    except ImportError:
        pm = False
    return {"latex": True, "grobid": grobid_alive(), "marker": marker_available(), "pymupdf": pm}
