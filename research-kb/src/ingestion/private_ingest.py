"""Ingest unpublished local data (MOKE imaging notes, MuMax3 scripts/logs, lab notes).

Everything ingested here is ``is_private=True``:
* stored under ``data/private_local`` only,
* fingerprinted so the egress guard can recognise it in any outbound payload,
* never passed to DeepSeek — MuMax3 parameters are extracted by a local parser instead.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from config.schema_physics import Method, Quantity, normalize_value
from config.settings import settings
from src.core.models import Chunk, Paper
from src.core.privacy import PrivacyViolation, is_private_path, register_private_text

TEXT_SUFFIXES = {".md", ".txt", ".csv", ".tsv", ".json", ".mx3", ".log", ".yaml", ".yml", ".tex", ".org"}

# MuMax3 parameter name -> (Quantity, SI unit as written in mx3)
MX3_PARAMS = {
    "msat": (Quantity.saturation_magnetization, "A/m"),
    "aex": (Quantity.exchange_stiffness, "J/m"),
    "dind": (Quantity.dmi_interfacial, "J/m^2"),
    "dbulk": (Quantity.dmi_bulk, "J/m^2"),
    "ku1": (Quantity.uniaxial_anisotropy, "J/m^3"),
    "alpha": (Quantity.gilbert_damping, ""),
}
_MX3_ASSIGN = re.compile(r"^\s*(Msat|Aex|Dind|Dbulk|Ku1|alpha)\s*=\s*([-+0-9.eE]+)\s*(?://.*)?$", re.M | re.I)


def _read(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        import pymupdf as fitz
        return "\n".join(p.get_text() for p in fitz.open(path))
    return path.read_text("utf-8", "ignore")


def _chunks(text: str, paper: Paper, section: str) -> list[Chunk]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    out, buf = [], []
    for para in paras:
        buf.append(para)
        if len(" ".join(buf).split()) > 280:
            out.append("\n\n".join(buf))
            buf = []
    if buf:
        out.append("\n\n".join(buf))
    return [Chunk(chunk_id=f"{paper.paper_id}:{i:04d}", paper_id=paper.paper_id, ord=i, section=section,
                  kind="data", text=t, title=paper.title, authors=paper.authors, year=paper.year,
                  venue="private", node=paper.node, sub_node=paper.sub_node, priority=paper.priority,
                  is_private=True) for i, t in enumerate(out)]


def parse_mx3(text: str) -> list[dict]:
    recs = []
    for m in _MX3_ASSIGN.finditer(text):
        name, val = m.group(1).lower(), float(m.group(2))
        q, unit = MX3_PARAMS[name]
        vn, un = normalize_value(q, val, unit)
        recs.append(dict(quantity=q.value, quantity_label=m.group(1), value=val, unit=unit,
                         value_norm=vn, unit_norm=un, evidence_quote=m.group(0).strip()))
    return recs


def ingest_private_path(kb, path: str | Path, node: str | None = None, material: str | None = None,
                        priority: str = "core") -> dict:
    from src.extraction.param_extractor import store_measurement_rows

    root = Path(path).resolve()
    if not is_private_path(root):
        raise PrivacyViolation(
            f"Private data must live under {settings.private_dir}. Move it there first (it is git-ignored).")
    files = [root] if root.is_file() else [p for p in root.rglob("*")
                                           if p.is_file() and (p.suffix.lower() in TEXT_SUFFIXES or p.suffix.lower() == ".pdf")]
    stats = {"files": 0, "chunks": 0, "fingerprints": 0, "mx3_params": 0}
    for f in files:
        try:
            text = _read(f)
        except Exception:
            continue
        if not text.strip():
            continue
        rel = f.relative_to(settings.private_dir.resolve())
        paper = Paper(title=f"[PRIVATE] {rel.as_posix()}", authors=["Local Lab"], source="private",
                      is_private=True, node=node, priority=priority, pdf_path=str(f), status="indexed",
                      venue="private")
        paper.paper_id = "priv_" + re.sub(r"[^A-Za-z0-9]+", "_", rel.as_posix())[-60:]
        kb.upsert_paper(paper)
        ch = _chunks(text, paper, section=f.suffix.lstrip(".") or "file")
        kb.replace_chunks(paper.paper_id, ch)
        stats["fingerprints"] += register_private_text(kb, paper.paper_id, text)
        stats["files"] += 1
        stats["chunks"] += len(ch)
        if f.suffix.lower() == ".mx3":
            rows = parse_mx3(text)
            for r in rows:
                r.update(formula_raw=material or "simulation", sample_form="simulation",
                         method=Method.micromagnetic.value, is_theoretical=1, quote_verified=1,
                         confidence=1.0, section="mx3", is_private=1, extractor="local-mx3")
            store_measurement_rows(kb, paper.paper_id, rows, replace=True)
            stats["mx3_params"] += len(rows)
    (settings.private_dir / ".ingest_manifest.json").write_text(json.dumps(stats), encoding="utf-8")
    return stats
