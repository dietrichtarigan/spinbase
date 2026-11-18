"""Layer 2: extract quantitative physics parameters into the relational tables.

Pipeline per paper
------------------
1. Pre-select chunks that are likely to contain numbers-with-units (regex), plus the
   abstract and all tables -> keeps prompts small and cheap.
2. DeepSeek JSON-mode call validated against :class:`PaperExtraction`.
3. **Grounding checks** (local, no LLM):
   * the ``evidence_quote`` must fuzzy-match the paper text (partial ratio >= 88),
   * the reported value must literally appear in the quote.
   Records failing either are stored with ``quote_verified=0`` and excluded from the
   default comparison view.
4. Units normalised to canonical units; materials normalised (Fe$_3$GaTe$_2$ -> Fe3GaTe2).
"""
from __future__ import annotations

import logging
import math
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from rapidfuzz import fuzz

from config.schema_physics import (PaperExtraction, ParameterRecord, Quantity, composition_key,
                                   json_schema_hint, normalize_formula, normalize_value)
from src.core.models import Chunk, Paper
from src.ingestion.chunker import n_tokens

log = logging.getLogger(__name__)

_NUM_UNIT = re.compile(
    r"\d(?:[\d.,]*)\s*(?:±\s*[\d.]+\s*)?(?:K\b|mJ|pJ|MJ|kJ|J\s*/|emu|kA|A/m|A\s*m|Oe|mT|\bT\b|nm|µm|μm|"
    r"μB|µB|mu_B|erg|%|m/s|S/cm|Ω|ohm|°)", re.I)
_QTY_WORDS = re.compile(r"Curie|T_?\{?C\}?|Dzyaloshinskii|DMI|\bD\s*=|saturation magneti|M_?\{?s\}?|"
                        r"exchange stiffness|A_?\{?ex\}?|anisotrop|K_?\{?u\}?|damping|skyrmion (size|diameter|radius)|"
                        r"coerciv|spin Hall angle|domain wall|spin wave|magnon|group velocity|decay length|"
                        r"linewidth|TMR|tunnel(ing)? magnetoresist|RA product|thermal stability|switching|"
                        r"oscillat|accuracy|memory capacity|energy per|fJ|GHz", re.I)

SYSTEM = (
    "You are a meticulous condensed-matter physics data curator. Extract ONLY quantitative "
    "parameters that are explicitly reported in the provided text (measured or computed by the "
    "authors of THIS paper, or explicitly quoted from other works — mark those with confidence <= 0.4). "
    "Never infer, convert, or guess numbers. Copy units exactly as written. The evidence_quote must be "
    "copied verbatim from the text. If nothing qualifies return an empty parameters list."
)


def select_context(chunks: list[Chunk], budget_tokens: int = 14000) -> str:
    scored = []
    for c in chunks:
        if c.kind in ("abstract", "table"):
            s = 100
        else:
            s = len(_NUM_UNIT.findall(c.text)) * 3 + len(_QTY_WORDS.findall(c.text))
        if s > 0:
            scored.append((s, c))
    scored.sort(key=lambda x: (-x[0], x[1].ord))
    picked, used = [], 0
    for s, c in scored:
        t = n_tokens(c.text)
        if used + t > budget_tokens:
            continue
        picked.append(c)
        used += t
    picked.sort(key=lambda c: c.ord)
    return "\n\n".join(f"[{c.section} | {c.kind}]\n{c.text}" for c in picked)


def _num_in(value: float | None, quote: str) -> bool:
    if value is None:
        return True
    nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", quote.replace(",", ""))]
    v = abs(value)
    if any(abs(n - v) <= 1e-6 * max(1.0, v) for n in nums):
        return True

    def mant(x: float) -> float:  # "1.2 × 10^5" in text vs 1.2e5 from the LLM
        return round(x / 10 ** math.floor(math.log10(x)), 6) if x > 0 else 0.0
    return v > 0 and any(n > 0 and mant(n) == mant(v) for n in nums)


def verify(rec: ParameterRecord, source_text: str) -> bool:
    q = rec.evidence_quote
    if len(q) < 8:
        return False
    norm = lambda s: re.sub(r"\s+", " ", s.replace("$", "").replace("{", "").replace("}", "")).lower()
    ok_quote = fuzz.partial_ratio(norm(q), norm(source_text)) >= 88
    return ok_quote and _num_in(rec.value if rec.value is not None else rec.value_min, q)


def _material_id(c, formula: str) -> int:
    fn = normalize_formula(formula) or "unknown"
    c.execute("INSERT OR IGNORE INTO materials (formula_norm, composition_key) VALUES (?,?)",
              (fn, composition_key(fn)))
    return c.execute("SELECT material_id FROM materials WHERE formula_norm=?", (fn,)).fetchone()[0]


def _method_id(c, name: str | None) -> int | None:
    if not name:
        return None
    c.execute("INSERT OR IGNORE INTO methods (name) VALUES (?)", (name,))
    return c.execute("SELECT method_id FROM methods WHERE name=?", (name,)).fetchone()[0]


_MCOLS = ["paper_id", "material_id", "method_id", "formula_raw", "stack", "doping", "sample_form", "thickness_nm",
          "layers", "substrate", "quantity", "quantity_label", "value", "value_min", "value_max", "uncertainty",
          "unit", "value_norm", "value_min_norm", "value_max_norm", "unit_norm", "temperature_K", "field_T",
          "method_detail", "is_theoretical", "evidence_quote", "quote_verified", "section", "confidence",
          "is_private", "extractor"]


def store_measurement_rows(kb, paper_id: str, rows: list[dict], replace: bool = True) -> int:
    with kb.tx() as c:
        if replace:
            c.execute("DELETE FROM measurements WHERE paper_id=?", (paper_id,))
        for r in rows:
            r = dict(r)
            r["paper_id"] = paper_id
            r["material_id"] = _material_id(c, r.get("formula_raw") or "unknown")
            r["method_id"] = _method_id(c, r.pop("method", None))
            c.execute(f"INSERT INTO measurements ({','.join(_MCOLS)}) VALUES ({','.join('?' * len(_MCOLS))})",
                      [r.get(k) for k in _MCOLS])
    return len(rows)


def records_to_rows(recs: list[ParameterRecord], source_text: str, is_private: bool, model: str) -> list[dict]:
    rows = []
    for r in recs:
        vn, un = normalize_value(r.quantity, r.value, r.unit)
        vmin, _ = normalize_value(r.quantity, r.value_min, r.unit)
        vmax, _ = normalize_value(r.quantity, r.value_max, r.unit)
        rows.append(dict(
            formula_raw=r.material.formula, stack=r.material.stack, doping=r.material.doping,
            sample_form=r.material.sample_form.value, thickness_nm=r.material.thickness_nm,
            layers=r.material.layers, substrate=r.material.substrate, quantity=r.quantity.value,
            quantity_label=r.quantity_label, value=r.value, value_min=r.value_min, value_max=r.value_max,
            uncertainty=r.uncertainty, unit=r.unit, value_norm=vn, value_min_norm=vmin, value_max_norm=vmax,
            unit_norm=un, temperature_K=r.temperature_K, field_T=r.field_T, method=r.method.value,
            method_detail=r.method_detail, is_theoretical=int(r.is_theoretical),
            evidence_quote=r.evidence_quote, quote_verified=int(verify(r, source_text)),
            section=r.section, confidence=r.confidence, is_private=int(is_private), extractor=model))
    return rows


def extract_paper(kb, llm, paper: Paper) -> dict:
    if paper.is_private:
        return {"paper_id": paper.paper_id, "skipped": "private (local-only)"}
    chunks = kb.chunks_for_paper(paper.paper_id)
    if not chunks:
        return {"paper_id": paper.paper_id, "skipped": "no chunks"}
    ctx = select_context(chunks)
    if not ctx:
        return {"paper_id": paper.paper_id, "skipped": "no quantitative text"}
    user = (f"Paper: {paper.title} ({paper.year}, {paper.venue})\n\n{json_schema_hint()}\n\n"
            f"TEXT:\n<<<\n{ctx}\n>>>")
    res = llm.structured([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
                         PaperExtraction, purpose="param_extraction", sources=[paper, *chunks],
                         max_tokens=6000)
    full = "\n".join(c.text for c in chunks)
    rows = records_to_rows(res.parameters, full, paper.is_private, llm.model)
    store_measurement_rows(kb, paper.paper_id, rows)
    with kb.tx() as c:
        c.execute("DELETE FROM paper_findings WHERE paper_id=?", (paper.paper_id,))
        c.executemany("INSERT INTO paper_findings VALUES (?,?)", [(paper.paper_id, f) for f in res.key_findings[:5]])
    kb.set_status(paper.paper_id, "extracted")
    return {"paper_id": paper.paper_id, "params": len(rows), "verified": sum(r["quote_verified"] for r in rows)}


def extract_all(kb, llm, papers: list[Paper], workers: int = 4, progress=None) -> list[dict]:
    out = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(extract_paper, kb, llm, p): p for p in papers}
        for i, f in enumerate(as_completed(futs)):
            p = futs[f]
            try:
                out.append(f.result())
            except Exception as e:
                log.warning("extraction failed for %s: %s", p.paper_id, e)
                out.append({"paper_id": p.paper_id, "error": str(e)[:200]})
            if progress:
                progress(i + 1, len(papers), p.title[:50])
    return out


# ------------------------------------------------------------ query helpers
def compare(kb, material: str | None = None, quantity: str | None = None, verified_only: bool = True,
            include_private: bool = True) -> list[dict]:
    """Cross-paper comparison rows, e.g. compare(kb, 'Fe3GaTe2', 'D_int')."""
    where, params = ["1=1"], []
    if material:
        where.append("(m.formula_norm = ? OR m.composition_key = ? OR ms.formula_raw LIKE ? OR ms.stack LIKE ?)")
        fn = normalize_formula(material)
        params += [fn, composition_key(fn), f"%{fn}%", f"%{fn}%"]
    if quantity:
        q = quantity
        try:
            q = Quantity(quantity).value
        except ValueError:
            q = next((x.value for x in Quantity if x.name == quantity), quantity)
        where.append("ms.quantity = ?")
        params.append(q)
    if verified_only:
        where.append("ms.quote_verified = 1")
    if not include_private:
        where.append("ms.is_private = 0")
    sql = f"""
      SELECT ms.measurement_id, m.formula_norm AS material, ms.stack, ms.sample_form, ms.thickness_nm,
             ms.quantity, ms.quantity_label, ms.value, ms.value_min, ms.value_max, ms.uncertainty, ms.unit,
             ms.value_norm, ms.unit_norm, ms.temperature_K, me.name AS method, ms.method_detail,
             ms.is_theoretical, ms.confidence, ms.quote_verified, ms.evidence_quote, ms.is_private,
             p.paper_id, p.title, p.authors, p.year, p.doi, p.arxiv_id, p.priority
      FROM measurements ms JOIN materials m USING(material_id)
      LEFT JOIN methods me USING(method_id) JOIN papers p USING(paper_id)
      WHERE {' AND '.join(where)}
      ORDER BY ms.quantity, m.formula_norm, p.year"""
    return [dict(r) for r in kb.conn.execute(sql, params)]
