"""Offline tests: no network, no model downloads (hash embedder + fake LLM)."""
from __future__ import annotations

import json
import os
import re

import pytest

os.environ.setdefault("EMBEDDING_BACKEND", "hash")

from config.schema_physics import (ParameterRecord, Quantity, composition_key,  # noqa: E402
                                   normalize_formula, normalize_value)
from src.core.db import KB, fts_normalize  # noqa: E402
from src.core.models import Paper  # noqa: E402
from src.core.privacy import PrivacyViolation, assert_public, register_private_text  # noqa: E402
from src.ingestion.chunker import chunk_document  # noqa: E402
from src.ingestion.deduplicator import register_candidate  # noqa: E402
from src.ingestion.pdf_parser import parse_latex  # noqa: E402
from src.retrieval.hybrid_store import HashEmbedder, HybridStore  # noqa: E402

LATEX = r"""
\documentclass{revtex4}
\title{Room-temperature skyrmions in Fe$_3$GaTe$_2$}
\begin{document}
\begin{abstract}We report N\'eel skyrmions in Fe$_3$GaTe$_2$ flakes above room temperature.\end{abstract}
\maketitle
\section{Introduction}
Two-dimensional magnets with perpendicular anisotropy host chiral textures. % comment
The Dzyaloshinskii-Moriya interaction (DMI) stabilises skyrmions~\cite{x}.
\section{Results}
From the stripe domain period we extract an interfacial DMI constant of $D = 0.38 \pm 0.05$ mJ/m$^2$
at 300 K using polar MOKE microscopy, with Curie temperature $T_C = 350$ K.
\begin{equation}
 E = A (\nabla m)^2 + D [m_z \nabla\cdot m - (m\cdot\nabla) m_z] \label{eq:1}
\end{equation}
The effective anisotropy is positive.
\begin{table}\caption{Magnetic parameters of Fe$_3$GaTe$_2$}
\begin{tabular}{cc} $M_s$ & 280 kA/m \\ $K_u$ & 0.5 MJ/m$^3$ \end{tabular}\end{table}
\section{Acknowledgments}
We thank everyone.
\end{document}
"""


@pytest.fixture()
def kb(tmp_path):
    return KB(tmp_path / "kb.sqlite")


def _ingest(kb, tmp_path, pid_suffix="", priority="core", doi="10.1000/fgt.1", title=None):
    src = tmp_path / f"src{pid_suffix}"
    src.mkdir()
    (src / "main.tex").write_text(LATEX, encoding="utf-8")
    p = Paper(title=title or "Room-temperature skyrmions in Fe3GaTe2", authors=["Doe, Jane", "Roe, R."],
              year=2024, doi=doi, priority=priority, node="2D vdW Magnets", latex_path=str(src))
    register_candidate(kb, p)
    doc = parse_latex(p.paper_id, src)
    kb.replace_chunks(p.paper_id, chunk_document(doc, p))
    return p, doc


# ------------------------------------------------------------------ physics
def test_formula_normalisation():
    assert normalize_formula("Fe$_3$GaTe$_2$") == "Fe3GaTe2"
    assert normalize_formula(r"\text{Fe}_3\text{GaTe}_2") == "Fe3GaTe2"
    assert normalize_formula("Fe_{3}GaTe_{2}") == "Fe3GaTe2"
    assert composition_key("GaFe3Te2") == composition_key("Fe3GaTe2")
    assert "Fe3GaTe2" in fts_normalize("flakes of Fe$_3$GaTe$_2$ and")


def test_unit_normalisation():
    assert normalize_value(Quantity.dmi_interfacial, 1.0, "erg/cm²") == (1.0, "mJ/m^2")
    assert normalize_value(Quantity.saturation_magnetization, 300, "emu/cm^3") == (300, "kA/m")
    v, u = normalize_value(Quantity.exchange_stiffness, 1e-6, "erg/cm")
    assert u == "pJ/m" and abs(v - 10.0) < 1e-9
    assert normalize_value(Quantity.coercive_field, 500, "Oe")[0] == pytest.approx(0.05)


# ------------------------------------------------------------------ parsing
def test_latex_parser_preserves_equations_and_tables(kb, tmp_path):
    p, doc = _ingest(kb, tmp_path)
    kinds = [b.kind for s in doc.sections for b in s.blocks]
    assert "equation" in kinds and "table" in kinds
    eq = next(b.text for s in doc.sections for b in s.blocks if b.kind == "equation")
    assert r"\nabla" in eq and "label" not in eq
    assert doc.abstract.startswith("We report")
    chunks = kb.chunks_for_paper(p.paper_id)
    assert all(c.doi == "10.1000/fgt.1" and c.priority == "core" for c in chunks)
    assert not any("thank everyone" in c.text for c in chunks)  # acknowledgments skipped


# -------------------------------------------------------------------- dedup
def test_dedup_preprint_vs_published(kb):
    a = Paper(title="Room-temperature skyrmions in Fe$_3$GaTe$_2$", authors=["Jane Doe"], year=2023,
              arxiv_id="2301.00001v2", venue="arXiv", priority="supporting")
    assert register_candidate(kb, a) == "new"
    b = Paper(title="Room temperature skyrmions in Fe3GaTe2", authors=["Doe, Jane"], year=2024,
              doi="https://doi.org/10.1038/XYZ", venue="Nat. Phys.", priority="core")
    assert register_candidate(kb, b) == "merged"
    ps = kb.papers()
    assert len(ps) == 1 and ps[0].doi == "10.1038/xyz" and ps[0].arxiv_id == "2301.00001"
    assert ps[0].priority == "core" and ps[0].venue == "Nat. Phys."


# ------------------------------------------------------------------ privacy
def test_privacy_guard_blocks_flags_and_fingerprints(kb):
    secret = ("Sample B7 MOKE run 14: the unpublished Fe3GaTe2 flake shows a coercive field of 0.42 T "
              "after annealing at 420 K for two hours in our cryostat")
    priv = Paper(title="[PRIVATE] lab note", is_private=True)
    kb.upsert_paper(priv)
    register_private_text(kb, priv.paper_id, secret)
    with pytest.raises(PrivacyViolation):
        assert_public([priv], kb=kb)
    with pytest.raises(PrivacyViolation):  # pasted inside an otherwise innocent prompt
        assert_public(text="Please summarise: " + secret[20:140] + " thanks", kb=kb)
    assert_public(text="What DMI constants are reported for Fe3GaTe2?", kb=kb)  # fine


def test_egress_allowlist():
    from src.core.http import check_egress
    check_egress("https://export.arxiv.org/api/query")
    with pytest.raises(PrivacyViolation):
        check_egress("https://evil.example.com/upload")


def test_mx3_local_parser():
    from src.ingestion.private_ingest import parse_mx3
    rows = parse_mx3("SetGridsize(256,256,1)\nMsat = 580e3\nAex = 13e-12 // J/m\nDind = 2.5e-3\nalpha=0.02\n")
    d = {r["quantity"]: r for r in rows}
    assert d["M_s"]["value_norm"] == pytest.approx(580) and d["D_int"]["value_norm"] == pytest.approx(2.5)
    assert d["A_ex"]["value_norm"] == pytest.approx(13)


# --------------------------------------------------------------- retrieval
def test_hybrid_search_and_priority_boost(kb, tmp_path):
    core, _ = _ingest(kb, tmp_path, "a", "core", "10.1/core")
    bg, _ = _ingest(kb, tmp_path, "b", "background", "10.1/bg", title="A background survey of chiral magnets")
    assert len(kb.papers()) == 2
    store = HybridStore(kb, HashEmbedder())
    assert store.index_missing() > 0
    hits = store.search("DMI constant Fe3GaTe2 MOKE", k=10)
    # identical text in both papers -> the core-tier copy must rank first
    assert hits and hits[0].chunk.priority == "core"
    assert {h.chunk.priority for h in hits} == {"core", "background"}
    assert {"dense", "bm25"} & hits[0].sources


class FakeLLM:
    """Mimics DeepSeekClient.structured and still enforces the privacy guard."""
    model = "fake"
    session_cost = 0.0

    def __init__(self, kb, synth_answer=None):
        self.kb, self.synth_answer, self.calls = kb, synth_answer, []

    def structured(self, messages, schema, *, purpose, sources=(), **kw):
        assert_public(list(sources), "\n".join(m["content"] for m in messages), kb=self.kb)
        self.calls.append(purpose)
        user = messages[-1]["content"]
        if purpose == "decompose":
            return schema.model_validate({"answer_type": "comparison", "sub_queries": [
                {"query": user, "route": "both", "material": "Fe3GaTe2", "quantity": "D_int"}]})
        if purpose == "rerank":
            ids = re.findall(r"\[C(\d+)\]", user)
            dmi = "DMI" in user.split("EXCERPTS:")[0] or "Dzyaloshinskii" in user.split("EXCERPTS:")[0]
            items = []
            for i in ids:
                seg = user.split(f"[C{i}]")[1].split("[C")[0]
                rel = dmi and ("0.38" in seg or "DMI" in seg)
                items.append({"id": f"C{i}", "score": 9 if rel else 1,
                              "summary": "D = 0.38 ± 0.05 mJ/m^2 at 300 K by polar MOKE" if rel else ""})
            return schema.model_validate({"items": items})
        if purpose == "assess":
            return schema.model_validate({"sufficient": True})
        if purpose == "synthesis":
            return schema.model_validate(self.synth_answer)
        raise AssertionError(purpose)


def test_agent_abstains_when_no_evidence(kb, tmp_path):
    from src.retrieval.agentic_rag import AgenticRAG
    _ingest(kb, tmp_path)
    store = HybridStore(kb, HashEmbedder())
    store.index_missing()
    llm = FakeLLM(kb)
    ans = AgenticRAG(kb, store, llm).ask("What is the superconducting Tc of Fe3GaTe2 under 20 GPa pressure?")
    assert ans.status == "insufficient_evidence"
    assert "synthesis" not in llm.calls  # abstained before synthesis


def test_agent_cited_answer_and_claim_checks(kb, tmp_path):
    from src.retrieval.agentic_rag import AgenticRAG
    _ingest(kb, tmp_path)
    store = HybridStore(kb, HashEmbedder())
    store.index_missing()
    synth = {"status": "answered",
             "answer": "Polar MOKE gives D = 0.38 ± 0.05 mJ/m^2 at 300 K [E1]. It also reports D = 9.9 mJ/m^2 [E1].",
             "claims": [{"text": "Polar MOKE gives D = 0.38 ± 0.05 mJ/m^2 at 300 K", "evidence": ["E1"]},
                        {"text": "It also reports D = 9.9 mJ/m^2", "evidence": ["E1"]}]}
    ans = AgenticRAG(kb, store, FakeLLM(kb, synth)).ask("Compare reported DMI constants for Fe3GaTe2")
    assert "[Doe et al., 2024, 10.1000/fgt.1]" in ans.answer_md
    sup = {c["claim"][:20]: c["supported"] for c in ans.claims}
    assert sup["Polar MOKE gives D ="] is True
    assert sup["It also reports D = "] is False  # hallucinated number caught
    assert any("9.9" in w for w in ans.warnings)


def test_private_chunks_never_reach_llm(kb, tmp_path):
    from src.retrieval.agentic_rag import AgenticRAG
    from src.core.models import Chunk
    _ingest(kb, tmp_path)
    priv = Paper(title="[PRIVATE] MOKE notes", is_private=True, priority="core")
    kb.upsert_paper(priv)
    text = "Unpublished: our Fe3GaTe2 device D07 DMI constant 0.91 mJ/m^2 measured by MOKE at 300 K in 2026 run"
    kb.replace_chunks(priv.paper_id, [Chunk(chunk_id=priv.paper_id + ":0", paper_id=priv.paper_id, ord=0,
                                            text=text, title=priv.title, priority="core", is_private=True)])
    register_private_text(kb, priv.paper_id, text)
    store = HybridStore(kb, HashEmbedder())
    store.index_missing()
    synth = {"status": "answered", "answer": "D = 0.38 mJ/m^2 [E1].",
             "claims": [{"text": "D = 0.38 mJ/m^2", "evidence": ["E1"]}]}
    ans = AgenticRAG(kb, store, FakeLLM(kb, synth)).ask("DMI constant Fe3GaTe2 MOKE")
    assert any(h.chunk.is_private for h in ans.private_hits)       # visible locally
    assert all(not e.is_private for e in ans.evidence)              # never in prompt evidence


def test_param_verification_and_compare(kb, tmp_path):
    from config.schema_physics import MaterialSystem
    from src.extraction.param_extractor import compare, records_to_rows, store_measurement_rows
    p, doc = _ingest(kb, tmp_path)
    full = "\n".join(c.text for c in kb.chunks_for_paper(p.paper_id))
    good = ParameterRecord(material=MaterialSystem(formula="Fe$_3$GaTe$_2$"), quantity=Quantity.dmi_interfacial,
                           quantity_label="D", value=0.38, uncertainty=0.05, unit="mJ/m^2", method="polar_MOKE",
                           evidence_quote="we extract an interfacial DMI constant of $D = 0.38 \\pm 0.05$ mJ/m$^2$")
    bad = good.model_copy(update={"value": 1.7, "evidence_quote": "an interfacial DMI constant of D = 1.7 mJ/m2 in our flakes"})
    rows = records_to_rows([good, bad], full, False, "fake")
    assert [r["quote_verified"] for r in rows] == [1, 0]
    store_measurement_rows(kb, p.paper_id, rows)
    out = compare(kb, "GaFe3Te2", "D_int")
    assert len(out) == 1 and out[0]["value"] == 0.38 and out[0]["method"] == "polar_MOKE"
    assert len(compare(kb, "Fe3GaTe2", "D_int", verified_only=False)) == 2
    edges = kb.conn.execute("SELECT COUNT(*) FROM kg_edges").fetchone()[0]
    assert edges == 6


def test_taxonomy_scoring():
    from src.acquisition.csv_loader import load_taxonomy
    from src.acquisition.arxiv_scraper import score_and_tag
    from config.settings import settings
    t = load_taxonomy(settings.keywords_dir / "taxonomy_example.csv")
    p = score_and_tag(Paper(title="Spin-orbit torque switching", abstract="We study spin orbit torque in Pt/Co."), t)
    assert p.node == "Spin-Orbit Torque" and p.priority == "supporting"
    p2 = score_and_tag(Paper(title="Skyrmions in Fe3GaTe2", abstract="magnetic skyrmion via MOKE"), t)
    assert p2.priority == "core"


def test_arxiv_feed_parsing():
    from src.acquisition.arxiv_scraper import parse_feed
    xml = """<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:arxiv="http://arxiv.org/schemas/atom" xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">
      <opensearch:totalResults>1</opensearch:totalResults>
      <entry><id>http://arxiv.org/abs/2401.01234v3</id><published>2024-01-03T00:00:00Z</published>
      <title>Above room temperature  skyrmions</title><summary>Abstract here.</summary>
      <author><name>A. Author</name></author><author><name>B. Author</name></author>
      <arxiv:doi>10.1103/PhysRevB.1.2</arxiv:doi><arxiv:journal_ref>PRB 1, 2 (2024)</arxiv:journal_ref></entry></feed>"""
    ps, total = parse_feed(xml)
    assert total == 1 and ps[0].arxiv_id == "2401.01234" and ps[0].doi == "10.1103/physrevb.1.2"
    assert ps[0].title == "Above room temperature skyrmions" and ps[0].year == 2024


def test_zotero_bibtex_tag_mapping(kb, tmp_path):
    from src.acquisition.csv_loader import load_taxonomy
    from src.acquisition.zotero_importer import import_zotero
    from config.settings import settings
    (tmp_path / "files").mkdir()
    pdf = tmp_path / "files" / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    bib = tmp_path / "lib.bib"
    bib.write_text("""@article{a, title={{Imaging DMI in Fe$_3$GaTe$_2$}}, author={Doe, Jane and Roe, Rick},
      year={2025}, doi={10.1/abc}, journal={PRL}, keywords={node:N03, priority:core},
      file={Full Text PDF:files/paper.pdf:application/pdf}}""", encoding="utf-8")
    t = load_taxonomy(settings.keywords_dir / "taxonomy_example.csv")
    st = import_zotero(kb, t, bib)
    assert st["with_pdf"] == 1
    p = kb.papers()[0]
    assert p.node == "Dzyaloshinskii-Moriya Interaction" and p.priority == "core" and p.status == "downloaded"
