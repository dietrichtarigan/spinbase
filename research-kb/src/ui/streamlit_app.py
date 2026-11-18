"""Streamlit UI: cited QA + parameter comparison table + corpus browser.  `kb ui`"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from config.schema_physics import Quantity  # noqa: E402
from src.core.db import get_kb  # noqa: E402
from src.extraction.param_extractor import compare  # noqa: E402
from src.retrieval.agentic_rag import AgenticRAG  # noqa: E402

st.set_page_config(page_title="Spin KB", layout="wide")
kb = get_kb()


@st.cache_resource
def rag():
    return AgenticRAG(kb)


s = kb.stats()
st.sidebar.title("Spin research KB")
st.sidebar.metric("Papers in corpus", sum(s["by_priority_in_corpus"].values()))
st.sidebar.metric("Chunks", s["chunks"])
st.sidebar.metric("Parameters", s["measurements"])
st.sidebar.caption(f"LLM spend: ${s['llm_cost_usd']}")
nodes = [r[0] for r in kb.conn.execute("SELECT DISTINCT node FROM papers WHERE node IS NOT NULL ORDER BY node")]

tab_qa, tab_params, tab_corpus = st.tabs(["Ask", "Parameter comparison", "Corpus"])

with tab_qa:
    q = st.text_area("Question", "Compare all reported DMI constants and measurement methods for Fe3GaTe2 across the corpus.")
    c1, c2 = st.columns(2)
    sel_nodes = c1.multiselect("Restrict to nodes", nodes)
    sel_prio = c2.multiselect("Restrict to tiers", ["core", "supporting", "background"])
    if st.button("Ask", type="primary"):
        with st.spinner("Decompose → hybrid + SQL retrieval → rerank → cited synthesis…"):
            a = rag().ask(q, nodes=sel_nodes or None, priorities=sel_prio or None)
        {"answered": st.success, "partial": st.warning}.get(a.status, st.error)(
            f"{a.status} · {a.seconds:.1f}s · ${a.cost_usd:.4f}")
        st.markdown(a.answer_md)
        for w in a.warnings:
            st.caption(f"⚠ {w}")
        if a.private_hits:
            with st.expander(f"🔒 Private local evidence ({len(a.private_hits)}) — never sent to the LLM"):
                for h in a.private_hits[:10]:
                    st.markdown(f"**{h.chunk.title}**\n\n{h.chunk.text[:600]}")
        with st.expander("Evidence"):
            for e in a.evidence:
                st.markdown(f"**{e.eid}** {e.citation} · *{e.priority}* · score {e.score:.2f}\n\n"
                            f"{e.summary or e.text[:500]}")
        with st.expander("Claim verification"):
            st.dataframe(pd.DataFrame(a.claims))
        with st.expander("Plan"):
            st.json(a.plan)

with tab_params:
    c1, c2, c3 = st.columns([2, 2, 1])
    mat = c1.text_input("Material (formula)", "Fe3GaTe2")
    qty = c2.selectbox("Quantity", [""] + [x.value for x in Quantity])
    ver = c3.checkbox("verified quotes only", True)
    rows = compare(kb, mat or None, qty or None, verified_only=ver)
    if rows:
        df = pd.DataFrame(rows)
        df["first_author"] = df["authors"].map(lambda a: (json.loads(a or "[]") or ["?"])[0])
        df.loc[df["is_private"] == 1, "first_author"] = "🔒 private"
        cols = ["material", "quantity", "quantity_label", "value", "uncertainty", "unit", "value_norm", "unit_norm",
                "temperature_K", "method", "method_detail", "sample_form", "thickness_nm", "stack", "first_author",
                "year", "doi", "priority", "quote_verified", "evidence_quote"]
        st.dataframe(df[cols], use_container_width=True, hide_index=True)
        plot = df.dropna(subset=["value_norm"])
        if qty and len(plot):
            st.scatter_chart(plot, x="year", y="value_norm", color="method")
        st.download_button("Download CSV", df[cols].to_csv(index=False), "params.csv")
    else:
        st.info("No parameters yet for this filter. Run `kb extract`.")

with tab_corpus:
    df = pd.read_sql_query("SELECT paper_id, title, year, venue, doi, arxiv_id, node, priority, status, source, mvp "
                           "FROM papers WHERE is_private=0 AND status!='candidate' ORDER BY priority, year DESC", kb.conn)
    st.dataframe(df, use_container_width=True, hide_index=True)
