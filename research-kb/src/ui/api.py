"""Minimal FastAPI server (localhost only by default). `kb api`"""
from __future__ import annotations

from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

from src.core.db import get_kb
from src.extraction.param_extractor import compare
from src.retrieval.agentic_rag import AgenticRAG

app = FastAPI(title="research-kb")
_rag = AgenticRAG(get_kb())


class AskReq(BaseModel):
    question: str
    nodes: Optional[list[str]] = None
    priorities: Optional[list[str]] = None


@app.post("/ask")
def ask(r: AskReq):
    a = _rag.ask(r.question, nodes=r.nodes, priorities=r.priorities)
    return {"status": a.status, "answer": a.answer_md, "warnings": a.warnings, "claims": a.claims,
            "evidence": [{"id": e.eid, "citation": e.citation, "priority": e.priority, "summary": e.summary}
                         for e in a.evidence if not e.is_private],
            "private_hits": len(a.private_hits), "cost_usd": a.cost_usd}


@app.get("/params")
def params(material: Optional[str] = None, quantity: Optional[str] = None, verified_only: bool = True):
    return compare(get_kb(), material, quantity, verified_only)


@app.get("/stats")
def stats():
    return get_kb().stats()
