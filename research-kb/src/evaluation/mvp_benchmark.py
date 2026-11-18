"""MVP evaluation harness (≈30 core papers, 15–20 domain questions).

Benchmark file: ``data/benchmark/mvp_questions.yaml``. Each item may specify

    id, question, type: comparison|explanation|factoid|abstain
    gold_dois:     [dois that a good answer must retrieve/cite]
    expected_values: [{material, quantity, value, unit, tol_rel}]   # numeric ground truth
    must_mention:  [keywords that must appear in the answer]
    should_abstain: true                                            # out-of-corpus question

Metrics per question (aggregated in the report):
* retrieval_recall   – fraction of gold DOIs present in the evidence set
* citation_recall    – fraction of gold DOIs actually cited in the answer
* citation_validity  – fraction of rendered citations that map to retrieved evidence
* grounded_claims    – fraction of claims passing local number/citation checks
* value_accuracy     – fraction of expected values found (±tol) in answer or param table
* abstention_correct – abstained iff should_abstain
* keyword_hit, latency_s, cost_usd
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import yaml

from config.schema_physics import normalize_formula
from config.settings import settings

BENCH = settings.data_dir / "benchmark" / "mvp_questions.yaml"


def load_questions(path: Path | None = None) -> list[dict]:
    p = path or BENCH
    return yaml.safe_load(p.read_text(encoding="utf-8"))["questions"]


def _value_found(ev: dict, answer_md: str, params: list[dict]) -> bool:
    tol = ev.get("tol_rel", 0.1)
    target = float(ev["value"])
    mat = normalize_formula(ev.get("material", "")).lower()
    for r in params:
        if ev.get("quantity") and r["quantity"] != ev["quantity"]:
            continue
        if mat and mat not in (r["material"] or "").lower():
            continue
        for v in (r["value"], r["value_norm"]):
            if v is not None and abs(v - target) <= tol * abs(target):
                return True
    for n in re.findall(r"\d+(?:\.\d+)?", answer_md):
        if abs(float(n) - target) <= tol * abs(target):
            return True
    return False


def score(q: dict, ans) -> dict:
    ev_dois = {e.doi for e in ans.evidence if e.doi}
    cited = set(re.findall(r"10\.\d{4,9}/[^\s\],;]+", ans.answer_md))
    gold = {d.lower() for d in q.get("gold_dois") or []}
    abstained = ans.status == "insufficient_evidence"
    claims = ans.claims or []
    labels = set(re.findall(r"\[[^\[\]]+?, (?:\d{4}|n\.d\.), [^\[\]]+?\]", ans.answer_md))
    ev_labels = {e.citation for e in ans.evidence}
    m = {
        "id": q["id"], "status": ans.status,
        "retrieval_recall": (len(gold & ev_dois) / len(gold)) if gold else None,
        "citation_recall": (len(gold & {c.lower().rstrip('.') for c in cited}) / len(gold)) if gold else None,
        "citation_validity": (len(labels & ev_labels) / len(labels)) if labels else None,
        "grounded_claims": (sum(c["supported"] for c in claims) / len(claims)) if claims else None,
        "value_accuracy": None, "keyword_hit": None,
        "abstention_correct": abstained == bool(q.get("should_abstain", False)),
        "latency_s": round(ans.seconds, 2), "cost_usd": ans.cost_usd, "warnings": len(ans.warnings),
    }
    if q.get("expected_values"):
        m["value_accuracy"] = sum(_value_found(v, ans.answer_md, ans.params) for v in q["expected_values"]) / len(q["expected_values"])
    if q.get("must_mention"):
        low = ans.answer_md.lower()
        m["keyword_hit"] = sum(k.lower() in low for k in q["must_mention"]) / len(q["must_mention"])
    return m


def run(rag, path: Path | None = None, only: list[str] | None = None, progress=None) -> dict:
    qs = [q for q in load_questions(path) if not only or q["id"] in only]
    results, answers = [], []
    for i, q in enumerate(qs):
        if progress:
            progress(i, len(qs), q["id"])
        try:
            ans = rag.ask(q["question"])
            results.append(score(q, ans))
            answers.append({"id": q["id"], "question": q["question"], "status": ans.status,
                            "answer": ans.answer_md, "warnings": ans.warnings, "claims": ans.claims})
        except Exception as e:
            results.append({"id": q["id"], "status": "error", "error": str(e)[:300]})
    agg = {}
    keys = ["retrieval_recall", "citation_recall", "citation_validity", "grounded_claims", "value_accuracy",
            "keyword_hit", "abstention_correct", "latency_s", "cost_usd"]
    for k in keys:
        vals = [float(r[k]) for r in results if r.get(k) is not None]
        agg[k] = round(sum(vals) / len(vals), 3) if vals else None
    agg["n"] = len(results)
    agg["errors"] = sum(r.get("status") == "error" for r in results)
    out_dir = settings.data_dir / "benchmark" / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    (out_dir / f"run-{stamp}.json").write_text(json.dumps({"aggregate": agg, "per_question": results,
                                                           "answers": answers}, indent=1, ensure_ascii=False),
                                               encoding="utf-8")
    return {"aggregate": agg, "per_question": results, "file": str(out_dir / f"run-{stamp}.json")}
