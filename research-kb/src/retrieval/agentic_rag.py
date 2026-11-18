"""PaperQA2 / OpenScholar-style agentic RAG with claim-level citations and strict abstention.

Loop
----
1. **Decompose** the question into sub-queries, each routed to ``text`` search,
   ``param`` (SQL over Layer 2), or both. (DeepSeek, JSON mode.)
2. **Retrieve in parallel**: hybrid dense+BM25 search per sub-query (priority-boosted)
   and structured SQL parameter lookups.
3. **Rerank + contextual summarise** (RCS): DeepSeek scores each candidate chunk 0–10 for
   the question and writes a 1–3 sentence evidence summary *quoting numbers verbatim*.
4. **Assess sufficiency**; if gaps remain, issue follow-up queries (≤ ``max_agent_rounds``).
5. **Abstain** without calling the synthesiser when no evidence clears the bar.
6. **Synthesise** with inline ``[E#]`` markers, then locally **verify** every claim:
   each claim must cite ≥1 valid evidence item and every number in the claim must occur
   in the cited evidence. Markers are rewritten to ``[Author et al., Year, DOI]``.

Privacy: private chunks / measurements are retrieved for *local display only* and are
never put into any prompt (and the DeepSeek client would block them anyway).
"""
from __future__ import annotations

import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from config.schema_physics import Quantity
from config.settings import settings
from src.extraction.param_extractor import compare
from src.retrieval.hybrid_store import Hit, HybridStore

log = logging.getLogger(__name__)


# ------------------------------------------------------------------ schemas
class SubQuery(BaseModel):
    query: str
    route: Literal["text", "param", "both"] = "text"
    material: Optional[str] = None
    quantity: Optional[str] = None


class QueryPlan(BaseModel):
    answer_type: Literal["comparison", "explanation", "factoid", "survey"] = "explanation"
    sub_queries: list[SubQuery] = Field(default_factory=list)


class RCSItem(BaseModel):
    id: str
    score: float = Field(ge=0, le=10)
    summary: str = ""


class RCSBatch(BaseModel):
    items: list[RCSItem] = Field(default_factory=list)


class Assessment(BaseModel):
    sufficient: bool = True
    missing: list[str] = Field(default_factory=list)
    followup_queries: list[str] = Field(default_factory=list)


class Claim(BaseModel):
    text: str
    evidence: list[str] = Field(default_factory=list)


class Synthesis(BaseModel):
    status: Literal["answered", "partial", "insufficient_evidence"] = "answered"
    answer: str = ""
    claims: list[Claim] = Field(default_factory=list)
    missing: str = ""


@dataclass
class Evidence:
    eid: str
    kind: str                    # chunk | param
    text: str
    citation: str
    paper_id: str
    doi: str | None
    priority: str
    score: float = 0.0
    summary: str = ""
    is_private: bool = False
    meta: dict = field(default_factory=dict)


@dataclass
class Answer:
    question: str
    status: str
    answer_md: str
    evidence: list[Evidence]
    params: list[dict]
    private_hits: list[Hit]
    warnings: list[str]
    claims: list[dict]
    plan: dict
    cost_usd: float
    seconds: float

    def references(self) -> list[str]:
        used = {e.citation for e in self.evidence if e.eid in self.answer_md or e.citation in self.answer_md}
        return sorted(used)


# ---------------------------------------------------------------- prompts
DECOMPOSE_SYS = (
    "You plan literature searches for a condensed-matter / spintronics knowledge base. Break the user "
    "question into 1-5 focused sub-queries. Route each: 'param' if it asks for a numeric material "
    "parameter that may live in an extracted parameter table (give material formula without LaTeX and "
    f"quantity code from: {', '.join(q.value for q in Quantity)}), 'text' for mechanisms/methods/context, "
    "'both' if unsure. Return JSON {\"answer_type\": comparison|explanation|factoid|survey, "
    "\"sub_queries\": [{\"query\": str, \"route\": text|param|both, \"material\": str|null, \"quantity\": str|null}]}"
)

RCS_SYS = (
    "You are a strict relevance judge for physics literature. For each excerpt, score 0-10 how much it "
    "DIRECTLY helps answer the question (10 = contains the exact fact/number; 5 = useful context; "
    "0 = irrelevant). Write a 1-3 sentence summary of only the relevant facts, copying numbers, units, "
    "materials and methods verbatim. If irrelevant, summary=''. Return JSON {\"items\": [{\"id\": str, "
    "\"score\": number, \"summary\": str}]}"
)

ASSESS_SYS = (
    "Given a question and evidence summaries, decide if the evidence is sufficient to answer every part "
    "of the question WITHOUT outside knowledge. Return JSON {\"sufficient\": bool, \"missing\": [str], "
    "\"followup_queries\": [str (<=2 short search queries for the missing parts)]}"
)

SYNTH_SYS = (
    "You write rigorous, cited answers for a physics researcher using ONLY the numbered evidence. Rules:\n"
    "1. Every factual sentence ends with one or more evidence markers like [E3] or [E3][P2].\n"
    "2. Never use outside knowledge, never extrapolate, never convert units unless the evidence does.\n"
    "3. Report numbers exactly as in the evidence, with material, method, temperature/thickness when given.\n"
    "4. If evidence disagrees, say so and cite both sides. If part of the question is not covered, say "
    "explicitly 'The corpus does not contain sufficient evidence for ...'.\n"
    "5. For comparisons, include a Markdown table with a Source column holding the markers.\n"
    "Return JSON {\"status\": answered|partial|insufficient_evidence, \"answer\": markdown, "
    "\"claims\": [{\"text\": str, \"evidence\": [\"E1\", ...]}], \"missing\": str}"
)

INSUFFICIENT = ("**Insufficient evidence.** The curated corpus does not contain evidence that clears the "
                "relevance bar for this question, so no answer is given (abstaining rather than extrapolating).")


def _clip(t: str, n: int = 1600) -> str:
    return t if len(t) <= n else t[:n] + " …"


_NUM = re.compile(r"(?<![A-Za-z_])[-−]?\d+(?:\.\d+)?(?:\s*[×x]\s*10\^?\{?[-−]?\d+\}?)?")


def _numbers(s: str) -> set[str]:
    out = set()
    for m in _NUM.findall(s):
        t = m.replace("−", "-").replace(" ", "")
        try:
            if "10" in t and ("×" in t or "x" in t):
                out.add(t)
            else:
                f = float(t)
                if f in (0, 1, 2, 3) and "." not in t:  # ignore trivial counts / list markers
                    continue
                out.add(f"{f:g}")
        except ValueError:
            pass
    return out


class AgenticRAG:
    def __init__(self, kb, store: HybridStore | None = None, llm=None):
        self.kb = kb
        self.store = store or HybridStore(kb)
        self._llm = llm

    @property
    def llm(self):
        if self._llm is None and settings.deepseek_api_key:
            from src.extraction.deepseek_client import get_llm
            self._llm = get_llm()
        return self._llm

    # ------------------------------------------------------------ 1. plan
    def decompose(self, question: str) -> QueryPlan:
        if not self.llm:
            return QueryPlan(sub_queries=[SubQuery(query=question, route="both")])
        try:
            plan = self.llm.structured(
                [{"role": "system", "content": DECOMPOSE_SYS}, {"role": "user", "content": question}],
                QueryPlan, purpose="decompose", max_tokens=800)
        except Exception as e:
            log.warning("decompose failed: %s", e)
            plan = QueryPlan()
        if not plan.sub_queries:
            plan.sub_queries = [SubQuery(query=question, route="both")]
        # the original question is always searched too
        if all(sq.query.strip().lower() != question.strip().lower() for sq in plan.sub_queries):
            plan.sub_queries.append(SubQuery(query=question, route="text"))
        return plan

    # -------------------------------------------------------- 2. retrieve
    def retrieve(self, plan: QueryPlan, filters: dict) -> tuple[list[Hit], list[dict], list[Hit]]:
        text_q = [sq for sq in plan.sub_queries if sq.route in ("text", "both")]
        param_q = [sq for sq in plan.sub_queries if sq.route in ("param", "both") and (sq.material or sq.quantity)]
        with ThreadPoolExecutor(max_workers=6) as ex:
            tf = [ex.submit(self.store.search, sq.query, settings.retrieve_k_per_query,
                            include_private=settings.include_private_in_local_results, **filters) for sq in text_q]
            pf = [ex.submit(compare, self.kb, sq.material, sq.quantity, True, True) for sq in param_q]
            hits: dict[str, Hit] = {}
            for f in tf:
                for h in f.result():
                    if h.chunk.chunk_id not in hits or hits[h.chunk.chunk_id].score < h.score:
                        hits[h.chunk.chunk_id] = h
            params, seen = [], set()
            for f in pf:
                for r in f.result():
                    if r["measurement_id"] not in seen:
                        seen.add(r["measurement_id"])
                        params.append(r)
        all_hits = sorted(hits.values(), key=lambda h: -h.score)
        public = [h for h in all_hits if not h.chunk.is_private]
        private = [h for h in all_hits if h.chunk.is_private]
        return public, params, private

    # ---------------------------------------------------------- 3. rerank
    def rerank(self, question: str, hits: list[Hit], top_n: int) -> list[Hit]:
        cand = hits[: max(top_n * 3, 24)]
        if not self.llm or not cand:
            for h in cand:
                h.rerank_score = None
            return cand[:top_n]
        batches = [cand[i:i + 8] for i in range(0, len(cand), 8)]

        def run(batch: list[Hit]) -> None:
            body = "\n\n".join(f"[C{j}] ({h.chunk.title[:90]} | {h.chunk.section})\n{_clip(h.chunk.text, 1400)}"
                               for j, h in enumerate(batch))
            try:
                res = self.llm.structured(
                    [{"role": "system", "content": RCS_SYS},
                     {"role": "user", "content": f"QUESTION: {question}\n\nEXCERPTS:\n{body}"}],
                    RCSBatch, purpose="rerank", sources=[h.chunk for h in batch], max_tokens=1500)
                for it in res.items:
                    j = int(re.sub(r"\D", "", it.id) or -1)
                    if 0 <= j < len(batch):
                        batch[j].rerank_score, batch[j].rerank_summary = it.score, it.summary
            except Exception as e:
                log.warning("rerank batch failed: %s", e)

        with ThreadPoolExecutor(max_workers=4) as ex:
            list(ex.map(run, batches))
        # final score: LLM relevance dominates; priority tier breaks ties & boosts core
        for h in cand:
            pw = settings.priority_weights.get(h.chunk.priority, 0.6)
            h.score = (h.rerank_score if h.rerank_score is not None else 0.0) * (0.7 + 0.3 * pw)
        kept = [h for h in cand if (h.rerank_score or 0) >= settings.rerank_min_score]
        kept.sort(key=lambda h: -h.score)
        return kept[:top_n]

    # ---------------------------------------------------------- 4. assess
    def assess(self, question: str, ev: list[Evidence]) -> Assessment:
        if not self.llm or not ev:
            return Assessment(sufficient=bool(ev))
        s = "\n".join(f"- {e.summary or _clip(e.text, 300)}" for e in ev)
        try:
            return self.llm.structured(
                [{"role": "system", "content": ASSESS_SYS},
                 {"role": "user", "content": f"QUESTION: {question}\n\nEVIDENCE SUMMARIES:\n{s}"}],
                Assessment, purpose="assess", max_tokens=400)
        except Exception:
            return Assessment(sufficient=True)

    # ------------------------------------------------------- evidence pack
    @staticmethod
    def _cite(paper_like: dict | Any) -> str:
        if hasattr(paper_like, "authors"):
            authors, year, doi, title = paper_like.authors, paper_like.year, paper_like.doi, paper_like.title
            arx = None
        else:
            a = paper_like.get("authors")
            authors = json.loads(a) if isinstance(a, str) else (a or [])
            year, doi, arx = paper_like.get("year"), paper_like.get("doi"), paper_like.get("arxiv_id")
        sur = "Unknown"
        if authors:
            f = authors[0]
            sur = f.split(",")[0].strip() if "," in f else (f.split()[-1] if f.split() else f)
        etal = " et al." if len(authors) > 1 else ""
        ident = doi or (f"arXiv:{arx}" if arx else "no-DOI")
        return f"[{sur}{etal}, {year or 'n.d.'}, {ident}]"

    def _evidence(self, hits: list[Hit], params: list[dict]) -> list[Evidence]:
        ev: list[Evidence] = []
        for i, h in enumerate(hits, 1):
            c = h.chunk
            ev.append(Evidence(f"E{i}", "chunk", c.text, self._cite(c), c.paper_id, c.doi, c.priority,
                               h.score, h.rerank_summary or "", c.is_private, {"section": c.section, "title": c.title}))
        for j, r in enumerate([p for p in params if not p["is_private"]][:40], 1):
            val = r["value"] if r["value"] is not None else f"{r['value_min']}–{r['value_max']}"
            txt = (f"{r['material']} ({r['sample_form']}, {r['stack'] or ''} t={r['thickness_nm']} nm): "
                   f"{r['quantity_label'] or r['quantity']} = {val} {r['unit']}"
                   f"{' ± ' + str(r['uncertainty']) if r['uncertainty'] else ''}"
                   f" [normalised {r['value_norm']} {r['unit_norm']}] at T={r['temperature_K']} K; "
                   f"method {r['method']} {r['method_detail'] or ''}; "
                   f"{'theory' if r['is_theoretical'] else 'experiment'}. Quote: \"{r['evidence_quote']}\"")
            ev.append(Evidence(f"P{j}", "param", txt, self._cite(r), r["paper_id"], r["doi"], r["priority"],
                               1.0, "", False, r))
        return ev

    # -------------------------------------------------------- 6. synthesise
    def synthesise(self, question: str, ev: list[Evidence], answer_type: str) -> Synthesis:
        public = [e for e in ev if not e.is_private]
        block = "\n\n".join(
            f"[{e.eid}] {e.citation} (tier: {e.priority})\n"
            + (f"Summary: {e.summary}\nExcerpt: {_clip(e.text, 1200)}" if e.kind == "chunk" else e.text)
            for e in public)
        return self.llm.structured(
            [{"role": "system", "content": SYNTH_SYS},
             {"role": "user", "content": f"QUESTION ({answer_type}): {question}\n\nEVIDENCE:\n{block}"}],
            Synthesis, purpose="synthesis", model=settings.deepseek_reasoning_model,
            sources=[], max_tokens=3500)

    @staticmethod
    def verify_claims(syn: Synthesis, ev: list[Evidence]) -> tuple[list[dict], list[str]]:
        by_id = {e.eid: e for e in ev}
        report, warnings = [], []
        markers = set(re.findall(r"\[((?:E|P)\d+)\]", syn.answer))
        bad = markers - set(by_id)
        if bad:
            warnings.append(f"Answer referenced unknown evidence ids {sorted(bad)} (removed).")
        for cl in syn.claims:
            ids = [i for i in cl.evidence if i in by_id]
            src = " ".join(by_id[i].text + " " + by_id[i].summary for i in ids)
            nums = _numbers(cl.text)
            src_nums = _numbers(src)
            unsupported = sorted(n for n in nums if n not in src_nums and n not in src)
            ok = bool(ids) and not unsupported
            report.append({"claim": cl.text, "evidence": ids, "supported": ok, "unsupported_numbers": unsupported})
            if not ids:
                warnings.append(f"Uncited claim: “{cl.text[:120]}”")
            elif unsupported:
                warnings.append(f"Numbers {unsupported} in claim not found in cited evidence: “{cl.text[:120]}”")
        # sentences in the answer without any marker
        for sent in re.split(r"(?<=[.!?])\s+", re.sub(r"\|.*\|", "", syn.answer)):
            s = sent.strip()
            if len(s) > 60 and not re.search(r"\[(?:E|P)\d+\]", s) and "does not contain" not in s \
                    and "insufficient" not in s.lower() and not s.startswith("#"):
                warnings.append(f"Sentence without citation: “{s[:120]}”")
        return report, warnings

    @staticmethod
    def render(answer: str, ev: list[Evidence]) -> str:
        by_id = {e.eid: e for e in ev}

        def rep(m):
            e = by_id.get(m.group(1))
            return e.citation if e else ""
        out = re.sub(r"\[((?:E|P)\d+)\]", rep, answer)
        # collapse duplicate adjacent citations
        return re.sub(r"(\[[^\]]+\])(\s*\1)+", r"\1", out)

    # ---------------------------------------------------------------- run
    def ask(self, question: str, *, nodes: list[str] | None = None, priorities: list[str] | None = None,
            year_min: int | None = None, year_max: int | None = None) -> Answer:
        t0 = time.time()
        cost0 = self.llm.session_cost if self.llm else 0.0
        filters = {k: v for k, v in dict(nodes=nodes, priorities=priorities, year_min=year_min,
                                         year_max=year_max).items() if v}
        warnings: list[str] = []
        plan = self.decompose(question)
        hits, params, private = self.retrieve(plan, filters)
        kept = self.rerank(question, hits, settings.rerank_top_n)
        ev = self._evidence(kept, params)

        rounds = 1
        while self.llm and rounds < settings.max_agent_rounds:
            a = self.assess(question, ev)
            if a.sufficient or not a.followup_queries:
                break
            rounds += 1
            more = QueryPlan(sub_queries=[SubQuery(query=q, route="text") for q in a.followup_queries[:2]])
            h2, p2, pr2 = self.retrieve(more, filters)
            seen = {h.chunk.chunk_id for h in kept}
            k2 = self.rerank(question, [h for h in h2 if h.chunk.chunk_id not in seen], 6)
            kept = sorted(kept + k2, key=lambda h: -h.score)[: settings.rerank_top_n + 4]
            private += pr2
            ev = self._evidence(kept, params)

        public_ev = [e for e in ev if not e.is_private]
        if len(public_ev) < settings.min_evidence_for_answer:
            return Answer(question, "insufficient_evidence", INSUFFICIENT, ev, params, private, warnings, [],
                          plan.model_dump(), self._cost(cost0), time.time() - t0)
        if not self.llm:
            md = ("_Local-only mode (no DEEPSEEK_API_KEY): showing ranked evidence without synthesis._\n\n" +
                  "\n".join(f"- {e.citation} — {_clip(e.text, 300)}" for e in public_ev))
            return Answer(question, "retrieval_only", md, ev, params, private, warnings, [], plan.model_dump(),
                          0.0, time.time() - t0)

        syn = self.synthesise(question, public_ev, plan.answer_type)
        claims, w = self.verify_claims(syn, public_ev)
        warnings += w
        status = syn.status
        if claims and sum(not c["supported"] for c in claims) / len(claims) > 0.3:
            status = "partial"
            warnings.insert(0, "More than 30% of claims failed local grounding checks — treat with caution.")
        md = self.render(syn.answer, public_ev)
        if syn.missing and status != "answered":
            md += f"\n\n> **Evidence gap:** {syn.missing}"
        return Answer(question, status, md, ev, params, private, warnings, claims, plan.model_dump(),
                      self._cost(cost0), time.time() - t0)

    def _cost(self, c0: float) -> float:
        return round((self.llm.session_cost - c0) if self.llm else 0.0, 5)
