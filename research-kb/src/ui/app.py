"""`kb` command-line interface. Run `kb --help`.

Typical MVP flow:
    kb taxonomy                     # check the CSV
    kb harvest --priority core      # arXiv (1 req / 3 s)
    kb zotero path/to/Library.bib   # paywalled/curated papers
    kb mvp --n 30                   # pick + download + parse + index + extract the MVP batch
    kb ask "Compare all reported DMI constants ... Fe3GaTe2"
    kb params --material Fe3GaTe2 --quantity D_int
    kb bench
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn
from rich.table import Table

from config.settings import PROJECT_ROOT, settings

for _s in (sys.stdout, sys.stderr):  # Windows cp1252 consoles choke on → ✓ 🔒
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

app = typer.Typer(add_completion=False, no_args_is_help=True, help="Local-first physics research KB")
con = Console()
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")


def _kb():
    from src.core.db import get_kb
    return get_kb()


def _tax(path: Optional[Path] = None):
    from src.acquisition.csv_loader import load_taxonomy
    return load_taxonomy(path)


class _Prog:
    def __init__(self, desc: str):
        self.p = Progress(TextColumn("[bold]{task.description}"), BarColumn(), TextColumn("{task.completed}/{task.total}"),
                          TextColumn("{task.fields[info]}"), TimeRemainingColumn(), console=con)
        self.desc, self.t = desc, None

    def __enter__(self):
        self.p.__enter__()
        self.t = self.p.add_task(self.desc, total=1, info="")
        return self

    def __call__(self, i, n, info=""):
        self.p.update(self.t, completed=i, total=n, info=str(info)[:50])

    def __exit__(self, *a):
        self.p.__exit__(*a)


# ---------------------------------------------------------------- setup
@app.command()
def taxonomy(csv: Optional[Path] = typer.Option(None, help="Taxonomy CSV (default: auto-detect)")):
    """Validate and summarise the keyword taxonomy CSV."""
    from src.acquisition.csv_loader import summarize
    t = _tax(csv)
    tb = Table(title=f"Taxonomy: {t.source}  ({len(t.keywords)} keywords, {len(t.nodes)} nodes)")
    for c in ("node_id", "node", "keywords", "core", "supporting", "background"):
        tb.add_column(c)
    for r in summarize(t):
        tb.add_row(*(str(r[c]) for c in ("node_id", "node", "keywords", "core", "supporting", "background")))
    con.print(tb)


@app.command()
def status():
    """Corpus / index / cost statistics and available parser backends."""
    from src.ingestion.pdf_parser import available_backends
    con.print_json(json.dumps({**_kb().stats(), "parsers": available_backends(),
                               "embedding": f"{settings.embedding_backend}:{settings.embedding_model}",
                               "deepseek_key_set": bool(settings.deepseek_api_key)}))


# ---------------------------------------------------------- acquisition
@app.command()
def harvest(priority: list[str] = typer.Option(["core", "supporting"], help="Tiers to query"),
            node: list[str] = typer.Option([], help="Restrict to node names"),
            category: list[str] = typer.Option([], help="arXiv categories, e.g. cond-mat.mtrl-sci"),
            csv: Optional[Path] = None,
            refresh: bool = typer.Option(False, help="Re-query keywords already harvested")):
    """Query arXiv for taxonomy keywords (rate limit: 1 request / 3 s). Reruns skip done keywords."""
    from src.acquisition.arxiv_scraper import harvest as h
    t = _tax(csv)
    n = len([k for k in t.keywords if k.priority in priority and (not node or k.node in node)])
    con.print(f"[yellow]{n} keyword queries · ≥{settings.arxiv_min_interval_s}s apart (≈{n * 3.2 / 60:.0f}+ min)")
    with _Prog("arXiv") as pr:
        st = h(t, _kb(), tuple(priority), category or None, node or None, progress=pr, refresh=refresh)
    con.print(st)


@app.command()
def snowball(doi: list[str] = typer.Option([], help="Seed DOIs (default: reviews + top core papers in corpus)"),
             direction: list[str] = typer.Option(["backward", "forward"]),
             top: int = typer.Option(15, help="Number of auto seeds"),
             min_score: float = typer.Option(1.0, help="Min taxonomy score to keep a neighbour"),
             core_only: bool = typer.Option(False, help="Keep only neighbours hitting a core keyword")):
    """Citation snowballing via OpenAlex / Semantic Scholar from key (review) papers."""
    from src.acquisition.snowballer import snowball as sb
    from src.core.models import Paper
    kb = _kb()
    if doi:
        seeds = [kb.papers("doi=?", (d.lower(),))[0] if kb.papers("doi=?", (d.lower(),)) else Paper(doi=d, title=d)
                 for d in doi]
    else:
        seeds = kb.papers("is_private=0 AND status != 'rejected' AND priority='core' "
                          "ORDER BY is_review DESC, relevance_score DESC LIMIT ?", (top,))
    with _Prog("snowball") as pr:
        st = sb(kb, _tax(), seeds, tuple(direction), min_score, core_only, progress=pr)
    con.print(st)


@app.command()
def zotero(export: Path = typer.Argument(..., help="Zotero BibTeX (.bib) or CSL-JSON (.json) export"),
           mvp: bool = typer.Option(False, help="Mark imported items as part of the MVP batch")):
    """Import a Zotero export (+ PDF attachments) and map tags to node/priority."""
    from src.acquisition.zotero_importer import import_zotero
    con.print(import_zotero(_kb(), _tax(), export, mvp=mvp))


@app.command()
def dedupe():
    """Second-pass deduplication over the whole paper table."""
    from src.ingestion.deduplicator import dedupe_existing
    con.print({"merged": dedupe_existing(_kb())})


@app.command()
def select(target: int = typer.Option(settings.corpus_target_size), min_per_node: int = 5):
    """Curate candidates into the corpus (tier shares + node coverage)."""
    from src.acquisition.selector import select_corpus
    con.print(select_corpus(_kb(), target, min_per_node))


# ------------------------------------------------------------ pipeline
def _download(papers, latex: bool):
    from src.acquisition import arxiv_scraper as ax
    kb = _kb()
    with _Prog("download") as pr:
        for i, p in enumerate(papers):
            pr(i, len(papers), p.arxiv_id or p.doi)
            if p.is_private or (p.pdf_path and Path(p.pdf_path).exists() and (p.latex_path or not latex)):
                continue
            if not p.arxiv_id and p.doi:
                try:
                    from src.acquisition.snowballer import oa_find_arxiv
                    p.arxiv_id = oa_find_arxiv(p)
                    kb.upsert_paper(p)
                except Exception:
                    pass
            if not p.arxiv_id:
                continue  # paywalled without preprint -> add via Zotero
            lp = ax.download_latex(p) if latex else None
            pdf = ax.download_pdf(p)
            if pdf or lp:
                kb.set_status(p.paper_id, "downloaded", pdf_path=str(pdf) if pdf else p.pdf_path,
                              latex_path=str(lp) if lp else None)
        pr(len(papers), len(papers))


def _parse_index(papers, parser: str | None = None):
    from src.ingestion.chunker import chunk_document
    from src.ingestion.pdf_parser import parse_paper
    from src.retrieval.hybrid_store import HybridStore
    kb = _kb()
    ok = fail = 0
    with _Prog("parse+chunk") as pr:
        for i, p in enumerate(papers):
            pr(i, len(papers), p.title[:40])
            p = kb.get_paper(p.paper_id)
            if not (p.pdf_path or p.latex_path):
                continue
            try:
                doc = parse_paper(p, prefer=parser)
                doc.save()
                kb.replace_chunks(p.paper_id, chunk_document(doc, p))
                kb.set_status(p.paper_id, "parsed")
                ok += 1
            except Exception as e:
                fail += 1
                con.print(f"[red]parse failed {p.paper_id}: {e}")
        pr(len(papers), len(papers))
    with _Prog("embed") as pr:
        n = HybridStore(kb).index_missing(progress=pr)
    with kb.tx() as c:
        c.execute("UPDATE papers SET status='indexed' WHERE status='parsed'")
    con.print({"parsed": ok, "failed": fail, "embedded_chunks": n})


def _scope(mvp_only: bool, status: str | None = None):
    where = ["is_private=0"]
    if mvp_only:
        where.append("mvp=1")
    if status:
        where.append(f"status IN ({','.join(repr(s) for s in status.split(','))})")
    else:
        where.append("status NOT IN ('candidate','rejected')")
    return _kb().papers(" AND ".join(where))


@app.command()
def download(mvp_only: bool = False, latex: bool = typer.Option(settings.prefer_latex_source)):
    """Download PDFs (+ LaTeX sources) for selected papers from arXiv."""
    _download(_scope(mvp_only, "selected"), latex)


@app.command()
def parse(mvp_only: bool = False, parser: Optional[str] = typer.Option(None, help="force latex|grobid|marker|pymupdf"),
          reparse: bool = False):
    """Parse downloaded papers locally, chunk them and build embeddings + BM25."""
    _parse_index(_scope(mvp_only, None if reparse else "downloaded"), parser)


@app.command()
def index():
    """(Re)build embeddings for any chunks missing vectors (e.g. after changing model)."""
    from src.retrieval.hybrid_store import HybridStore
    with _Prog("embed") as pr:
        con.print({"embedded": HybridStore(_kb()).index_missing(progress=pr)})


@app.command()
def extract(mvp_only: bool = False, workers: int = 4, redo: bool = False):
    """Extract structured physics parameters with DeepSeek (public papers only)."""
    from src.extraction.deepseek_client import get_llm
    from src.extraction.param_extractor import extract_all
    papers = _scope(mvp_only, "indexed,extracted" if redo else "indexed")
    with _Prog("extract") as pr:
        res = extract_all(_kb(), get_llm(), papers, workers, progress=pr)
    con.print({"papers": len(res), "params": sum(r.get("params", 0) for r in res),
               "verified": sum(r.get("verified", 0) for r in res), "errors": sum("error" in r for r in res),
               "cost_usd": round(get_llm().session_cost, 4)})


@app.command()
def mvp(n: int = typer.Option(settings.mvp_batch_size), harvest_first: bool = False, skip_extract: bool = False):
    """One-shot MVP: (harvest core) → pick N core papers → download → parse → index → extract."""
    from src.acquisition.selector import select_mvp
    if harvest_first:
        harvest(priority=["core"], node=[], category=[], csv=None, refresh=False)
    ids = select_mvp(_kb(), n)
    con.print(f"[green]MVP batch: {len(ids)} papers")
    _download(_scope(True, "selected"), settings.prefer_latex_source)
    _parse_index(_scope(True, "downloaded"))
    if not skip_extract and settings.deepseek_api_key:
        extract(mvp_only=True, workers=4, redo=False)


@app.command("private")
def private_ingest(path: Path = typer.Argument(..., help="File/dir under data/private_local"),
                   node: Optional[str] = None, material: Optional[str] = None):
    """Ingest unpublished local data (never sent to any API). MuMax3 .mx3 params parsed locally."""
    from src.ingestion.private_ingest import ingest_private_path
    from src.retrieval.hybrid_store import HybridStore
    st = ingest_private_path(_kb(), path, node=node, material=material)
    st["embedded"] = HybridStore(_kb()).index_missing()
    con.print(st)


# ---------------------------------------------------------------- query
@app.command()
def search(query: str, k: int = 10, private: bool = False, node: list[str] = typer.Option([])):
    """Raw hybrid search (no LLM)."""
    from src.retrieval.hybrid_store import HybridStore
    tb = Table("score", "tier", "src", "citation", "section", "text")
    for h in HybridStore(_kb()).search(query, k, include_private=private, nodes=node or None):
        c = h.chunk
        tb.add_row(f"{h.score:.4f}", c.priority, "+".join(sorted(h.sources)),
                   f"{c.authors[0] if c.authors else '?'} {c.year}{' 🔒' if c.is_private else ''}",
                   c.section[:25], c.text[:160].replace("\n", " "))
    con.print(tb)


@app.command()
def ask(question: str, node: list[str] = typer.Option([]), priority: list[str] = typer.Option([]),
        show_evidence: bool = False, as_json: bool = False):
    """Agentic RAG answer with claim-level citations (abstains when evidence is insufficient)."""
    from src.retrieval.agentic_rag import AgenticRAG
    ans = AgenticRAG(_kb()).ask(question, nodes=node or None, priorities=priority or None)
    if as_json:
        con.print_json(json.dumps({"status": ans.status, "answer": ans.answer_md, "warnings": ans.warnings,
                                   "claims": ans.claims, "cost_usd": ans.cost_usd}))
        return
    color = {"answered": "green", "partial": "yellow"}.get(ans.status, "red")
    con.print(Panel(Markdown(ans.answer_md), title=f"[{color}]{ans.status}[/] · {ans.seconds:.1f}s · ${ans.cost_usd:.4f}"))
    for w in ans.warnings[:8]:
        con.print(f"[yellow]⚠ {w}")
    if ans.private_hits:
        con.print(Panel("\n".join(f"🔒 {h.chunk.title} — {h.chunk.text[:200]}" for h in ans.private_hits[:5]),
                        title="Private local evidence (shown locally, NOT sent to the LLM)", border_style="magenta"))
    if show_evidence:
        for e in ans.evidence:
            con.print(f"[cyan]{e.eid}[/] {e.citation} ({e.priority}, {e.score:.2f}) {e.summary or e.text[:200]}")


@app.command()
def params(material: Optional[str] = None, quantity: Optional[str] = None, all_: bool = typer.Option(False, "--all", help="include unverified"),
           csv_out: Optional[Path] = None):
    """Cross-paper parameter comparison table from Layer 2 (pure SQL, no LLM)."""
    from src.extraction.param_extractor import compare
    rows = compare(_kb(), material, quantity, verified_only=not all_)
    tb = Table(title=f"{material or 'all'} · {quantity or 'all quantities'} · {len(rows)} rows")
    for c in ("material", "quantity", "value", "unit", "norm", "T (K)", "method", "form/thick", "paper", "✓"):
        tb.add_column(c)
    for r in rows:
        val = r["value"] if r["value"] is not None else f"{r['value_min']}–{r['value_max']}"
        au = json.loads(r["authors"] or "[]")
        tb.add_row(r["material"], r["quantity_label"] or r["quantity"], f"{val}{' ± ' + str(r['uncertainty']) if r['uncertainty'] else ''}",
                   r["unit"] or "", f"{r['value_norm']:.4g} {r['unit_norm']}" if r["value_norm"] is not None else "",
                   str(r["temperature_K"] or ""), r["method"] or "", f"{r['sample_form']} {r['thickness_nm'] or ''}",
                   ("🔒 private" if r["is_private"] else f"{(au[0].split(',')[0] if au else '?')} {r['year']} {r['doi'] or r['arxiv_id'] or ''}"),
                   "✓" if r["quote_verified"] else "?")
    con.print(tb)
    if csv_out:
        import csv
        with open(csv_out, "w", newline="", encoding="utf-8") as f:
            if rows:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
        con.print(f"wrote {csv_out}")


@app.command()
def bench(only: list[str] = typer.Option([]), questions: Optional[Path] = None):
    """Run the MVP benchmark suite and write a JSON report."""
    from src.evaluation.mvp_benchmark import run
    from src.retrieval.agentic_rag import AgenticRAG
    with _Prog("benchmark") as pr:
        res = run(AgenticRAG(_kb()), questions, only or None, progress=pr)
    tb = Table("id", "status", "ret_recall", "cit_recall", "cit_valid", "grounded", "values", "abstain_ok", "s", "$")
    f = lambda v: "-" if v is None else (f"{v:.2f}" if isinstance(v, float) else str(v))
    for r in res["per_question"]:
        tb.add_row(r["id"], r.get("status", ""), *(f(r.get(k)) for k in (
            "retrieval_recall", "citation_recall", "citation_validity", "grounded_claims", "value_accuracy",
            "abstention_correct", "latency_s", "cost_usd")))
    con.print(tb)
    con.print(res["aggregate"])
    con.print(f"report: {res['file']}")


@app.command()
def ui():
    """Launch the Streamlit interface (pip install -e .[ui])."""
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(PROJECT_ROOT / "src" / "ui" / "streamlit_app.py")])


@app.command()
def api(host: str = "127.0.0.1", port: int = 8000):
    """Launch the FastAPI server (pip install -e .[api]); binds to localhost by default."""
    import uvicorn
    uvicorn.run("src.ui.api:app", host=host, port=port)


def main():
    app()


if __name__ == "__main__":
    main()
