# research-kb — curated physics knowledge base with agentic, claim-cited RAG

A local-first research assistant for a **tightly curated ~500-paper corpus** (2D vdW magnets,
skyrmions, DMI, MOKE, MuMax3, …) governed by your keyword taxonomy. It draws on NASA INDUS
(domain-tuned retrieval), PaperQA2 (agentic retrieve → rerank-and-summarise → cited answer) and
OpenScholar (citation-grounded synthesis with self-checks).

```
           ┌──────────── acquisition ────────────┐        ┌──────── ingestion (100% local) ────────┐
 CSV  ───► │ arXiv API (1 req/3 s, 1 connection) │        │ arXiv LaTeX source ─┐                  │
 taxonomy  │ OpenAlex / S2 snowballing           │──dedup─►│ GROBID (local) ─────┼─► ParsedDocument │
 Zotero ──►│ BibTeX/CSL-JSON + PDFs, tag mapping │ (DOI → │ Marker ─────────────┤   (JSON + .md)   │
           └─────────────────────────────────────┘  fuzzy) │ PyMuPDF fallback ───┘        │         │
                                                           └──────────────────────────────┼─────────┘
                                                                structure-aware chunks ◄──┘
 private_local/ ──► local-only ingest (fingerprinted, MuMax3 .mx3 parsed locally) ──┐
                                                                                    ▼
   ┌───────────────── Layer 1: hybrid text ─────────────────┐   ┌──── Layer 2: physics tables ────┐
   │ bge-m3 / SPECTER2 vectors (local) + SQLite FTS5 BM25   │   │ DeepSeek JSON → Pydantic →      │
   │ RRF fusion × priority weight (core > supp. > backgr.)  │   │ quote-verified, unit-normalised │
   └────────────────────────────┬───────────────────────────┘   │ materials/methods/measurements  │
                                │                               └───────────────┬─────────────────┘
                                ▼                                               ▼
        Agentic RAG: decompose → parallel hybrid + SQL → DeepSeek rerank/summarise → sufficiency
        check (+follow-up round) → abstain OR cited synthesis → local claim/number verification
```

## Non-negotiables, and where the code enforces them

| Principle | Enforced in |
|---|---|
| Curated corpus, core-first retrieval | `acquisition/selector.py` (tier shares, node coverage), `retrieval/hybrid_store.py` (priority multiplier), `agentic_rag.rerank` (tier term in final score) |
| arXiv ToU: ≤ 1 request / 3 s, single connection | `core/http.py` `RateLimiter` (persisted across runs) + `httpx.Limits(max_connections=1)`; downloads share the limiter |
| Dedup preprint ↔ published | `ingestion/deduplicator.py`: DOI → arXiv id → OpenAlex/S2 id → fuzzy title (≥93) + first author + year ±2; the journal DOI/venue is kept as the citable version |
| DeepSeek, text only | `extraction/deepseek_client.py` accepts only string message content; nothing uploads files |
| Local parsing and embeddings | `ingestion/pdf_parser.py` (GROBID must be on localhost), `retrieval/hybrid_store.py` |
| **Privacy air-gap** | `core/privacy.py`: (1) `is_private` flags are rejected, (2) **hashed 8-gram fingerprints** of every private document are checked against every outbound payload, (3) private paths are tainted, (4) `core/http.py` egress allowlist. Private evidence is shown locally and never placed in a prompt |
| Dual-layer storage | One SQLite file (`data/db/kb.sqlite`): chunks + FTS5 + vectors + `materials/methods/measurements` + a `kg_edges` view for the Phase-2 graph |
| Claim-level citations, abstention | `agentic_rag.py`: abstains before synthesis if nothing clears the rerank bar; every claim needs `[E#]`; numbers in a claim must appear in the cited evidence; markers are rewritten to `[Author et al., Year, DOI]` |
| MVP first | `kb mvp --n 30` and `kb bench` (18 domain questions in `data/benchmark/mvp_questions.yaml`) |

### Additions beyond the brief
* **arXiv LaTeX sources.** For arXiv papers the e-print is fetched and parsed directly. You get the authors' own LaTeX for equations and tables, which no PDF parser can match. GROBID, Marker and PyMuPDF are fallbacks for PDFs, and GROBID also supplies references.
* **Formula-aware BM25.** `Fe$_3$GaTe$_2$`, `Fe_{3}GaTe_{2}` and `\text{Fe}_3\text{GaTe}_2` are all indexed as `Fe3GaTe2`. Materials also get an order-independent composition key, so `GaFe3Te2` matches `Fe3GaTe2`.
* **Grounded parameter extraction.** Each extracted number must come with a verbatim quote. The quote is fuzzy-matched against the paper, and the number must appear inside it. Records that fail are flagged `quote_verified=0` and left out of comparisons by default. Units are normalised (for example erg/cm² → mJ/m², emu/cm³ → kA/m, Oe → T).
* **MuMax3 private parser.** `Msat`, `Aex`, `Dind`, `Ku1` and `alpha` are read from your `.mx3` scripts locally, with no LLM. Your simulation parameters then appear in the same comparison table as the literature values, marked 🔒.
* **Cost ledger.** Every DeepSeek call is logged in `llm_usage`. A per-run budget limit applies (`DEEPSEEK_BUDGET_USD`).

## Setup (Windows / PowerShell)

```powershell
cd research-kb
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -e ".[dev,ui]"          # core + Streamlit
pip install -e ".[embed]"           # bge-m3 / SPECTER2 (pulls torch, ~2 GB model on first use)
copy .env.example .env               # then put DEEPSEEK_API_KEY=... in .env
```

**GROBID** (optional, recommended for journal PDFs). It needs Docker Desktop: `docker compose up -d grobid`.
Without GROBID the parser uses arXiv LaTeX or PyMuPDF, and Marker if you `pip install -e ".[marker]"`.

**Taxonomy.** Put your CSV (`node_id,node,sub_node,keyword,priority`) in `data/keywords/`. The
`taxonomy_example.csv` there is only a placeholder and is ignored once your file exists.

## MVP workflow (~30 core papers)

```powershell
kb taxonomy                               # validate CSV: 14 nodes / ~419 keywords
kb harvest --priority core                # arXiv, ≥3 s between requests
kb zotero C:\path\to\Library.bib          # paywalled + curated papers (Export Files ticked)
kb snowball --top 10 --core-only          # refs/citations of key reviews, filtered by taxonomy
kb dedupe
kb mvp --n 30                             # pick → download (LaTeX+PDF) → parse → embed → extract
kb status
kb ask "Compare all reported DMI constants and measurement methods for Fe3GaTe2 across the corpus." --show-evidence
kb params --material Fe3GaTe2 --quantity D_int
kb bench                                  # fill gold_dois / expected_values first
kb ui                                     # Streamlit: Ask · Parameter comparison · Corpus
```

Scale-up to ~500: `kb harvest --priority core --priority supporting --priority background`,
then `kb snowball`, `kb select --target 500`, `kb download`, `kb parse`, `kb extract`.

### Zotero tag conventions
`node:<name|id>`, `sub:<sub_node>`, `priority:core|supporting|background` (or a bare tier tag),
or any tag equal to a node, sub_node or keyword in the CSV. `private` marks the item as private, and its PDF is copied into
`data/private_local/zotero/`. Optional explicit mapping: `data/keywords/zotero_tag_map.yaml`.

### Private data
```powershell
kb private data/private_local/mumax_runs --material Fe3GaTe2
```

## Layer-2 schema (Phase-2 KG ready)
`papers` ─< `measurements` >─ `materials` (formula_norm, composition_key), and `measurements` >─ `methods`.
Each measurement holds quantity, value, range, uncertainty, unit, normalised value, T, field, sample form,
thickness, stack, evidence quote, `quote_verified`, confidence and `is_private`. The `kg_edges` view already
emits `Material -HAS_PROPERTY-> Property -MEASURED_BY-> Method` and `Measurement -REPORTED_IN-> Paper`
triples, ready to load into Neo4j or Kùzu later. DuckDB can attach the SQLite file for analytics.

## Tests
`pytest` runs offline. It uses the hash embedder and a fake LLM, and covers the privacy guard, dedup,
the formula/unit normalisation, LaTeX parsing, hybrid search with tier boosting, abstention and claim verification.
