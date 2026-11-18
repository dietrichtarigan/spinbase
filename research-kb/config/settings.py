"""Central configuration (Pydantic settings, read from env / .env).

Every tunable lives here: API keys, privacy flags, rate limits, priority
weights, retrieval knobs. Import the singleton with ``from config.settings import settings``.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent

Priority = Literal["core", "supporting", "background"]
PRIORITIES: tuple[str, ...] = ("core", "supporting", "background")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---------------------------------------------------------------- paths
    data_dir: Path = PROJECT_ROOT / "data"

    @property
    def keywords_dir(self) -> Path: return self.data_dir / "keywords"
    @property
    def raw_pdfs_dir(self) -> Path: return self.data_dir / "raw_pdfs"
    @property
    def latex_src_dir(self) -> Path: return self.data_dir / "raw_latex"
    @property
    def private_dir(self) -> Path: return self.data_dir / "private_local"
    @property
    def parsed_dir(self) -> Path: return self.data_dir / "parsed_md"
    @property
    def db_dir(self) -> Path: return self.data_dir / "db"
    @property
    def db_path(self) -> Path: return self.db_dir / "kb.sqlite"

    taxonomy_csv: Path | None = Field(
        default=None,
        description="Path to the keyword taxonomy CSV. Default: first *.csv in data/keywords/ "
                    "(non-example), else the workspace root.",
    )

    # ------------------------------------------------------------- DeepSeek
    deepseek_api_key: SecretStr | None = None
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"            # e.g. deepseek-chat / deepseek-v4-flash
    deepseek_reasoning_model: str = "deepseek-chat"  # used for synthesis; can be deepseek-reasoner
    deepseek_timeout_s: float = 120.0
    deepseek_max_retries: int = 4
    # USD per 1M tokens (override in .env when pricing changes)
    deepseek_price_input_per_m: float = 0.27
    deepseek_price_input_cached_per_m: float = 0.07
    deepseek_price_output_per_m: float = 1.10
    deepseek_budget_usd: float = 20.0  # hard stop for a single process run

    # -------------------------------------------------------------- privacy
    # When True (default) the DeepSeek client refuses any payload that contains
    # fingerprints of private documents, and private chunks are never placed in prompts.
    privacy_hard_block: bool = True
    privacy_shingle_size: int = 8  # word n-gram length for private-text fingerprints
    # Hosts that the ONLY outbound HTTP helper may contact.
    egress_allowlist: list[str] = [
        "api.deepseek.com", "export.arxiv.org", "arxiv.org",
        "api.openalex.org", "api.semanticscholar.org",
    ]

    # ------------------------------------------------------- acquisition
    arxiv_min_interval_s: float = 3.0  # arXiv ToU: <= 1 request / 3 s, single connection
    arxiv_page_size: int = 50
    arxiv_max_results_per_keyword: dict[str, int] = {"core": 40, "supporting": 15, "background": 5}
    prefer_latex_source: bool = True   # download arXiv e-print (LaTeX) -> perfect equations
    openalex_mailto: str = ""          # optional, joins OpenAlex polite pool (left empty by default)
    semantic_scholar_api_key: SecretStr | None = None
    s2_min_interval_s: float = 1.1
    openalex_min_interval_s: float = 0.15
    corpus_target_size: int = 500
    mvp_batch_size: int = 30

    # ------------------------------------------------------------ parsing
    pdf_parser: Literal["auto", "grobid", "marker", "pymupdf"] = "auto"
    parser_order: list[str] = ["grobid", "marker", "pymupdf"]
    grobid_url: str = "http://localhost:8070"
    grobid_for_references: bool = True

    # ----------------------------------------------------------- chunking
    chunk_target_tokens: int = 380
    chunk_max_tokens: int = 650
    chunk_overlap_sentences: int = 1

    # ---------------------------------------------------------- embeddings
    embedding_backend: Literal["bge-m3", "specter2", "sentence-transformers", "hash"] = "bge-m3"
    embedding_model: str = "BAAI/bge-m3"
    embedding_device: str = "cpu"
    embedding_batch_size: int = 16

    # ----------------------------------------------------------- retrieval
    priority_weights: dict[str, float] = {"core": 1.00, "supporting": 0.80, "background": 0.60}
    rrf_k: int = 60
    dense_weight: float = 1.0
    bm25_weight: float = 1.0
    retrieve_k_per_query: int = 30
    rerank_top_n: int = 12
    rerank_min_score: float = 5.0   # 0-10 scale from the LLM reranker
    min_evidence_for_answer: int = 1
    max_agent_rounds: int = 2
    include_private_in_local_results: bool = True

    @field_validator("priority_weights")
    @classmethod
    def _check_weights(cls, v: dict[str, float]) -> dict[str, float]:
        missing = set(PRIORITIES) - set(v)
        if missing:
            raise ValueError(f"priority_weights missing tiers: {missing}")
        return v

    def ensure_dirs(self) -> None:
        for d in (self.keywords_dir, self.raw_pdfs_dir, self.latex_src_dir, self.private_dir,
                  self.parsed_dir, self.db_dir):
            d.mkdir(parents=True, exist_ok=True)

    def resolve_taxonomy_csv(self) -> Path:
        if self.taxonomy_csv:
            return Path(self.taxonomy_csv)
        cands = [p for p in sorted(self.keywords_dir.glob("*.csv")) if "example" not in p.stem]
        cands += [p for p in sorted(PROJECT_ROOT.glob("*.csv"))]
        cands += [p for p in sorted(PROJECT_ROOT.parent.glob("*.csv"))]
        if cands:
            return cands[0]
        ex = self.keywords_dir / "taxonomy_example.csv"
        if ex.exists():
            return ex
        raise FileNotFoundError("No taxonomy CSV found. Put it in data/keywords/ or set TAXONOMY_CSV.")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    s = Settings()
    s.ensure_dirs()
    return s


settings = get_settings()
