"""Application settings and tuning constants.

Every tunable number in the project lives here, so no other module carries a
magic value. Model ids are DISCOVERED, never guessed: run

    python -m app.llm.models --write

with a valid key in .env and it rewrites the two DEFAULT_*_MODEL lines below.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.errors import ConfigError

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Providers -------------------------------------------------------------
# The active LLM provider. Switch with one line in .env: LLM_PROVIDER=anthropic
PROVIDER_GEMINI = "gemini"
PROVIDER_ANTHROPIC = "anthropic"
SUPPORTED_PROVIDERS = (PROVIDER_GEMINI, PROVIDER_ANTHROPIC)
DEFAULT_PROVIDER = PROVIDER_GEMINI

# --- Model ids -------------------------------------------------------------
# Filled in by `python -m app.llm.models --write` (decision D4 in PLAN.md).
# Empty means "not yet discovered"; the app starts but refuses LLM calls.
# Kept per provider so switching back does not reuse the other one's ids.
DEFAULT_GEMINI_EXTRACTION_MODEL = "gemini-3.5-flash-lite"
DEFAULT_GEMINI_ANSWER_MODEL = "gemini-3.8-flash"
DEFAULT_ANTHROPIC_EXTRACTION_MODEL = ""
DEFAULT_ANTHROPIC_ANSWER_MODEL = ""

# --- Chunking (Phase 1) ----------------------------------------------------
CHUNK_TARGET_CHARS = 800
CHUNK_OVERLAP_CHARS = 120
CHUNK_MIN_CHARS = 80
DOC_ID_HASH_LENGTH = 16

# --- Vector search (Phase 2) ----------------------------------------------
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSIONS = 384
VECTOR_TOP_K = 8
CHROMA_COLLECTION_NAME = "graphrag_chunks"

# --- Graph traversal (Phase 6) --------------------------------------------
MAX_HOPS = 3
HOP_DECAY = (1.0, 0.5, 0.25)
MAX_SUBGRAPH_NODES = 60
MAX_ENTITY_DEGREE = 25
PATH_BOOST = 1.5
SEED_NAME_MATCH_THRESHOLD = 0.85

# --- Deduplication (Phase 5, layers 1-5 only; see decision D2) ------------
DEDUP_FUZZY_MERGE_THRESHOLD = 92
DEDUP_FUZZY_GREY_FLOOR = 82
DEDUP_EMBED_MERGE_THRESHOLD = 0.92
DEDUP_EMBED_GREY_FLOOR = 0.85

# --- LLM calls -------------------------------------------------------------
EXTRACTION_MAX_TOKENS = 4096
ANSWER_MAX_TOKENS = 16000
LLM_TIMEOUT_SECONDS = 120.0

# Free-tier friendly: 2 concurrent extraction calls, not 5. A free Gemini key
# has a low requests-per-minute ceiling, and going wider just trips 429s.
EXTRACTION_CONCURRENCY = 2

# Thinking budget for per-chunk extraction. 0 disables internal reasoning,
# which is both faster and cheaper on a quota measured in requests.
EXTRACTION_THINKING_BUDGET = 0

# --- Retry / backoff on rate limits ---------------------------------------
RETRY_MAX_ATTEMPTS = 5
RETRY_BASE_DELAY_SECONDS = 2.0
RETRY_MAX_DELAY_SECONDS = 60.0

# --- Logging ---------------------------------------------------------------
VALID_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")

# --- Uploads ---------------------------------------------------------------
# 1 MB, per decision D14. A 1 MB document is roughly 200 chunks, which is the
# quota budget: ~2 batched embedding calls plus ~40 batched extraction calls.
MAX_UPLOAD_BYTES = 1 * 1024 * 1024
SUPPORTED_EXTENSIONS = (".pdf", ".txt", ".md")


class Settings(BaseSettings):
    """Values read from the environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    llm_provider: str = Field(default=DEFAULT_PROVIDER, alias="LLM_PROVIDER")

    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")

    # Per-provider model ids, written by discovery.
    gemini_extraction_model: str = Field(
        default=DEFAULT_GEMINI_EXTRACTION_MODEL, alias="GEMINI_EXTRACTION_MODEL"
    )
    gemini_answer_model: str = Field(
        default=DEFAULT_GEMINI_ANSWER_MODEL, alias="GEMINI_ANSWER_MODEL"
    )
    anthropic_extraction_model: str = Field(
        default=DEFAULT_ANTHROPIC_EXTRACTION_MODEL, alias="ANTHROPIC_EXTRACTION_MODEL"
    )
    anthropic_answer_model: str = Field(
        default=DEFAULT_ANTHROPIC_ANSWER_MODEL, alias="ANTHROPIC_ANSWER_MODEL"
    )

    # Provider-agnostic overrides; win over the per-provider values above.
    extraction_model_override: str = Field(default="", alias="GRAPHRAG_EXTRACTION_MODEL")
    answer_model_override: str = Field(default="", alias="GRAPHRAG_ANSWER_MODEL")

    data_dir: Path = Field(default=Path("data"), alias="GRAPHRAG_DATA_DIR")
    log_level: str = Field(default="INFO", alias="GRAPHRAG_LOG_LEVEL")

    @field_validator("log_level")
    @classmethod
    def _known_log_level(cls, value: str) -> str:
        """Reject a bad level here, not inside logging at startup.

        `logging.setLevel("verbose")` raises a bare ValueError from the
        lifespan handler, which kills boot with a traceback instead of one of
        our typed errors.
        """
        normalised = value.strip().upper()
        if normalised not in VALID_LOG_LEVELS:
            raise ValueError(
                f"GRAPHRAG_LOG_LEVEL must be one of {', '.join(VALID_LOG_LEVELS)}; "
                f"got {value!r}"
            )
        return normalised

    @field_validator("llm_provider")
    @classmethod
    def _known_provider(cls, value: str) -> str:
        normalised = value.strip().lower()
        if normalised not in SUPPORTED_PROVIDERS:
            raise ValueError(
                f"LLM_PROVIDER must be one of {', '.join(SUPPORTED_PROVIDERS)}; "
                f"got {value!r}"
            )
        return normalised

    # --- active provider's models ---

    @property
    def provider(self) -> str:
        return self.llm_provider

    @property
    def extraction_model(self) -> str:
        if self.extraction_model_override.strip():
            return self.extraction_model_override.strip()
        if self.provider == PROVIDER_GEMINI:
            return self.gemini_extraction_model.strip()
        return self.anthropic_extraction_model.strip()

    @property
    def answer_model(self) -> str:
        if self.answer_model_override.strip():
            return self.answer_model_override.strip()
        if self.provider == PROVIDER_GEMINI:
            return self.gemini_answer_model.strip()
        return self.anthropic_answer_model.strip()

    # --- derived paths ---

    @property
    def resolved_data_dir(self) -> Path:
        if self.data_dir.is_absolute():
            return self.data_dir
        return PROJECT_ROOT / self.data_dir

    @property
    def chroma_dir(self) -> Path:
        return self.resolved_data_dir / "chroma"

    @property
    def uploads_dir(self) -> Path:
        return self.resolved_data_dir / "uploads"

    @property
    def graph_path(self) -> Path:
        return self.resolved_data_dir / "graph.json"

    @property
    def extraction_cache_dir(self) -> Path:
        return self.resolved_data_dir / "extraction_cache"

    @property
    def merge_log_path(self) -> Path:
        return self.resolved_data_dir / "merge_log.jsonl"

    @property
    def dedup_candidates_path(self) -> Path:
        return self.resolved_data_dir / "dedup_candidates.jsonl"

    def ensure_dirs(self) -> None:
        for path in (
            self.resolved_data_dir,
            self.chroma_dir,
            self.uploads_dir,
            self.extraction_cache_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    # --- validation ---

    def require_api_key(self) -> str:
        """Return the active provider's key, or fail loudly.

        Only the active provider's key is required: running on Gemini does not
        need an Anthropic key, and vice versa.
        """
        if self.provider == PROVIDER_GEMINI:
            key = self.gemini_api_key.strip()
            if not key:
                raise ConfigError(
                    "GEMINI_API_KEY is not set (LLM_PROVIDER=gemini).",
                    hint=(
                        "Copy .env.example to .env and add a free key from "
                        "https://aistudio.google.com/apikey"
                    ),
                )
            return key

        key = self.anthropic_api_key.strip()
        if not key:
            raise ConfigError(
                "ANTHROPIC_API_KEY is not set (LLM_PROVIDER=anthropic).",
                hint=(
                    "Add your key from https://console.anthropic.com/settings/keys, "
                    "or set LLM_PROVIDER=gemini to use the free Gemini tier."
                ),
            )
        return key

    def require_models(self) -> tuple[str, str]:
        """Return (extraction_model, answer_model) or explain how to get them."""
        missing = [
            name
            for name, value in (
                ("extraction", self.extraction_model),
                ("answer", self.answer_model),
            )
            if not value.strip()
        ]
        if missing:
            raise ConfigError(
                f"No {self.provider} model id configured for: {', '.join(missing)}.",
                hint="Run `python -m app.llm.models --write` to discover and store them.",
            )
        return self.extraction_model.strip(), self.answer_model.strip()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
