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
# Filled in by `python -m app.llm.models --write` (D4/D16). Empty means
# "not yet discovered"; the app starts but refuses to embed.
DEFAULT_EMBEDDING_MODEL = "gemini-embedding-2"

# 768 is a deliberate choice, not the API default (D16). gemini-embedding-2
# defaults to 3072 and supports 768/1536/3072; at 3072 each chunk's vector is
# ~12 KB against ~3 KB at 768, and Neo4j Aura Free storage is the binding
# constraint. Changing this means rebuilding the Neo4j vector index.
EMBEDDING_DIMENSIONS = 768

# Texts per embedding request. The API accepts a batch, so this is what keeps
# a 200-chunk document to ~2 calls instead of 200 (D14).
EMBEDDING_BATCH_SIZE = 100

VECTOR_TOP_K = 8

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

# --- Access control (D11) --------------------------------------------------
SESSION_COOKIE_NAME = "graphrag_session"
SESSION_TOKEN_PAYLOAD = "authenticated"
SESSION_TTL_SECONDS = 12 * 60 * 60
# Browsers refuse a Secure cookie over plain http, so local development turns
# this off via GRAPHRAG_COOKIE_SECURE=false. Production keeps it on.
SESSION_COOKIE_SECURE = True
# Login attempts allowed per IP per window, so the password cannot be guessed
# at machine speed.
LOGIN_MAX_ATTEMPTS = 8
LOGIN_WINDOW_SECONDS = 300

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
    embedding_model: str = Field(
        default=DEFAULT_EMBEDDING_MODEL, alias="GEMINI_EMBEDDING_MODEL"
    )

    # Provider-agnostic overrides; win over the per-provider values above.
    extraction_model_override: str = Field(default="", alias="GRAPHRAG_EXTRACTION_MODEL")
    answer_model_override: str = Field(default="", alias="GRAPHRAG_ANSWER_MODEL")

    app_password: str = Field(default="", alias="APP_PASSWORD")
    session_secret: str = Field(default="", alias="SESSION_SECRET")
    cookie_secure: bool = Field(default=SESSION_COOKIE_SECURE, alias="GRAPHRAG_COOKIE_SECURE")
    trust_proxy_headers: bool = Field(default=False, alias="GRAPHRAG_TRUST_PROXY")

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

    @staticmethod
    def _first_set(*candidates: str) -> str:
        """The first non-blank candidate.

        A blank environment variable means "not configured", not "configured to
        nothing". Without this, `.env.example`'s empty GEMINI_ANSWER_MODEL=
        shadows the id discovery just wrote into config.py, and every request
        fails with "no model configured" - the exact opposite of what copying
        the template should do.
        """
        for candidate in candidates:
            if candidate and candidate.strip():
                return candidate.strip()
        return ""

    @property
    def extraction_model(self) -> str:
        if self.provider == PROVIDER_GEMINI:
            return self._first_set(
                self.extraction_model_override,
                self.gemini_extraction_model,
                DEFAULT_GEMINI_EXTRACTION_MODEL,
            )
        return self._first_set(
            self.extraction_model_override,
            self.anthropic_extraction_model,
            DEFAULT_ANTHROPIC_EXTRACTION_MODEL,
        )

    @property
    def answer_model(self) -> str:
        if self.provider == PROVIDER_GEMINI:
            return self._first_set(
                self.answer_model_override,
                self.gemini_answer_model,
                DEFAULT_GEMINI_ANSWER_MODEL,
            )
        return self._first_set(
            self.answer_model_override,
            self.anthropic_answer_model,
            DEFAULT_ANTHROPIC_ANSWER_MODEL,
        )

    @property
    def resolved_embedding_model(self) -> str:
        return self._first_set(self.embedding_model, DEFAULT_EMBEDDING_MODEL)

    # --- derived paths ---

    @property
    def resolved_data_dir(self) -> Path:
        if self.data_dir.is_absolute():
            return self.data_dir
        return PROJECT_ROOT / self.data_dir

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
        # D10 keeps application data out of the filesystem; what remains is
        # only scratch space, and Phase 3 removes the rest.
        for path in (self.resolved_data_dir,):
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

    def require_app_password(self) -> str:
        """Return the gate password, or refuse to run.

        This app is reachable from the public internet, so starting without a
        gate would expose the upload endpoint and the API quota behind it. That
        is worth failing loudly for, not warning about.
        """
        password = self.app_password.strip()
        if not password:
            raise ConfigError(
                "APP_PASSWORD is not set, so the app would be open to anyone.",
                hint=(
                    "Set APP_PASSWORD in .env locally, or in the host's "
                    "environment settings when deployed."
                ),
            )
        return password

    def require_embedding_model(self) -> str:
        """Return the embedding model id, or explain how to discover one."""
        model = self.resolved_embedding_model
        if not model:
            raise ConfigError(
                "No embedding model id configured.",
                hint="Run `python -m app.llm.models --write` to discover and store one.",
            )
        return model

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
