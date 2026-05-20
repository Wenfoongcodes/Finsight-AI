from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings

BASE_DIR = Path(__file__).resolve().parent.parent

# Default local data root — overridden to /data on HuggingFace Spaces
_DEFAULT_DATA_ROOT = BASE_DIR / "data"


class Settings(BaseSettings):
    # ── Project ───────────────────────────────────────────────────────────────
    PROJECT_NAME: str = "FinSight AI"
    VERSION: str = "1.0.0"
    DEBUG: bool = Field(default=False, env="DEBUG")
    ENVIRONMENT: str = Field(default="development", env="ENVIRONMENT")

    # ── API Keys ──────────────────────────────────────────────────────────────
    OPENAI_API_KEY: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    ALPHA_VANTAGE_API_KEY: Optional[str] = Field(
        default=None, env="ALPHA_VANTAGE_API_KEY"
    )

    # ── Data Paths — env-driven so HF Spaces can redirect to /data ───────────
    # Local default: <project_root>/data/*
    # HF Spaces:     /data/*  (persistent volume, set via Dockerfile.hf env)
    DATA_DIR: Path = Field(default=_DEFAULT_DATA_ROOT, env="DATA_DIR")
    RAW_DATA_DIR: Path = Field(default=_DEFAULT_DATA_ROOT / "raw", env="RAW_DATA_DIR")
    PROCESSED_DATA_DIR: Path = Field(
        default=_DEFAULT_DATA_ROOT / "processed", env="PROCESSED_DATA_DIR"
    )
    EMBEDDINGS_DIR: Path = Field(
        default=_DEFAULT_DATA_ROOT / "embeddings", env="EMBEDDINGS_DIR"
    )
    MODELS_DIR: Path = Field(
        default=_DEFAULT_DATA_ROOT / "models", env="MODELS_DIR"
    )
    LOGS_DIR: Path = Field(default=BASE_DIR / "logs", env="LOGS_DIR")

    # ── Database ──────────────────────────────────────────────────────────────
    DATABASE_URL: str = Field(default="sqlite:///./finsight.db", env="DATABASE_URL")
    VECTOR_DB_PATH: str = Field(
        default=str(_DEFAULT_DATA_ROOT / "embeddings" / "faiss_index"),
        env="VECTOR_DB_PATH",
    )

    # ── ML Configuration ──────────────────────────────────────────────────────
    DEFAULT_TICKER: str = "AAPL"
    DEFAULT_PERIOD_YEARS: int = 5
    TRAIN_TEST_SPLIT_DATE: Optional[str] = None
    WALK_FORWARD_FOLDS: int = 5
    RANDOM_SEED: int = 42
    TARGET_COLUMN: str = "target"

    # ── Feature Engineering ───────────────────────────────────────────────────
    RSI_PERIOD: int = 14
    MACD_FAST: int = 12
    MACD_SLOW: int = 26
    MACD_SIGNAL: int = 9
    BB_PERIOD: int = 20
    BB_STD: float = 2.0
    ATR_PERIOD: int = 14
    MOMENTUM_PERIOD: int = 10
    ROLLING_WINDOWS: List[int] = [5, 10, 20, 50]

    # ── LLM Configuration ────────────────────────────────────────────────────
    LLM_MODEL: str = Field(default="gpt-4o-mini", env="LLM_MODEL")
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_TOKENS: int = 1024
    LLM_BASE_URL: Optional[str] = Field(default=None, env="LLM_BASE_URL")
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    # ── RAG Configuration ────────────────────────────────────────────────────
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64
    RAG_TOP_K: int = 5

    # ── FastAPI ───────────────────────────────────────────────────────────────
    API_HOST: str = Field(default="0.0.0.0", env="API_HOST")
    # Port 7860 is the only publicly accessible port on HuggingFace Spaces.
    # Dockerfile.hf sets API_PORT=7860; local dev keeps 8000.
    API_PORT: int = Field(default=8000, env="API_PORT")

    # ── CORS ──────────────────────────────────────────────────────────────────
    # Comma-separated list of allowed CORS origins, or "*" for fully public.
    # On HF Spaces set to your Dashboard Space URL in Space Secrets.
    # Example: "https://YOUR_USERNAME-finsight-dashboard.hf.space"
    ALLOWED_ORIGINS_RAW: str = Field(
        default="http://localhost:3000,http://localhost:8501",
        env="ALLOWED_ORIGINS",
    )

    # ── Security — API Key Auth ───────────────────────────────────────────────
    # Set API_KEY_ENABLED=true and API_SECRET_KEY=<random-secret> to gate
    # every non-health endpoint behind a shared API key.
    API_KEY_ENABLED: bool = Field(default=False, env="API_KEY_ENABLED")
    API_SECRET_KEY: Optional[str] = Field(default=None, env="API_SECRET_KEY")

    # ── Security — Rate Limiting ──────────────────────────────────────────────
    RATE_LIMIT_ENABLED: bool = Field(default=True, env="RATE_LIMIT_ENABLED")
    RATE_LIMIT_MAX_REQUESTS: int = Field(default=120, env="RATE_LIMIT_MAX_REQUESTS")
    RATE_LIMIT_WINDOW_S: int = Field(default=60, env="RATE_LIMIT_WINDOW_S")

    # ── Frontend / Dashboard ──────────────────────────────────────────────────
    # Override in production: FRONTEND_API_BASE=https://your-api.hf.space/api/v1
    FRONTEND_API_BASE: str = Field(
        default="http://localhost:8000/api/v1",
        env="FRONTEND_API_BASE",
    )

    # ── Trusted Proxies ───────────────────────────────────────────────────────
    TRUSTED_PROXIES_RAW: str = Field(
        default="127.0.0.1",
        env="TRUSTED_PROXIES",
    )

    # ── Data cache ────────────────────────────────────────────────────────────
    CACHE_MAX_AGE_DAYS: int = Field(default=1, env="CACHE_MAX_AGE_DAYS")

    # ── Computed properties ───────────────────────────────────────────────────

    @property
    def ALLOWED_ORIGINS(self) -> List[str]:
        """Parse ALLOWED_ORIGINS_RAW into a list."""
        raw = self.ALLOWED_ORIGINS_RAW.strip()
        if raw == "*":
            return ["*"]
        return [o.strip() for o in raw.split(",") if o.strip()]

    @property
    def TRUSTED_PROXIES(self) -> List[str]:
        """Parse TRUSTED_PROXIES_RAW into a list of IPs / CIDR blocks."""
        return [p.strip() for p in self.TRUSTED_PROXIES_RAW.split(",") if p.strip()]

    @model_validator(mode="after")
    def _ensure_dirs(self) -> "Settings":
        """
        Create all runtime directories eagerly on first load.

        On HuggingFace Spaces the /data volume is writable by UID 1000.
        All paths are resolved from env vars so this works identically
        locally and in the cloud.
        """
        dirs_to_create = [
            self.DATA_DIR,
            self.RAW_DATA_DIR,
            self.PROCESSED_DATA_DIR,
            self.EMBEDDINGS_DIR,
            self.MODELS_DIR,
            self.LOGS_DIR,
        ]
        for d in dirs_to_create:
            try:
                Path(d).mkdir(parents=True, exist_ok=True)
            except PermissionError:
                # On some read-only filesystems (e.g. HF build stage) this
                # may fail — that is acceptable; the runtime will have /data.
                pass
        return self

    @model_validator(mode="after")
    def _warn_insecure_config(self) -> "Settings":
        """Emit warnings for insecure production configurations."""
        if self.ENVIRONMENT == "production":
            if not self.API_KEY_ENABLED:
                import warnings
                warnings.warn(
                    "ENVIRONMENT=production but API_KEY_ENABLED=false. "
                    "The API is publicly accessible without authentication. "
                    "Set API_KEY_ENABLED=true and API_SECRET_KEY to secure it.",
                    stacklevel=2,
                )
            if self.ALLOWED_ORIGINS_RAW == "*" and not self.API_KEY_ENABLED:
                import warnings
                warnings.warn(
                    "ALLOWED_ORIGINS=* with API_KEY_ENABLED=false is a fully open API.",
                    stacklevel=2,
                )
        return self

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


@lru_cache
def get_settings() -> Settings:
    """Return cached settings singleton."""
    return Settings()


settings = get_settings()