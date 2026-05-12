"""
FinSight AI — Global Configuration Settings
Centralizes all environment-driven configuration using Pydantic BaseSettings.

Design notes
------------
* All path constants that must exist at runtime are auto-created by the
  ``_ensure_dirs`` model validator so the application never crashes on a
  missing directory regardless of how the container or venv is set up.
* ``LLM_BASE_URL`` is an optional override for the OpenAI client base URL.
  Leave it unset to use the default OpenAI endpoint.  Set it to the Groq,
  Azure, or Ollama endpoint when using an alternative provider — no code
  changes required.
"""

from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings

BASE_DIR = Path(__file__).resolve().parent.parent


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

    # ── Paths ─────────────────────────────────────────────────────────────────
    DATA_DIR: Path = BASE_DIR / "data"
    RAW_DATA_DIR: Path = BASE_DIR / "data" / "raw"
    PROCESSED_DATA_DIR: Path = BASE_DIR / "data" / "processed"
    EMBEDDINGS_DIR: Path = BASE_DIR / "data" / "embeddings"
    MODELS_DIR: Path = BASE_DIR / "data" / "models"
    LOGS_DIR: Path = BASE_DIR / "logs"

    # ── Database ──────────────────────────────────────────────────────────────
    DATABASE_URL: str = Field(default="sqlite:///./finsight.db", env="DATABASE_URL")
    VECTOR_DB_PATH: str = Field(
        default="data/embeddings/faiss_index", env="VECTOR_DB_PATH"
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

    # Optional base URL override for OpenAI-compatible providers (Groq, Azure,
    # Ollama, etc.).  Leave unset to use the official OpenAI API endpoint.
    LLM_BASE_URL: Optional[str] = Field(default=None, env="LLM_BASE_URL")

    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    # ── RAG Configuration ────────────────────────────────────────────────────
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64
    RAG_TOP_K: int = 5

    # ── FastAPI ───────────────────────────────────────────────────────────────
    API_HOST: str = Field(default="0.0.0.0", env="API_HOST")
    API_PORT: int = Field(default=8000, env="API_PORT")
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8501"]

    # ── Data cache ────────────────────────────────────────────────────────────
    # Number of days before a cached parquet file is considered stale and
    # re-fetched from the data source.  Default: 1 day.
    CACHE_MAX_AGE_DAYS: int = Field(default=1, env="CACHE_MAX_AGE_DAYS")

    @model_validator(mode="after")
    def _ensure_dirs(self) -> "Settings":
        """
        Create all runtime directories eagerly on first load.

        Using a single ``model_validator`` instead of per-field validators
        guarantees that every required directory exists regardless of whether
        Pydantic resolves the fields in a particular order, and avoids the
        silent omission bug where only ``MODELS_DIR`` and ``LOGS_DIR`` were
        previously auto-created.
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
            Path(d).mkdir(parents=True, exist_ok=True)
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
