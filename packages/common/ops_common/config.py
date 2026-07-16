from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, PostgresDsn, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="OPS_",
        extra="ignore",
        case_sensitive=False,
    )

    # ---- Environment ----
    environment: str = Field(default="development")
    debug: bool = Field(default=True)
    log_level: str = Field(default="INFO")

    # ---- Postgres (the hub) ----
    postgres_host: str = Field(default="postgres")
    postgres_port: int = Field(default=5432)
    postgres_db: str = Field(default="ops")
    postgres_user: str = Field(default="ops")
    postgres_password: str = Field(default="ops")

    # ---- DuckDB (analytics) ----
    duckdb_path: str = Field(default="/data/analytics.duckdb")
    duckdb_pg_alias: str = Field(default="pg")

    # ---- API gateway ----
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    api_cors_origins: list[str] = Field(default=["http://localhost:5173"])

    # ---- LLM (mapping suggester + RAG answering) ----
    anthropic_api_key: str = Field(default="")
    llm_model: str = Field(default="claude-sonnet-4-6")
    llm_max_tokens: int = Field(default=1024)
    llm_enabled: bool = Field(default=True)


    # ---- RAG (Level 3 — document assistant) ----
    # Single config block; embedder, schema dimension, retriever and QA chain
    # all read from here. Switching model = change model + dimension only.
    embedding_provider: str = Field(default="local")            # 'local' | 'anthropic' | 'openai'
    embedding_model: str = Field(default="all-MiniLM-L6-v2")
    embedding_dimension: int = Field(default=384)               # must match the model + rag.embeddings column
    rag_chunk_size: int = Field(default=600)                    # tokens (approx) per chunk
    rag_chunk_overlap: int = Field(default=100)
    rag_top_k: int = Field(default=5)                           # chunks retrieved per query
    rag_max_answer_tokens: int = Field(default=800)

    # ---- RAG answer LLM (provider-configurable) ----
    # 'ollama' = local on-premise inference (no key, no cost, no internet).
    # 'anthropic' = cloud API (needs a valid key). Switch via OPS_LLM_PROVIDER.
    llm_provider: str = Field(default="ollama")
    ollama_url: str = Field(default="http://host.docker.internal:11434")
    ollama_model: str = Field(default="llama3.2:3b")

    # ---- Intelligence page LLM polish (independent of RAG) ----
    # Off by default so the Business Intelligence page loads instantly on templates.
    # RAG still uses the LLM (llm_enabled) regardless of this flag.
    intelligence_llm_polish: bool = Field(default=False)
    
    # ---- Auth / JWT (Item 6) ----
    # DEV DEFAULT ONLY — override OPS_JWT_SECRET in .env for any real deployment.
    jwt_secret: str = Field(default="dev-insecure-change-me-in-production")
    jwt_algorithm: str = Field(default="HS256")
    access_token_expire_minutes: int = Field(default=30)       # short-lived access token
    refresh_token_expire_days: int = Field(default=7)          # DB-stored, revocable
    # Google OAuth (Phase 2 — unused until OAuth is wired; safe to leave blank)
    google_client_id: str = Field(default="")
    google_client_secret: str = Field(default="")
    
    # ---- Paths ----
    project_root: Path = Field(default=Path(__file__).resolve().parents[4])
    upload_dir: Path = Field(default=Path("/data/uploads"))
    rag_upload_dir: Path = Field(default=Path("/data/rag_uploads"))
    mapping_config_dir: Path = Field(
        default=Path(__file__).resolve().parent / "mapping_configs"
    )

    # ---- Derived connection strings ----
    @computed_field  # type: ignore[prop-decorator]
    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def sqlalchemy_dsn(self) -> str:
        return (
            f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def duckdb_attach_dsn(self) -> str:
        return (
            f"dbname={self.postgres_db} host={self.postgres_host} "
            f"port={self.postgres_port} user={self.postgres_user} "
            f"password={self.postgres_password}"
        )

    def ensure_dirs(self) -> None:
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.rag_upload_dir.mkdir(parents=True, exist_ok=True)
        self.mapping_config_dir.mkdir(parents=True, exist_ok=True)
        Path(self.duckdb_path).parent.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings


settings = get_settings()