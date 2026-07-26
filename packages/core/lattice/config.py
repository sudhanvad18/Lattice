"""
Lattice platform configuration.

All settings are loaded from environment variables or .env file.
Pydantic Settings validates types and provides defaults for local development.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class InferenceProvider(str, Enum):
    OLLAMA = "ollama"
    ANTHROPIC = "anthropic"
    OPENAI = "openai"


class GraphBackend(str, Enum):
    NETWORKX = "networkx"
    NEO4J = "neo4j"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="LATTICE_",
        case_sensitive=False,
    )

    # --- Inference ---
    default_provider: InferenceProvider = InferenceProvider.OLLAMA
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"
    anthropic_api_key: Optional[str] = None
    anthropic_model: str = "claude-sonnet-4-20250514"
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4o"

    # --- Knowledge Graph ---
    graph_backend: GraphBackend = GraphBackend.NETWORKX
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: Optional[str] = None

    # --- Vector Store ---
    chroma_persist_dir: Path = Path("data/chroma")
    embedding_model: str = "all-MiniLM-L6-v2"

    # --- Redis (optional for local dev) ---
    redis_url: Optional[str] = None

    # --- GitHub (for write-back) ---
    github_token: Optional[str] = None

    # --- Observability ---
    langfuse_public_key: Optional[str] = None
    langfuse_secret_key: Optional[str] = None
    langfuse_host: str = "https://cloud.langfuse.com"
    enable_tracing: bool = False

    # --- General ---
    data_dir: Path = Path("data")
    log_level: str = "INFO"
    debug: bool = False


def get_settings() -> Settings:
    """Load settings (cached at module level after first call)."""
    return Settings()
