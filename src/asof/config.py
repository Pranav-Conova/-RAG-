from functools import lru_cache
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

RERANK_URL_TEMPLATE = "https://ai.api.nvidia.com/v1/retrieval/{model}/reranking"


class Settings(BaseSettings):
    """All runtime configuration, read from environment variables / .env."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # NVIDIA API catalog
    nvidia_api_key: SecretStr = SecretStr("")
    nim_base_url: str = "https://integrate.api.nvidia.com/v1"
    nim_requests_per_minute: int = 35
    request_timeout_s: float = 90.0
    max_retries: int = 5

    # LLMs. NVIDIA retires catalog models often: `asof smoke` checks these are still live.
    llm_model: str = "nvidia/nemotron-3.5-super-vl-120b-a12b"
    llm_fast_model: str = "nvidia/nemotron-3.5-super-vl-120b-a12b"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 4096
    llm_guided_json: bool = False

    # Embeddings
    embed_model: str = "nvidia/nemotron-3.5-embed-1b"
    embed_dim: int = 1024
    embed_send_dimensions: bool = True
    embed_batch_size: int = 32

    # Reranker: "openrouter" (hosted cross-encoder, falls back to llm), "llm" (fast LLM scores
    # passages), "nim" (NVIDIA hosted cross-encoder), or "none"
    rerank_backend: Literal["openrouter", "llm", "nim", "none"] = "llm"

    # OpenRouter (reranking). Free models: 20 req/min; 50 req/day, or 1000/day once the account
    # has bought >= $10 of credits in total.
    openrouter_api_key: SecretStr = SecretStr("")
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_rerank_model: str = "nvidia/nemotron-3.5-rerank-8b:free"
    openrouter_requests_per_minute: int = 18

    rerank_model: str = "nvidia/nemotron-3.5-rerank-8b"  # used only when backend is "nim"
    rerank_url: str = ""

    # Database
    database_url: str = "postgresql://asof:asof@localhost:5432/asof"

    @property
    def resolved_rerank_url(self) -> str:
        if self.rerank_url:
            return self.rerank_url
        # API catalog paths replace dots with underscores, e.g. llama-3.2 -> llama-3_2
        return RERANK_URL_TEMPLATE.format(model=self.rerank_model.replace(".", "_"))

    def require_api_key(self) -> str:
        key = self.nvidia_api_key.get_secret_value()
        if not key:
            raise RuntimeError(
                "NVIDIA_API_KEY is not set. Create one at build.nvidia.com and put it in .env"
            )
        return key


@lru_cache
def get_settings() -> Settings:
    return Settings()
