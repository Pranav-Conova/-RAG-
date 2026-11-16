from collections.abc import Sequence
from typing import Any, Literal

from openai import OpenAI

from asof.config import Settings
from asof.providers.ratelimit import RateLimiter, shared_limiter

InputType = Literal["query", "passage"]


class Embedder:
    """Embeddings via the NVIDIA OpenAI-compatible /embeddings endpoint.

    Retrieval embedders are asymmetric: index documents with input_type="passage"
    and search with input_type="query".
    """

    def __init__(
        self,
        settings: Settings,
        client: OpenAI | None = None,
        limiter: RateLimiter | None = None,
    ) -> None:
        self.settings = settings
        self.client = client or OpenAI(
            api_key=settings.require_api_key(),
            base_url=settings.nim_base_url,
            timeout=settings.request_timeout_s,
            max_retries=settings.max_retries,
        )
        self.limiter = limiter or shared_limiter(settings.nim_requests_per_minute)

    @property
    def dim(self) -> int:
        return self.settings.embed_dim

    def _embed(self, texts: Sequence[str], input_type: InputType) -> list[list[float]]:
        kwargs: dict[str, Any] = {}
        if self.settings.embed_send_dimensions:
            kwargs["dimensions"] = self.dim
        self.limiter.acquire()
        resp = self.client.embeddings.create(
            model=self.settings.embed_model,
            input=list(texts),
            encoding_format="float",
            extra_body={"input_type": input_type, "truncate": "END"},
            **kwargs,
        )
        vectors = [d.embedding for d in sorted(resp.data, key=lambda d: d.index)]
        for v in vectors:
            if len(v) != self.dim:
                raise ValueError(
                    f"{self.settings.embed_model} returned {len(v)}-dim vectors but EMBED_DIM="
                    f"{self.dim}. Fix EMBED_DIM / EMBED_SEND_DIMENSIONS (and the DB schema)."
                )
        return vectors

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        size = self.settings.embed_batch_size
        for i in range(0, len(texts), size):
            out.extend(self._embed(texts[i : i + size], "passage"))
        return out

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], "query")[0]
