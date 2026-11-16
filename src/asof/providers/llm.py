import json
from typing import Any, TypeVar

from openai import OpenAI
from pydantic import BaseModel, ValidationError

from asof.config import Settings
from asof.providers.ratelimit import RateLimiter, shared_limiter

T = TypeVar("T", bound=BaseModel)

Message = dict[str, Any]

_JSON_INSTRUCTION = (
    "Respond with a single JSON object that conforms to the JSON Schema below. "
    "Output only the JSON object: no prose, no markdown, no code fences.\n\nJSON Schema:\n{schema}"
)


class LLMError(RuntimeError):
    pass


def extract_json(text: str) -> Any:
    """Parse the first JSON object in `text`, tolerating code fences or stray prose."""
    start = text.find("{")
    if start == -1:
        raise ValueError("no JSON object found in model output")
    obj, _ = json.JSONDecoder().raw_decode(text[start:])
    return obj


class LLM:
    """Chat + structured output over the OpenAI-compatible NVIDIA endpoint."""

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
            max_retries=settings.max_retries,  # SDK backs off on 429 / 5xx
        )
        self.limiter = limiter or shared_limiter(settings.nim_requests_per_minute)

    def chat(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> str:
        model = model or self.settings.llm_model
        self.limiter.acquire()
        resp = self.client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            temperature=self.settings.llm_temperature if temperature is None else temperature,
            max_tokens=max_tokens or self.settings.llm_max_tokens,
            extra_body=extra_body,
        )
        choice = resp.choices[0]
        content = choice.message.content
        if not content:
            # Reasoning models can spend the whole budget thinking and return no answer.
            raise LLMError(
                f"{model} returned no content (finish_reason={choice.finish_reason}); "
                "try a larger max_tokens"
            )
        return content

    def structured(
        self,
        messages: list[Message],
        schema: type[T],
        *,
        model: str | None = None,
        retries: int = 1,
        max_tokens: int | None = None,
    ) -> T:
        """Return a validated `schema` instance; on bad JSON, show the model its error and retry."""
        json_schema = schema.model_json_schema()
        convo: list[Message] = [
            {"role": "system", "content": _JSON_INSTRUCTION.format(schema=json.dumps(json_schema))},
            *messages,
        ]
        extra = {"nvext": {"guided_json": json_schema}} if self.settings.llm_guided_json else None
        model = model or self.settings.llm_fast_model

        last_error: Exception | None = None
        for _ in range(retries + 1):
            raw = self.chat(convo, model=model, temperature=0.0, max_tokens=max_tokens,
                            extra_body=extra)
            try:
                return schema.model_validate(extract_json(raw))
            except (ValueError, ValidationError) as err:
                last_error = err
                convo += [
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": f"That output was invalid: {err}. "
                                                "Return only the corrected JSON object."},
                ]
        raise LLMError(f"{model} did not return valid {schema.__name__}: {last_error}")
