import json
from typing import Any, TypeVar

from openai import BadRequestError, OpenAI
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


class LLMTruncated(LLMError):
    """The model hit max_tokens; its output is incomplete."""


def extract_json(text: str) -> Any:
    """Parse the first JSON object in `text`, tolerating code fences or stray prose."""
    start = text.find("{")
    if start == -1:
        raise ValueError("no JSON object found in model output")
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[start:])
        return obj
    except json.JSONDecodeError as err:
        # Models quoting legal text often leave inner quotes unescaped
        # ('"quote": "for the words "ten per cent" ..."'). Repair instead of failing.
        from json_repair import repair_json

        repaired = repair_json(text[start:], return_objects=True)
        if isinstance(repaired, dict) and repaired:
            return repaired
        raise ValueError(f"invalid JSON from model: {err}") from err


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
        # Set when the provider rejects the "no thinking" switch, so we stop sending it.
        self._thinking_switch_unsupported = False

    def chat(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        extra_body: dict[str, Any] | None = None,
        thinking: bool | None = None,
    ) -> str:
        """`thinking=None` uses settings.llm_thinking. Nemotron 3.5 is a reasoning model: with
        thinking on, calls take 10-40 s and can spend the whole max_tokens budget thinking."""
        model = model or self.settings.llm_model
        thinking = self.settings.llm_thinking if thinking is None else thinking
        extra_body = dict(extra_body or {})
        auto_switch = (not thinking and not self._thinking_switch_unsupported
                       and "chat_template_kwargs" not in extra_body)
        if auto_switch:
            extra_body["chat_template_kwargs"] = {"enable_thinking": False}

        def create(body: dict[str, Any]):
            self.limiter.acquire()
            return self.client.chat.completions.create(
                model=model,
                messages=messages,  # type: ignore[arg-type]
                temperature=self.settings.llm_temperature if temperature is None else temperature,
                max_tokens=max_tokens or self.settings.llm_max_tokens,
                extra_body=body or None,
            )

        try:
            resp = create(extra_body)
        except BadRequestError as first:
            if not auto_switch:
                raise
            # Maybe the provider rejects the switch: retry once without it. Only remember that
            # if the retry succeeds (an unrelated 400, e.g. context length, must not flip it).
            without = {k: v for k, v in extra_body.items() if k != "chat_template_kwargs"}
            try:
                resp = create(without)
            except BadRequestError:
                raise first from None
            self._thinking_switch_unsupported = True
        choice = resp.choices[0]
        content = choice.message.content
        truncated = choice.finish_reason == "length"
        if not content:
            # Reasoning models can spend the whole budget thinking and return no answer.
            raise (LLMTruncated if truncated else LLMError)(
                f"{model} returned no content (finish_reason={choice.finish_reason}); "
                "try a larger max_tokens"
            )
        if truncated:
            # Cut off at max_tokens: JSON repair would turn this into a plausible but
            # incomplete answer, so fail loudly instead.
            raise LLMTruncated(f"{model} output was cut off at max_tokens")
        return content

    def structured(
        self,
        messages: list[Message],
        schema: type[T],
        *,
        model: str | None = None,
        retries: int = 1,
        max_tokens: int | None = None,
        thinking: bool | None = None,
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
        budget = max_tokens or self.settings.llm_max_tokens
        for _ in range(retries + 1):
            try:
                raw = self.chat(convo, model=model, temperature=0.0, max_tokens=budget,
                                extra_body=extra, thinking=thinking)
            except LLMTruncated as err:
                # Retry once with twice the budget rather than accept a cut-off answer.
                last_error, budget = err, budget * 2
                continue
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
