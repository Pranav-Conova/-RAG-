"""Command line entry point: `asof db-init`, `asof smoke`."""

import argparse
import sys
import time
from collections.abc import Callable

from pydantic import BaseModel

from asof.config import Settings, get_settings


def cmd_db_init(settings: Settings) -> int:
    from asof.db import init_schema

    init_schema(settings)
    print(f"Schema ready (embedding dim {settings.embed_dim}, model {settings.embed_model}).")
    return 0


class _Ping(BaseModel):
    answer: str
    number: int


def cmd_smoke(settings: Settings) -> int:
    """Hit every configured NVIDIA model once so config problems show up immediately."""
    import httpx

    from asof.providers import LLM, Embedder, make_reranker

    query = "What was the provisional ITC cap under rule 36(4) in February 2020?"
    docs = [
        "Rule 36(4): ITC in respect of invoices not uploaded shall not exceed 10 per cent ...",
        "Section 50: interest is payable on delayed payment of tax ...",
        "Rule 86B: restriction on use of amount available in electronic credit ledger ...",
    ]
    failures = 0

    def step(name: str, fn: Callable[[], str]) -> None:
        nonlocal failures
        t0 = time.perf_counter()
        try:
            detail = fn()
            print(f"  OK   {name:<38} {time.perf_counter() - t0:5.1f}s  {detail}")
        except Exception as err:  # noqa: BLE001 - report and continue
            failures += 1
            print(f"  FAIL {name:<38} {type(err).__name__}: {err}")

    print("NVIDIA API catalog smoke test")

    def catalog() -> str:
        # NVIDIA retires catalog models regularly (HTTP 410). Catch that before spending calls.
        resp = httpx.get(f"{settings.nim_base_url}/models", timeout=30)
        resp.raise_for_status()
        live = {m["id"] for m in resp.json()["data"]}
        wanted = {settings.llm_model, settings.llm_fast_model, settings.embed_model}
        missing = sorted(wanted - live)
        if missing:
            raise RuntimeError(f"not in catalog (retired?): {missing}")
        return f"{len(live)} live models, all configured chat/embed models present"

    step("catalog", catalog)
    llm = LLM(settings)
    for model in dict.fromkeys([settings.llm_model, settings.llm_fast_model]):
        step(f"chat  {model}", lambda m=model: llm.chat(
            [{"role": "user", "content": "Reply with exactly the word: pong"}],
            model=m, max_tokens=1024).strip()[:40])
        step(f"json  {model}", lambda m=model: repr(llm.structured(
            [{"role": "user", "content": "Set answer to 'pong' and number to 42."}],
            _Ping, model=m, max_tokens=1024)))

    emb = Embedder(settings)
    step(f"embed {settings.embed_model}",
         lambda: f"query dim={len(emb.embed_query(query))}, "
                 f"docs={len(emb.embed_documents(docs))}")

    labels = {"nim": settings.rerank_model, "openrouter": settings.openrouter_rerank_model}
    label = labels.get(settings.rerank_backend, settings.rerank_backend)

    def rerank() -> str:
        # fallback=False: the smoke test must expose a failing primary, not paper over it.
        out = make_reranker(settings, fallback=False).rerank(query, docs)
        return (f"order={[r.index for r in out]} (expect 0 first) "
                f"scores={[round(r.score, 3) for r in out]}")

    step(f"rerank {label}", rerank)

    if settings.rerank_backend == "openrouter":
        def quota() -> str:
            resp = httpx.get(f"{settings.openrouter_base_url}/key", timeout=30, headers={
                "Authorization": f"Bearer {settings.openrouter_api_key.get_secret_value()}"})
            resp.raise_for_status()
            data = resp.json().get("data", {})
            daily = data.get("free_model_daily_requests")
            return f"free tier={data.get('is_free_tier')}, free-model requests today={daily}"

        step("openrouter key/quota", quota)

    print("All good." if not failures else f"{failures} step(s) failed.")
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="asof")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("db-init", help="create/upgrade the Postgres schema")
    sub.add_parser("smoke", help="call each configured NVIDIA model once")
    args = parser.parse_args(argv)

    settings = get_settings()
    commands = {"db-init": cmd_db_init, "smoke": cmd_smoke}
    return commands[args.command](settings)


if __name__ == "__main__":
    sys.exit(main())
