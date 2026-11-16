"""Render aggregated metrics as summary.md (human) and summary.json (machine)."""

from typing import Any

from asof.evaluation.questions import QUESTION_TYPES

# (key, label, kind) - kind: "rate" (share with k/n), "mean" (average with n), "ms", "count"
METRIC_ROWS: list[tuple[str, str, str]] = [
    ("n_questions", "Questions", "count"),
    ("n_errors", "Errors (excluded from rates)", "count"),
    ("answer_accuracy", "Answer accuracy (key facts present, no stale facts)", "rate"),
    ("recall@1", "Retrieval recall@1 (gold doc ranked first)", "rate"),
    ("recall@5", "Retrieval recall@5", "rate"),
    ("recall@10", "Retrieval recall@10", "rate"),
    ("citation_precision", "Citation precision (cited docs that are gold)", "mean"),
    ("citation_hit_rate", "Citation hit (>= 1 verified citation to a gold doc)", "rate"),
    ("stale_source_rate", "Stale-source rate (cites a doc not in force on as_of) - lower is better",
     "rate"),
    ("abstention_accuracy", "Abstention accuracy", "rate"),
    ("latency_p50_ms", "Latency p50", "ms"),
    ("latency_p95_ms", "Latency p95", "ms"),
]

TYPE_COLUMNS: list[tuple[str, str]] = [
    ("answer_accuracy", "answer"),
    ("recall@5", "R@5"),
    ("stale_source_rate", "stale"),
    ("abstention_accuracy", "abstain"),
]


def fmt(value: Any, kind: str) -> str:
    if value is None:
        return "n/a"
    if kind == "count":
        return str(value)
    if kind == "ms":
        return f"{value / 1000:.1f} s" if value >= 1000 else f"{value:.0f} ms"
    v, den = value.get("value"), value.get("den", 0)
    if v is None:
        return "n/a"
    if kind == "rate":
        return f"{v * 100:.1f}% ({value['num']}/{den})"
    return f"{v:.2f} (n={den})"


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def render_markdown(summary: dict[str, Any]) -> str:
    systems: dict[str, Any] = summary["systems"]
    names = list(systems)
    out = [
        "# AsOf evaluation summary",
        "",
        f"- Generated: {summary['generated_at']}",
        f"- Questions file: `{summary['questions_file']}`",
        f"- Systems: {', '.join(f'`{n}`' for n in names)}",
    ]
    for n in names:
        files = ", ".join(f"`{f}`" for f in systems[n].get("results_files", []))
        out.append(f"- `{n}` results: {files or 'n/a'}")
    if summary.get("incomplete"):
        out.append(f"- **Incomplete run:** {summary['incomplete']}")
    out += [
        "",
        "Rates show the share and the raw count `(successes/applicable questions)`; a metric only "
        "counts questions it applies to (e.g. recall needs a gold document). With small question "
        "counts, differences of a few points are noise. See `eval/README.md` for definitions.",
        "",
        "## Overall",
        "",
    ]
    rows = [[label] + [fmt(systems[n]["metrics"].get(key), kind) for n in names]
            for key, label, kind in METRIC_ROWS]
    out.append(_table(["Metric"] + [f"`{n}`" for n in names], rows))

    types = [t for t in QUESTION_TYPES if any(t in systems[n]["per_type"] for n in names)]
    if types:
        out += ["", "## By question type", ""]
        header = ["Type", "n"] + [f"{n} {label}" for n in names for _, label in TYPE_COLUMNS]
        trows = []
        for t in types:
            first = next(systems[n]["per_type"][t] for n in names if t in systems[n]["per_type"])
            row = [t, str(first["n_questions"])]
            for n in names:
                m = systems[n]["per_type"].get(t, {})
                row += [fmt(m.get(key), "rate") for key, _ in TYPE_COLUMNS]
            trows.append(row)
        out.append(_table(header, trows))
    return "\n".join(out) + "\n"
