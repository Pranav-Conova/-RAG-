# AsOf evaluation summary

- Generated: 2026-11-16T21:20:17+00:00
- Questions file: `eval/benchmark.jsonl`
- Systems: `asof`, `baseline`
- `asof` results: `eval/results/20261116-212017-asof.jsonl`
- `baseline` results: `eval/results/20261116-212017-baseline.jsonl`
- **Incomplete run:** stopped after 3 consecutive rate-limit errors; rerun later with --resume

Rates show the share and the raw count `(successes/applicable questions)`; a metric only counts questions it applies to (e.g. recall needs a gold document). With small question counts, differences of a few points are noise. See `eval/README.md` for definitions.

## Overall

| Metric | `asof` | `baseline` |
|---|---|---|
| Questions | 49 | 5 |
| Errors (excluded from rates) | 1 | 3 |
| Answer accuracy (key facts present, no stale facts) | 59.0% (23/39) | 50.0% (1/2) |
| Retrieval recall@1 (gold doc ranked first) | 48.7% (19/39) | 0.0% (0/2) |
| Retrieval recall@5 | 74.4% (29/39) | 50.0% (1/2) |
| Retrieval recall@10 | 84.6% (33/39) | 50.0% (1/2) |
| Citation precision (cited docs that are gold) | 0.59 (n=37) | 0.25 (n=2) |
| Citation hit (>= 1 verified citation to a gold doc) | 71.8% (28/39) | 50.0% (1/2) |
| Stale-source rate (cites a doc not in force on as_of) - lower is better | 2.6% (1/38) | 0.0% (0/2) |
| Abstention accuracy | 93.8% (45/48) | 100.0% (2/2) |
| Latency p50 | 7.3 s | 3.6 s |
| Latency p95 | 16.1 s | 3.9 s |

## By question type

| Type | n | asof answer | asof R@5 | asof stale | asof abstain | baseline answer | baseline R@5 | baseline stale | baseline abstain |
|---|---|---|---|---|---|---|---|---|---|
| point_in_time | 32 | 56.2% (18/32) | 71.9% (23/32) | 0.0% (0/30) | 93.8% (30/32) | 50.0% (1/2) | 50.0% (1/2) | 0.0% (0/2) | 100.0% (2/2) |
| current | 1 | 100.0% (1/1) | 100.0% (1/1) | 0.0% (0/1) | 100.0% (1/1) | n/a | n/a | n/a | n/a |
| history | 2 | 0.0% (0/2) | 100.0% (2/2) | 50.0% (1/2) | 100.0% (2/2) | n/a | n/a | n/a | n/a |
| supersession_trap | 2 | 100.0% (2/2) | 50.0% (1/2) | 0.0% (0/2) | 100.0% (2/2) | n/a | n/a | n/a | n/a |
| unanswerable | 10 | n/a | n/a | 0.0% (0/1) | 88.9% (8/9) | n/a | n/a | n/a | n/a |
| lookup | 2 | 100.0% (2/2) | 100.0% (2/2) | 0.0% (0/2) | 100.0% (2/2) | n/a | n/a | n/a | n/a |
