# AsOf evaluation summary

- Generated: 2026-11-16T21:06:40+00:00
- Questions file: `eval/benchmark.jsonl`
- Systems: `asof`, `baseline`
- `asof` results: `eval/results/20261116-205419-asof.jsonl`
- `baseline` results: `eval/results/20261116-205419-baseline.jsonl`

Rates show the share and the raw count `(successes/applicable questions)`; a metric only counts questions it applies to (e.g. recall needs a gold document). With small question counts, differences of a few points are noise. See `eval/README.md` for definitions.

## Overall

| Metric | `asof` | `baseline` |
|---|---|---|
| Questions | 49 | 49 |
| Errors (excluded from rates) | 0 | 0 |
| Answer accuracy (key facts present, no stale facts) | 64.1% (25/39) | 38.5% (15/39) |
| Retrieval recall@1 (gold doc ranked first) | 59.0% (23/39) | 20.5% (8/39) |
| Retrieval recall@5 | 79.5% (31/39) | 46.2% (18/39) |
| Retrieval recall@10 | 84.6% (33/39) | 53.8% (21/39) |
| Citation precision (cited docs that are gold) | 0.65 (n=37) | 0.47 (n=30) |
| Citation hit (>= 1 verified citation to a gold doc) | 76.9% (30/39) | 41.0% (16/39) |
| Stale-source rate (cites a doc not in force on as_of) - lower is better | 0.0% (0/39) | 28.6% (10/35) |
| Abstention accuracy | 89.8% (44/49) | 69.4% (34/49) |
| Latency p50 | 8.3 s | 3.2 s |
| Latency p95 | 16.1 s | 8.1 s |

## By question type

| Type | n | asof answer | asof R@5 | asof stale | asof abstain | baseline answer | baseline R@5 | baseline stale | baseline abstain |
|---|---|---|---|---|---|---|---|---|---|
| point_in_time | 32 | 62.5% (20/32) | 78.1% (25/32) | 0.0% (0/30) | 87.5% (28/32) | 31.2% (10/32) | 40.6% (13/32) | 30.4% (7/23) | 59.4% (19/32) |
| current | 1 | 100.0% (1/1) | 100.0% (1/1) | 0.0% (0/1) | 100.0% (1/1) | 0.0% (0/1) | 0.0% (0/1) | 0.0% (0/1) | 0.0% (0/1) |
| history | 2 | 0.0% (0/2) | 100.0% (2/2) | 0.0% (0/2) | 100.0% (2/2) | 50.0% (1/2) | 100.0% (2/2) | 0.0% (0/2) | 100.0% (2/2) |
| supersession_trap | 2 | 100.0% (2/2) | 50.0% (1/2) | 0.0% (0/2) | 100.0% (2/2) | 100.0% (2/2) | 50.0% (1/2) | 0.0% (0/2) | 100.0% (2/2) |
| unanswerable | 10 | n/a | n/a | 0.0% (0/2) | 90.0% (9/10) | n/a | n/a | 60.0% (3/5) | 90.0% (9/10) |
| lookup | 2 | 100.0% (2/2) | 100.0% (2/2) | 0.0% (0/2) | 100.0% (2/2) | 100.0% (2/2) | 100.0% (2/2) | 0.0% (0/2) | 100.0% (2/2) |
