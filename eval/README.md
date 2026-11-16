# AsOf evaluation

This folder holds the question set used to measure AsOf, and the results of each run.
The harness itself lives in `src/asof/evaluation/`.

The aim is to answer one question with numbers: **does time-aware retrieval give more correct,
less outdated answers than ordinary RAG over the same documents?** Every question is run through
both systems:

- `asof`: the full pipeline, which keeps only text in force on the question's date.
- `baseline`: the same corpus and model, with no date filtering.

## Files

| File | What it is |
|---|---|
| `questions.jsonl` | The evaluation questions, one JSON object per line (format below) |
| `candidates.jsonl` | Draft questions from the generator, for human review (not used by runs) |
| `results/<timestamp>-<system>.jsonl` | One record per question: the full answer, sources and scores |
| `results/summary.md`, `results/summary.json` | Metric tables comparing the systems |

`results/` is gitignored apart from `.gitkeep`. Commit a summary on purpose when you want to publish it.

## Question format

```json
{"id": "pit-36-4-2020-02",
 "question": "What was the provisional ITC limit under rule 36(4) in February 2020?",
 "as_of": "2020-02-15",
 "type": "point_in_time",
 "gold_doc_numbers": ["75/2019-Central Tax"],
 "must_contain": ["10 per cent"],
 "must_not_contain": ["5 per cent"],
 "notes": "Why this is the right answer, and anything to double-check"}
```

- `as_of`: the date the question is about (`YYYY-MM-DD`), or `null` for "today". It is passed
  to the system as the as-of date.
- `gold_doc_numbers`: the canonical numbers of the documents that state the answer
  (e.g. `94/2020-Central Tax`, `123/42/2019-GST`). Leading zeros, spacing and a
  "Notification No." prefix are ignored when comparing.
- `must_contain`: short key facts that a correct answer states. Keep them short: a number with
  its unit, a form name. Equivalent phrasings can be joined with `||`, e.g.
  `"one crore and fifty lakh || 1.5 crore"`; any one of them counts.
- `must_not_contain`: facts that would show the answer used the wrong period, usually a value
  that only applied later. Do not list a value that a good answer might mention as history.

The loader (`asof.evaluation.questions.load_questions`) checks every line and reports all
problems with line numbers. Its rules: ids are unique; unanswerable questions have no
`must_contain`; other types need a gold document or a key fact; `point_in_time` and
`supersession_trap` need a date; `current` must have `as_of: null`; and no fact can be in both
lists.

### Question types

| Type | What it tests | Example |
|---|---|---|
| `point_in_time` | The rule on a past date | "What was the rule 36(4) cap in February 2020?" |
| `current` | The rule today, after all amendments | "Is the rule 36(4) provisional credit still available today?" |
| `history` | How a rule changed over time | "How did the rule 36(4) cap change from 2019 to 2022?" |
| `supersession_trap` | A date where the most similar text is a later amendment, or one issued but not yet in force | "On 28 Dec 2019, what was the cap?" (the 10% amendment was issued on 26 Dec but applied from 1 Jan) |
| `unanswerable` | The system should decline: outside the corpus, or before the provision existed | "What was the rule 36(4) cap in August 2019?" (the rule did not exist yet) |
| `lookup` | Finding a specific document or fact | "What topic does Circular 123/42/2019-GST clarify?" |

## Metrics

All metrics are computed in code, in `src/asof/evaluation/metrics.py`. No LLM judges the
answers. Each metric counts only the questions it applies to, and the summary shows the raw
count next to every percentage, e.g. `75.0% (6/8)`. Questions that raised an error are counted
separately and left out of every rate.

| Metric | Meaning |
|---|---|
| **Answer accuracy** | The answer contains every `must_contain` fact and no `must_not_contain` fact. An abstention counts as wrong. Only for answerable questions that have facts. |
| **Recall@k** (k = 1, 5, 10) | A gold document appears among the top k retrieved passages. |
| **Citation precision** | Of the distinct documents the answer cites, the share that are gold, averaged over questions. |
| **Citation hit** | The answer has at least one *verified* citation (the quote was found in the source) to a gold document. |
| **Stale-source rate** | The share of answers that cite at least one passage not in force on the question's date. Lower is better. This is the headline temporal metric. |
| **Abstention accuracy** | The system declined on unanswerable questions and answered the others. |
| **Latency p50 / p95** | Wall-clock time per question, as seen by the harness. |

**Fact matching** is normalised so that wording differences do not count as errors:
case and spacing are ignored, `%` = `percent` = `per cent`, number words become digits
(`ten` = `10`, `ninety-nine` = `99`, `one hundred` = `100`), `1,00,000` = `100000`, and currency
markers are dropped (`Rs. 20 crore` = `20 crore rupees`). Matches must be whole tokens, so
`5 per cent` does not match inside `25 per cent` or `2.5 per cent`, and `1 crore` does not match
`1.5 crore`. Dates are not normalised: split a date fact into its month and year.

**"Not in force"** uses the passage's validity window `[valid_from, valid_to)`: a passage is
stale on date D if `valid_from > D` (not yet in force) or `valid_to <= D` (already replaced).
When `valid_from` is missing, the document's issue date is used, because a document cannot
apply before it was issued. When nothing is known, the passage counts as in force. The
baseline is scored with the same validity data, so stale-source rates are only meaningful if
the pipeline returns validity dates on baseline hits too.

**Limits.** Substring facts can miss a correct answer that is phrased differently, and can
accept an answer that states the right number for the wrong reason. Recall measures the
documents available, so a question whose gold document is not in the corpus fails for both
systems; check coverage before reading too much into a number. With a few dozen questions, a
gap of a few points is noise.

## Running

The commands below assume the database is loaded and the API keys are set, as for
`asof ask`. Free-tier quotas are small, so use `--limit` and `--sleep` while trying things out.

```bash
# Both systems, all questions
python -m asof.evaluation.run --systems asof,baseline

# A quick check: first 5 questions, 3 s between calls
python -m asof.evaluation.run --systems asof --limit 5 --sleep 3

# Specific questions
python -m asof.evaluation.run --systems asof,baseline --ids pit-36-4-2020-02,trap-einvoice-2022-09

# Which questions' gold documents are missing from the corpus? (no model calls)
python -m asof.evaluation.run --check-corpus data/parsed

# Continue an interrupted run (skips ids already in that file and appends to it)
python -m asof.evaluation.run --systems asof --resume eval/results/20260101-120000-asof.jsonl
```

A question that raises an error is recorded with the error message and the run continues.
After 3 rate-limit errors in a row the run stops, writes the summary marked
**Incomplete run**, and exits with code 1. Resume it later. Records are written as each
question finishes, so an interrupted run loses nothing.

## Drafting more questions

```bash
python -m asof.evaluation.generate --parsed-dir data/parsed --n 40 --out eval/candidates.jsonl
python -m asof.evaluation.generate --n 40 --dry-run      # only list the documents it would use
python -m asof.evaluation.generate --n 5 --max-calls 5   # cap live LLM calls
```

The generator picks parsed documents: notifications that amend the Rules first, then
circulars, spread across years. It asks the LLM for one factual question per document, then
checks the result in code:

- every `must_contain` fact must appear in the document text (after the same normalisation),
  or the candidate is dropped. The model cannot add facts that are not in the source.
- a question that names a notification or circular number is dropped, because it would make
  retrieval trivial.
- `as_of` is the effective date only if that date is actually written in the document,
  otherwise the issue date. Questions about current law get `as_of: null`.
- the notes record whether the model's evidence sentence was found in the text.

LLM replies are cached in `data/cache/evalgen/`, so a rerun costs nothing. The generator stops
at the first rate-limit error; the cache keeps its progress.

**Candidates are drafts.** A fact can be in the document and still not be the answer
(e.g. "eight" and "digits" for a question about which goods need 8-digit HSN codes). Read each
candidate, fix the question and facts, and only then copy it into `questions.jsonl`.

## Current question set

`questions.jsonl` has 22 hand-written seed questions about rule 36(4), rule 86B, e-invoicing
thresholds, QRMP and the composition limit. They come from published CBIC notifications.
Each note says why the answer is right and what to double-check. Some of their gold documents
(e.g. 49/2019, 75/2019 and 10/2023-Central Tax) were not in the parsed corpus when the questions
were written. Ingest those documents, or drop or replace those questions, before publishing
results.
