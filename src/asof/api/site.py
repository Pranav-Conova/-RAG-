"""Public landing page, robots.txt, sitemap.xml and llms.txt for the deployed app.

Search engines (SEO), answer engines (AEO) and generative engines (GEO) read the same things:
a clear title and description, a one-sentence definition, direct question/answer pairs, and
schema.org JSON-LD. Content lives here once and is rendered into each format.
"""

import json
from html import escape

from asof import __version__

NAME = "AsOf"
TAGLINE = "Time-aware RAG for Indian GST law"
REPO_URL = "https://github.com/Pranav-Conova/-RAG-"
STATIC_PREFIX = "/static/"  # main.py mounts src/asof/api/static here
DESCRIPTION = (
    "AsOf is an open-source, time-aware retrieval-augmented generation (RAG) system for Indian "
    "GST law. It answers \"what was the rule on date X?\" by tracking when each notification, "
    "circular and provision was in force, following amendment and supersession chains, and "
    "citing the exact source text."
)
KEYWORDS = [
    "time-aware RAG", "temporal RAG", "point-in-time retrieval", "legal RAG", "GST",
    "Indian GST law", "CGST Rules", "CBIC notifications", "regulatory compliance AI",
    "retrieval-augmented generation", "pgvector", "hybrid search", "reranking", "FastAPI",
]

FAQ: list[tuple[str, str]] = [
    (
        "What is AsOf?",
        "AsOf is an open-source, time-aware retrieval-augmented generation (RAG) system for "
        "Indian GST law. It answers questions as of a specific date, using only the "
        "notifications, circulars and provisions that were in force on that date.",
    ),
    (
        "What is time-aware (temporal) RAG?",
        "Time-aware RAG adds a date dimension to retrieval. Every piece of text stores the "
        "period it was valid for, and a question is answered only from text that was in force "
        "on the requested date, instead of whichever text sounds most similar.",
    ),
    (
        "How is AsOf different from a standard RAG chatbot?",
        "A standard RAG chatbot retrieves the most semantically similar passage, which is often "
        "an amended or superseded version of a rule. AsOf filters by validity dates, follows "
        "amendment chains, and verifies that every cited quote appears in its source.",
    ),
    (
        "Which documents does AsOf cover?",
        "The planned corpus is the CGST Act 2017, the CGST Rules 2017, CBIC Central Tax "
        "notifications and CBIC GST circulars from 2017 onwards.",
    ),
    (
        "Is AsOf free to run?",
        "Yes. It uses free-tier hosted models (NVIDIA API catalog for the LLM and embeddings, "
        "OpenRouter for reranking), a free Neon Postgres database and Render's free web "
        "service. Free tiers are rate limited and meant for development and demos.",
    ),
    (
        "Is AsOf legal or tax advice?",
        "No. AsOf is a research and portfolio project. Always confirm answers against the "
        "official CBIC source documents or with a qualified professional.",
    ),
    (
        "What is the current status of AsOf?",
        "The web app answers questions as of a date and can compare AsOf with a baseline "
        "RAG pipeline side by side. An evaluation harness scores both on a hand-written "
        "question set (answer correctness, retrieval recall, citation precision, stale-source "
        "rate and abstention); results are published in the repository as the corpus grows.",
    ),
]


def _json_ld(base_url: str) -> str:
    graph = [
        {
            "@type": "SoftwareSourceCode",
            "name": NAME,
            "alternateName": TAGLINE,
            "description": DESCRIPTION,
            "codeRepository": REPO_URL,
            "programmingLanguage": "Python",
            "runtimePlatform": "Python 3.11+",
            "version": __version__,
            "keywords": ", ".join(KEYWORDS),
            "url": base_url,
            "author": {"@type": "Person", "name": "Pranav-Conova",
                       "url": "https://github.com/Pranav-Conova"},
        },
        {
            "@type": "FAQPage",
            "mainEntity": [
                {"@type": "Question", "name": q,
                 "acceptedAnswer": {"@type": "Answer", "text": a}}
                for q, a in FAQ
            ],
        },
    ]
    # Escape "<" so text can never close the <script> tag.
    return json.dumps({"@context": "https://schema.org", "@graph": graph}).replace("<", "\\u003c")


# Example questions shown as chips. They are prompts to try, not statements of law.
EXAMPLES: list[tuple[str, str]] = [
    ("Rule 36(4) limit in Feb 2020",
     "What was the provisional ITC limit under rule 36(4) in February 2020?"),
    ("Is rule 36(4) still in force?",
     "Is the rule 36(4) restriction still in force today?"),
    ("E-invoicing above 20 crore",
     "When did e-invoicing become mandatory for turnover above 20 crore?"),
    ("E-invoicing threshold, Sep 2022",
     "What was the e-invoicing turnover threshold on 15 September 2022?"),
]

HOW_IT_WORKS: list[tuple[str, str]] = [
    ("Pick a date",
     "Ask about any date since GST began in July 2017, or leave the date empty for today. "
     "AsOf also reads dates written in the question."),
    ("Retrieve only text in force",
     "Every passage carries the period it was valid for. Retrieval keeps only passages in force "
     "on that date, following amendments and supersessions."),
    ("Answer with checked citations",
     "The answer cites its sources as [S1], [S2]. Each quoted passage is checked against the "
     "source text, and AsOf declines to answer when the sources do not support one."),
]


def _ask_panel() -> str:
    chips = "\n      ".join(
        f'<button type="button" class="chip" data-q="{escape(q)}" title="{escape(q)}">'
        f"{escape(label)}</button>"
        for label, q in EXAMPLES
    )
    return f"""<section id="ask" class="card ask" aria-labelledby="ask-title">
    <h2 id="ask-title">Ask about GST law on any date</h2>
    <form id="ask-form" data-endpoint="/api/ask" novalidate>
      <label for="q" class="label">Question</label>
      <textarea id="q" name="question" rows="3" maxlength="1000" required
        aria-describedby="q-hint"
        placeholder="e.g. What was the provisional ITC limit under rule 36(4) in February 2020?"
      ></textarea>
      <p id="q-hint" class="hint">Press Ctrl+Enter to ask. Answers cite CBIC source documents.</p>
      <div class="controls">
        <div class="field">
          <label for="as-of" class="label">As of date</label>
          <input type="date" id="as-of" name="as_of" min="2017-07-01" aria-describedby="as-of-hint">
          <span id="as-of-hint" class="hint">Empty = today, or a date in the question</span>
        </div>
        <div class="field switch-field">
          <input type="checkbox" id="compare" name="compare" role="switch">
          <label for="compare"><span class="label">Compare with baseline RAG</span>
            <span class="hint">Runs both side by side; the baseline ignores dates</span></label>
        </div>
        <button type="submit" id="ask-btn" class="btn btn-primary">Ask</button>
      </div>
      <div class="examples" role="group" aria-labelledby="examples-label">
        <span id="examples-label" class="hint">Example questions:</span>
      {chips}
      </div>
    </form>
    <noscript><p class="hint">The question form needs JavaScript. The same API is available as
      <code>POST /api/ask</code>; see <a href="docs">/docs</a>.</p></noscript>
    <p id="status" class="status" role="status" aria-live="polite"></p>
    <div id="results" class="results"></div>
  </section>"""


def landing_html(base_url: str) -> str:
    title = f"{NAME}: {TAGLINE}"
    faq_html = "\n  ".join(
        f"<details><summary>{escape(q)}</summary><p>{escape(a)}</p></details>" for q, a in FAQ
    )
    steps = "\n      ".join(
        f"<li><h3>{escape(t)}</h3><p>{escape(d)}</p></li>" for t, d in HOW_IT_WORKS
    )
    v = escape(__version__)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(title)}</title>
<meta name="description" content="{escape(DESCRIPTION)}">
<meta name="keywords" content="{escape(", ".join(KEYWORDS))}">
<link rel="canonical" href="{escape(base_url)}">
<meta property="og:type" content="website">
<meta property="og:title" content="{escape(title)}">
<meta property="og:description" content="{escape(DESCRIPTION)}">
<meta property="og:url" content="{escape(base_url)}">
<meta name="twitter:card" content="summary">
<meta name="color-scheme" content="light dark">
<meta name="theme-color" content="#f6f7f9" media="(prefers-color-scheme: light)">
<meta name="theme-color" content="#0d1117" media="(prefers-color-scheme: dark)">
<link rel="alternate" type="text/plain" href="{escape(base_url)}llms.txt" title="llms.txt">
<script type="application/ld+json">{_json_ld(base_url)}</script>
<style>
  /* Readable fallback if the stylesheet fails to load; app.css overrides this. */
  :root {{ color-scheme: light dark; }}
  body {{ font: 16px/1.6 system-ui, sans-serif; max-width: 60rem; margin: 0 auto;
         padding: 0 1rem; }}
</style>
<link rel="stylesheet" href="{STATIC_PREFIX}app.css?v={v}">
<script src="{STATIC_PREFIX}app.js?v={v}" defer></script>
</head>
<body>
<a class="skip-link" href="#q">Skip to the question box</a>
<header class="site-header">
  <div class="wrap nav">
    <a class="brand" href="#top"><span class="logo" aria-hidden="true"></span>{escape(NAME)}</a>
    <nav aria-label="Page sections">
      <a href="#how">How it works</a>
      <a href="#faq">FAQ</a>
      <a href="docs">API</a>
      <a href="{REPO_URL}">GitHub</a>
    </nav>
  </div>
</header>
<main id="top" class="wrap">
  <section class="hero">
    <p class="eyebrow">Open source &middot; point-in-time retrieval &middot; checked citations</p>
    <h1>{escape(NAME)}: {escape(TAGLINE)}</h1>
    <p class="lede">{escape(DESCRIPTION)}</p>
  </section>
  {_ask_panel()}
  <section id="how" aria-labelledby="how-title">
    <h2 id="how-title">How it works</h2>
    <ol class="steps">
      {steps}
    </ol>
    <p>Turn on <strong>Compare</strong> to run the same question through a baseline RAG
      pipeline that ignores dates. Source cards are marked <em>in force</em>,
      <em>not yet in force</em> or <em>superseded</em> for the date asked about, so you can see
      when an answer leans on outdated law.</p>
  </section>
  <section aria-labelledby="facts-title">
    <h2 id="facts-title">Key facts</h2>
    <ul class="facts">
      <li><strong>Type:</strong> open-source time-aware RAG (retrieval-augmented generation)</li>
      <li><strong>Domain:</strong> Indian GST law (CGST Act, CGST Rules, CBIC notifications
        and circulars)</li>
      <li><strong>Core idea:</strong> every text chunk has a validity date range; questions are
        answered as of a date</li>
      <li><strong>Stack:</strong> Python, FastAPI, PostgreSQL + pgvector, NVIDIA Nemotron models,
        OpenRouter reranking</li>
      <li><strong>Source code:</strong> <a href="{REPO_URL}">{REPO_URL}</a></li>
    </ul>
  </section>
  <section id="faq" aria-labelledby="faq-title">
    <h2 id="faq-title">Frequently asked questions</h2>
  {faq_html}
  </section>
  <section aria-labelledby="api-title">
    <h2 id="api-title">API</h2>
    <p>Interactive API documentation: <a href="docs">/docs</a> &middot; health:
      <a href="health">/health</a> &middot; ask: <code>POST /api/ask</code> with
      <code>{{"question": "...", "as_of": "YYYY-MM-DD", "mode": "asof"}}</code></p>
  </section>
</main>
<footer class="site-footer"><div class="wrap"><p>Not legal or tax advice: always check the
  official CBIC source documents. Version {v}.</p></div></footer>
</body>
</html>"""


def robots_txt(base_url: str) -> str:
    # Explicitly welcome search and AI crawlers; there is nothing private on this site.
    agents = ["*", "Googlebot", "Bingbot", "GPTBot", "OAI-SearchBot", "ChatGPT-User",
              "ClaudeBot", "Claude-SearchBot", "PerplexityBot", "Google-Extended", "CCBot"]
    rules = "\n\n".join(f"User-agent: {a}\nAllow: /" for a in agents)
    return f"{rules}\n\nSitemap: {base_url}sitemap.xml\n"


def sitemap_xml(base_url: str) -> str:
    urls = ["", "llms.txt", "docs"]
    entries = "\n".join(f"  <url><loc>{escape(base_url + u)}</loc></url>" for u in urls)
    return ('<?xml version="1.0" encoding="UTF-8"?>\n'
            '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
            f"{entries}\n</urlset>\n")


def llms_txt(base_url: str) -> str:
    faq = "\n".join(f"- **{q}** {a}" for q, a in FAQ)
    return f"""# {NAME}

> {DESCRIPTION}

{NAME} is a portfolio and research project, not legal or tax advice.

## Links

- [Source code]({REPO_URL}): Python package, schema, tests
- [README]({REPO_URL}#readme): overview, glossary, FAQ
- [Architecture and roadmap]({REPO_URL}/blob/main/PLAN.md)
- [API documentation]({base_url}docs): OpenAPI / Swagger UI

## FAQ

{faq}
"""
