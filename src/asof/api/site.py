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
        "Phase 0 is complete: database schema, model providers, CLI and API scaffold. Document "
        "ingestion, temporal retrieval and the evaluation set are in progress.",
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


def landing_html(base_url: str) -> str:
    title = f"{NAME}: {TAGLINE}"
    faq_html = "\n".join(
        f"<details><summary>{escape(q)}</summary><p>{escape(a)}</p></details>" for q, a in FAQ
    )
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
<link rel="alternate" type="text/plain" href="{escape(base_url)}llms.txt" title="llms.txt">
<script type="application/ld+json">{_json_ld(base_url)}</script>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font: 16px/1.6 system-ui, sans-serif; max-width: 46rem; margin: 2rem auto;
         padding: 0 1rem; }}
  summary {{ font-weight: 600; cursor: pointer; margin-top: .75rem; }}
  code {{ font-size: .95em; }}
</style>
</head>
<body>
<header>
  <h1>{escape(NAME)}: {escape(TAGLINE)}</h1>
  <p>{escape(DESCRIPTION)}</p>
</header>
<main>
  <h2>Key facts</h2>
  <ul>
    <li><strong>Type:</strong> open-source time-aware RAG (retrieval-augmented generation)</li>
    <li><strong>Domain:</strong> Indian GST law (CGST Act, CGST Rules, CBIC notifications
      and circulars)</li>
    <li><strong>Core idea:</strong> every text chunk has a validity date range; questions are
      answered as of a date</li>
    <li><strong>Stack:</strong> Python, FastAPI, PostgreSQL + pgvector, NVIDIA Nemotron models,
      OpenRouter reranking</li>
    <li><strong>Source code:</strong> <a href="{REPO_URL}">{REPO_URL}</a></li>
  </ul>
  <h2>Frequently asked questions</h2>
  {faq_html}
  <h2>API</h2>
  <p>Interactive API documentation: <a href="docs">/docs</a> &middot; health:
    <a href="health">/health</a></p>
</main>
<footer><p>Not legal or tax advice. Version {escape(__version__)}.</p></footer>
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
