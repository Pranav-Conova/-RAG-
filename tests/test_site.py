"""Landing page / web UI shell: structure, accessibility hooks and static asset safety."""

import re
from html.parser import HTMLParser
from pathlib import Path

from asof.api import site

STATIC = Path(site.__file__).parent / "static"
HTML = site.landing_html("https://asof.example/")


class _Collector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: list[str] = []
        self.label_for: set[str] = set()
        self.tags: list[tuple[str, dict[str, str | None]]] = []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        self.tags.append((tag, a))
        if a.get("id"):
            self.ids.append(a["id"])  # type: ignore[arg-type]
        if tag == "label" and a.get("for"):
            self.label_for.add(a["for"])  # type: ignore[arg-type]


def parsed() -> _Collector:
    c = _Collector()
    c.feed(HTML)
    return c


def test_page_references_static_assets_that_exist():
    assert re.search(r'<link rel="stylesheet" href="/static/app\.css\?v=[^"]+">', HTML)
    assert re.search(r'<script src="/static/app\.js\?v=[^"]+" defer></script>', HTML)
    assert (STATIC / "app.css").is_file() and (STATIC / "app.js").is_file()


def test_ask_form_controls_are_labelled_and_ids_unique():
    c = parsed()
    assert len(c.ids) == len(set(c.ids)), "duplicate element ids"
    for control in ("q", "as-of", "compare"):
        assert control in c.ids and control in c.label_for
    form = next(a for t, a in c.tags if t == "form")
    assert form["data-endpoint"] == "/api/ask"
    assert any(t == "p" and a.get("role") == "status" and a.get("aria-live") == "polite"
               for t, a in c.tags)
    assert any(t == "input" and a.get("role") == "switch" for t, a in c.tags)


def test_example_chips_are_buttons_with_escaped_questions():
    chips = [a for t, a in parsed().tags if t == "button" and "chip" in (a.get("class") or "")]
    assert len(chips) == len(site.EXAMPLES) >= 3
    assert all(a.get("type") == "button" and a.get("data-q") for a in chips)
    assert {a["data-q"] for a in chips} == {q for _, q in site.EXAMPLES}


def test_seo_content_is_kept():
    assert "<h1>AsOf: Time-aware RAG for Indian GST law</h1>" in HTML
    assert HTML.count("<details><summary>") == len(site.FAQ)
    assert '<script type="application/ld+json">' in HTML
    assert 'href="docs"' in HTML and 'href="health"' in HTML
    # The ask panel comes before the FAQ.
    assert HTML.index('id="ask"') < HTML.index('id="faq"')


def test_js_never_uses_html_injection_sinks():
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    for sink in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval("):
        assert sink not in js, sink


def test_css_supports_dark_mode_and_reduced_motion():
    css = (STATIC / "app.css").read_text(encoding="utf-8")
    assert "@media (prefers-color-scheme: dark)" in css
    assert "@media (prefers-reduced-motion: reduce)" in css
    assert ":focus-visible" in css
