import json
import re

from fastapi.testclient import TestClient

from asof.api.main import app

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_landing_page_has_seo_metadata_and_valid_json_ld():
    resp = client.get("/")
    assert resp.status_code == 200
    html = resp.text
    assert "<title>AsOf: Time-aware RAG for Indian GST law</title>" in html
    assert '<meta name="description"' in html
    assert '<link rel="canonical" href="http://testserver/">' in html

    blob = re.search(r'<script type="application/ld\+json">(.*?)</script>', html, re.S).group(1)
    data = json.loads(blob)
    types = {node["@type"] for node in data["@graph"]}
    assert types == {"SoftwareSourceCode", "FAQPage"}
    faq = next(n for n in data["@graph"] if n["@type"] == "FAQPage")
    assert all(q["acceptedAnswer"]["text"] for q in faq["mainEntity"])


def test_crawler_files():
    robots = client.get("/robots.txt").text
    assert "User-agent: GPTBot\nAllow: /" in robots
    assert "Sitemap: http://testserver/sitemap.xml" in robots

    sitemap = client.get("/sitemap.xml")
    assert sitemap.headers["content-type"].startswith("application/xml")
    assert "<loc>http://testserver/</loc>" in sitemap.text

    llms = client.get("/llms.txt").text
    assert llms.startswith("# AsOf\n\n> ")
    assert "## FAQ" in llms


def test_site_routes_hidden_from_openapi():
    paths = client.get("/openapi.json").json()["paths"]
    assert "/health" in paths
    assert "/" not in paths and "/robots.txt" not in paths
