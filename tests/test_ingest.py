"""Ingestion unit tests (no network, no database)."""

from datetime import date
from pathlib import Path

import pytest

from asof.ingest import load as load_mod
from asof.ingest.chunk import chunk_document, detect_section_ref, split_sentences
from asof.ingest.crawl import (
    normalise_circular,
    normalise_notification,
    parse_circulars,
    parse_date,
    parse_notifications,
)
from asof.ingest.parse import clean_page_text, is_mostly_devanagari, strip_running_headers
from asof.models import DocMeta, Page, ParsedDoc

FIX = Path(__file__).parent / "fixtures" / "ingest"
NOTIF_URL = "https://cbic-gst.gov.in/hindi/central-tax-notifications.html"
CIRC_URL = "https://cbic-gst.gov.in/hindi/circulars.html"


# ------------------------------------------------------------------ crawl


@pytest.mark.parametrize(
    "cell, number, issued",
    [
        ("27/2022 - Central Tax dated 26.12.2022", "27/2022-Central Tax", date(2022, 12, 26)),
        ("17/2022- Central Tax, dated 01.08.2022", "17/2022-Central Tax", date(2022, 8, 1)),
        ("05/2022-Central Tax dated 17.05.2022", "5/2022-Central Tax", date(2022, 5, 17)),
        ("64/2020-Central Tax, dated. 31.08.2020", "64/2020-Central Tax", date(2020, 8, 31)),
        ("54/2020-Central Tax, dt. 24-06-2020", "54/2020-Central Tax", date(2020, 6, 24)),
        ("56/2019-Central Tax ,dt. 14-11-2019", "56/2019-Central Tax", date(2019, 11, 14)),
    ],
)
def test_normalise_notification(cell: str, number: str, issued: date) -> None:
    assert normalise_notification(cell) == (number, issued)


def test_normalise_notification_rejects_other_rows() -> None:
    assert normalise_notification("Corrigendum to notification No. 12/2020") is None
    assert normalise_notification("2022") is None


@pytest.mark.parametrize(
    "cell, number",
    [
        ("183/15/2022-GST", "183/15/2022-GST"),
        ("92/11/2019-GST (Central GST)", "92/11/2019-GST"),
        ("90/9/2019-GST", "90/09/2019-GST"),
        ("Order-01/2022-GST", None),
        ("44/18/2018-CGST", None),
    ],
)
def test_normalise_circular(cell: str, number: str | None) -> None:
    assert normalise_circular(cell) == number


def test_parse_date_formats() -> None:
    assert parse_date("dated 26.12.2022") == date(2022, 12, 26)
    assert parse_date("27-12-2022") == date(2022, 12, 27)
    assert parse_date("31.02.2022") is None
    assert parse_date("no date") is None


def test_parse_notifications_fixture() -> None:
    html = (FIX / "notifications.html").read_text(encoding="utf-8")
    res = parse_notifications(html, NOTIF_URL, years=(2019, 2022))
    numbers = [d.number for d in res.docs]
    assert numbers == [
        "27/2022-Central Tax",
        "17/2022-Central Tax",
        "53/2020-Central Tax",
        "64/2020-Central Tax",
    ]
    first = res.docs[0]
    assert first.kind == "notification"
    assert first.issue_date == date(2022, 12, 26)
    assert first.url == "https://cbic-gst.gov.in/pdf/central-tax/ct27-2022.pdf"
    assert first.title == "Notification under sub-rule (4B) of rule 8 of CGST Rules, 2017"
    assert first.key == "notification_27-2022-central-tax"
    # absolute links are kept, whitespace in the subject is collapsed
    assert res.docs[1].url == "https://cbec-gst.gov.in/pdf/central-tax/ct17-2022.pdf"
    assert "  " not in res.docs[1].title
    # the corrigendum link is not taken as the English PDF
    assert res.docs[2].url.endswith("notfctn-53-central-tax-english-2020.pdf")
    assert res.counts["skip_duplicate"] == 1
    assert res.counts["skip_unparsed_number"] == 1
    assert res.counts["skip_out_of_years"] == 1


def test_parse_circulars_fixture() -> None:
    html = (FIX / "circulars.html").read_text(encoding="utf-8")
    res = parse_circulars(html, CIRC_URL, years=(2019, 2022))
    assert [d.number for d in res.docs] == ["183/15/2022-GST", "122/41/2019-GST", "92/11/2019-GST"]
    assert all(d.kind == "circular" for d in res.docs)
    assert res.docs[0].issue_date == date(2022, 12, 27)
    assert res.docs[0].url == "https://cbic-gst.gov.in/pdf/circular-183.pdf"
    # row with a stray nested <td>: date and subject still found
    assert res.docs[1].issue_date == date(2019, 11, 5)
    assert res.docs[1].title.startswith("Generation and quoting of Document Identification")
    assert res.counts["skip_unparsed_number"] == 1  # the -CGST 2018 row; orders tab ignored


# ------------------------------------------------------------------ parse

HINDI = "केंद्रीय माल और सेवा कर नियम, 2017 के नियम 36 में संशोधन"


def test_devanagari_detection() -> None:
    assert is_mostly_devanagari(HINDI)
    assert not is_mostly_devanagari("In rule 36 of the Central Goods and Services Tax Rules")
    assert not is_mostly_devanagari("1234 / 2019")
    # mixed line with mostly English letters stays
    assert not is_mostly_devanagari("Notification No. 49/2019-Central Tax (कर)")


def test_clean_page_text_drops_hindi_lines() -> None:
    raw = (
        f"{HINDI}\n"
        "भारत सरकार\n"
        "GOVERNMENT OF INDIA\n"
        "\n\n\n"
        "Page 1 of 3\n"
        "In  rule 36, in sub-rule (4),� the words\n"
    )
    cleaned = clean_page_text(raw)
    assert cleaned == "GOVERNMENT OF INDIA\n\nIn rule 36, in sub-rule (4), the words"


def test_strip_running_headers() -> None:
    pages = [
        Page(page=1, text="Circular No. 92/11/2019-GST\nSubject: sales promotion\nBody one."),
        Page(page=2, text="Circular No. 92/11/2019-GST\nBody two continues here."),
    ]
    out = strip_running_headers(pages)
    assert out[0].text.startswith("Circular No.")
    assert out[1].text == "Body two continues here."


# ------------------------------------------------------------------ chunk


def _doc(pages: list[str], kind: str = "notification") -> ParsedDoc:
    meta = DocMeta(
        kind=kind,  # type: ignore[arg-type]
        number="49/2019-Central Tax",
        issue_date=date(2019, 10, 9),
        title="Seeks to amend the CGST Rules",
        url="https://example.org/x.pdf",
    )
    return ParsedDoc(meta=meta, pages=[Page(page=i, text=t) for i, t in enumerate(pages, 1)])


def _para(n: int, words: int) -> str:
    body = " ".join(f"word{n}x{i}" for i in range(words - 1))
    return f"{n}. {body} end."


def test_split_sentences_respects_abbreviations() -> None:
    s = "See Notification No. 49/2019-Central Tax dt. 9.10.2019. The limit is Rs. 10 lakh. Done."
    assert split_sentences(s) == [
        "See Notification No. 49/2019-Central Tax dt. 9.10.2019.",
        "The limit is Rs. 10 lakh.",
        "Done.",
    ]


def test_short_document_is_one_chunk() -> None:
    chunks = chunk_document(_doc(["In rule 36, in sub-rule (4), for the words x, y shall be."]))
    assert len(chunks) == 1
    assert chunks[0].ord == 0
    assert chunks[0].page == 1
    assert chunks[0].section_ref == "rule 36(4)"


def test_chunk_sizes_pages_and_order() -> None:
    page1 = "\n".join(_para(n, 120) for n in range(1, 6))  # 600 words
    page2 = "\n".join(_para(n, 120) for n in range(6, 11))  # 600 words
    chunks = chunk_document(_doc([page1, page2], kind="circular"))
    assert [c.ord for c in chunks] == list(range(len(chunks)))
    sizes = [len(c.text.split()) for c in chunks]
    assert all(s <= 450 * 1.3 for s in sizes)
    assert all(s >= 100 for s in sizes)
    assert chunks[0].page == 1 and chunks[-1].page == 2
    # every paragraph appears in some chunk, and none is cut mid-sentence
    full = " ".join(c.text for c in chunks)
    for n in range(1, 11):
        assert f"word{n}x0" in full and f"word{n}x118 end." in full
    for c in chunks:
        assert c.text.rstrip().endswith(".")
    assert chunks[0].section_ref == "para 1"


def test_chunks_overlap_by_a_sentence() -> None:
    sentences = " ".join(f"Sentence {i} has exactly seven words here." for i in range(200))
    chunks = chunk_document(_doc([sentences]))
    assert len(chunks) >= 3
    for a, b in zip(chunks, chunks[1:], strict=False):
        first = b.text.split(". ")[0]
        assert first + "." in a.text  # b starts with a sentence carried over from a
        assert len(first.split()) <= 40


def test_long_unbroken_text_is_hard_split() -> None:
    blob = " ".join(f"w{i}" for i in range(2000))
    chunks = chunk_document(_doc([blob]))
    assert len(chunks) >= 4
    assert all(len(c.text.split()) <= 600 for c in chunks)


@pytest.mark.parametrize(
    "text, kind, ref",
    [
        ("In exercise of powers under section 164 of the Act ... in rule 36, in sub-rule (4), "
         "for the words", "notification", "rule 36(4)"),
        ("the provisions of sub-rule (4A) of rule 8 of the said rules shall not apply",
         "notification", "rule 8(4A)"),
        ("as per section 16(2)(c) of the CGST Act", "notification", "section 16(2)(c)"),
        ("[TO BE PUBLISHED IN THE GAZETTE OF INDIA, EXTRAORDINARY, PART II, SECTION 3, "
         "SUB-SECTION (i)] NOTIFICATION under section 168", "notification", "section 168"),
        ("3. The issue has been examined. As per section 34 ...", "circular", "para 3"),
        ("4.2 It is clarified that ...", "circular", "para 4.2"),
        ("as clarified in para 5 of the earlier circular", "circular", "para 5"),
        ("No references at all.", "notification", None),
    ],
)
def test_detect_section_ref(text: str, kind: str, ref: str | None) -> None:
    assert detect_section_ref(text, kind) == ref


# ------------------------------------------------------------------ load (no DB)


def test_embedding_input_header() -> None:
    doc = _doc(["text"])
    chunk = chunk_document(doc)[0]
    assert load_mod.embedding_input(doc, chunk) == (
        "49/2019-Central Tax dated 09.10.2019 — Seeks to amend the CGST Rules\ntext"
    )


def test_load_documents_batches_across_documents(monkeypatch: pytest.MonkeyPatch) -> None:
    written: list[tuple[str, int]] = []

    def fake_write(conn, doc, vectors):  # noqa: ANN001
        assert len(vectors) == len(doc.chunks)
        written.append((doc.parsed.meta.number, len(vectors)))
        return len(written)

    monkeypatch.setattr(load_mod, "write_document", fake_write)
    calls: list[int] = []

    def fake_embed(texts):  # noqa: ANN001
        calls.append(len(texts))
        return [[0.0] * 4 for _ in texts]

    docs = []
    for i in range(5):
        d = _doc(["x"])
        d.meta.number = f"{i}/2019-Central Tax"
        docs.append(load_mod.PreparedDoc(d, chunk_document(d) * 3))
    docs.append(load_mod.PreparedDoc(_doc([""]), []))  # empty doc: row only, no embedding
    n = load_mod.load_documents(None, docs, fake_embed, batch_chunks=6)  # type: ignore[arg-type]
    assert n == 15
    assert calls == [6, 6, 3]
    assert len(written) == 6


class _Err(Exception):
    def __init__(self, status: int) -> None:
        super().__init__(f"HTTP {status}")
        self.status_code = status


def test_adaptive_embedder_splits_batches_on_5xx() -> None:
    sizes: list[int] = []

    def flaky(texts):  # noqa: ANN001
        sizes.append(len(texts))
        if len(texts) > 2:
            raise _Err(500)
        return [[float(len(t))] for t in texts]

    emb = load_mod.AdaptiveEmbedder(flaky, start_batch=8, sleep=lambda s: None)
    texts = [f"t{i}" for i in range(7)]
    out = emb(texts)
    assert out == [[2.0]] * 7
    assert sizes[:3] == [7, 3, 1]
    assert emb.failures == 2


def test_adaptive_embedder_gives_up() -> None:
    def broken(texts):  # noqa: ANN001
        raise _Err(500)

    emb = load_mod.AdaptiveEmbedder(broken, start_batch=2, max_failures=3, sleep=lambda s: None)
    with pytest.raises(load_mod.EmbeddingUnavailable):
        emb(["a", "b"])

    def forbidden(texts):  # noqa: ANN001
        raise _Err(403)

    with pytest.raises(load_mod.EmbeddingUnavailable):
        load_mod.AdaptiveEmbedder(forbidden, sleep=lambda s: None)(["a"])


def test_adaptive_embedder_skips_rejected_text() -> None:
    def picky(texts):  # noqa: ANN001
        if any("bad" in t for t in texts):
            raise _Err(422)
        return [[1.0] for _ in texts]

    emb = load_mod.AdaptiveEmbedder(picky, start_batch=4, sleep=lambda s: None)
    out = emb(["a", "bad", "c", "d", "e"])
    assert out == [[1.0], None, [1.0], [1.0], [1.0]]
    assert emb.rejected == 1


def test_download_falls_back_to_tax_portal(tmp_path: Path) -> None:
    import base64
    import json as _json

    import httpx

    from asof.ingest.crawl import PoliteClient, TaxPortal, download

    pdf = b"%PDF-1.7 fake"

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url.endswith("authenticate-token"):
            return httpx.Response(200, json={"id_token": "tok"})
        if "fetchNotificationByYearAndCategory" in url:
            assert request.headers["Authorization1"] == "homeToken tok"
            rows = [
                {"id": 7, "notificationNo": "49/2019-Central Tax", "contentLanguage": "ENGLISH"}
            ]
            return httpx.Response(200, content=_json.dumps(rows))
        if url.endswith("/download/7/ENG"):
            return httpx.Response(200, json={"data": base64.b64encode(pdf).decode()})
        return httpx.Response(404)

    client = PoliteClient(min_interval_s=0, retries=0)
    client.client = httpx.Client(transport=httpx.MockTransport(handler))
    meta = DocMeta(
        kind="notification",
        number="49/2019-Central Tax",
        issue_date=date(2019, 10, 9),
        title="t",
        url="https://cbic-gst.gov.in/pdf/central-tax/notfctn-49-central-tax-english-2019.pdf",
    )
    counts = download(client, [meta], tmp_path, fallback=TaxPortal(client))
    assert counts["via_tax_portal"] == 1
    assert (tmp_path / f"{meta.key}.pdf").read_bytes() == pdf
    assert meta.pdf_path
