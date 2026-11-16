import pytest

from asof.temporal.normalize import (
    find_notification_refs,
    find_provision_refs,
    is_ancestor,
    normalize_circular_ref,
    normalize_notification_ref,
    normalize_provision_ref,
    normalize_target,
    provision_key,
    related_refs,
)


@pytest.mark.parametrize("raw,expected", [
    ("No. 49/2019 – Central Tax", "49/2019-Central Tax"),
    ("49/2019-CT", "49/2019-Central Tax"),
    ("notification number 49/2019- Central Tax", "49/2019-Central Tax"),
    ("Notification No. 49/2019-Central Tax, dated the 9th October, 2019", "49/2019-Central Tax"),
    ("notification No. 01/2017-Central Tax", "1/2017-Central Tax"),
    ("No. 2/2017-Central Tax (Rate)", "2/2017-Central Tax (Rate)"),
    ("11/2017 – Central Tax(Rate)", "11/2017-Central Tax (Rate)"),
    ("12/17-CT(R)", "12/2017-Central Tax (Rate)"),
    ("4/2018-Integrated Tax", "4/2018-Integrated Tax"),
    ("1/2017-Integrated Tax (Rate)", "1/2017-Integrated Tax (Rate)"),
    ("3/2019-Union Territory Tax", "3/2019-Union Territory Tax"),
    ("1/2020-Compensation Cess (Rate)", "1/2020-Compensation Cess (Rate)"),
])
def test_notification_refs(raw, expected):
    assert normalize_notification_ref(raw) == expected


def test_notification_ref_absent():
    assert normalize_notification_ref("rule 36(4) of the CGST Rules") is None
    # a circular number is not a notification number
    assert normalize_notification_ref("Circular No. 123/42/2019-GST") is None


def test_find_notification_refs():
    text = ("in supersession of notification No. 13/2017-Central Tax and 14/2017 - CT, "
            "except as respects things done; see also 13/2017-Central Tax")
    assert find_notification_refs(text) == ["13/2017-Central Tax", "14/2017-Central Tax"]


@pytest.mark.parametrize("raw,expected", [
    ("Circular No. 123/42/2019 - GST", "123/42/2019-GST"),
    ("circular no.183/15/2022-GST dated 27.12.2022", "183/15/2022-GST"),
    ("Circular No. 01/01/2017-IGST", "1/1/2017-IGST"),
    ("CBIC Circular 170/02/2022 –GST", "170/2/2022-GST"),
])
def test_circular_refs(raw, expected):
    assert normalize_circular_ref(raw) == expected


@pytest.mark.parametrize("raw,expected", [
    ("rule 36(4)", ("cgst_rules", "rule 36(4)")),
    ("Rule 36 (4)", ("cgst_rules", "rule 36(4)")),
    ("sub-rule (4) of rule 36", ("cgst_rules", "rule 36(4)")),
    ("in the said rules, in rule 36, in sub-rule (4), for the words", ("cgst_rules", "rule 36(4)")),
    ("in rule 36, sub-rule (4) shall be omitted", ("cgst_rules", "rule 36(4)")),
    ("clause (c) of sub-rule (4) of rule 89", ("cgst_rules", "rule 89(4)(c)")),
    ("rule 86b", ("cgst_rules", "rule 86B")),
    ("section 16(4) of the CGST Act", ("cgst_act", "section 16(4)")),
    ("sub-section (4) of section 16", ("cgst_act", "section 16(4)")),
    ("Section 2 (17)", ("cgst_act", "section 2(17)")),
    ("section 5 of the Integrated Goods and Services Tax Act, 2017", ("igst_act", "section 5")),
    ("section 5 of the IGST Act", ("igst_act", "section 5")),
])
def test_provision_refs(raw, expected):
    assert normalize_provision_ref(raw) == expected


def test_provision_ref_absent_and_depth():
    assert normalize_provision_ref("notification No. 49/2019-Central Tax") is None
    assert normalize_provision_ref("sub-rule (4)") is None  # no parent rule: unknown
    assert normalize_provision_ref("rule 89(4)(c)", max_depth=1) == ("cgst_rules", "rule 89(4)")


def test_find_provision_refs_multiple():
    text = "in rule 36, in sub-rule (4), for the words; in rule 37, sub-rule (1) shall be omitted"
    assert find_provision_refs(text) == ["rule 36(4)", "rule 37(1)"]


def test_related_and_ancestor():
    assert related_refs("rule 36", "rule 36(4)")
    assert related_refs("rule 36(4)", "rule 36(4)")
    assert not related_refs("rule 3", "rule 36(4)")
    assert is_ancestor("rule 36", "rule 36(4)(a)")
    assert not is_ancestor("rule 36(4)", "rule 36")


@pytest.mark.parametrize("kind,ref,expected", [
    ("notification", "Notification No. 49/2019 – Central Tax", ("notification",
                                                                    "49/2019-Central Tax")),
    ("notification", "Circular No. 123/42/2019-GST", ("circular", "123/42/2019-GST")),
    ("circular", "123/42/2019 - GST", ("circular", "123/42/2019-GST")),
    ("rule", "sub-rule (4) of rule 36", ("rule", "rule 36(4)")),
    ("section", "section 5 of the IGST Act", ("section", "section 5 of the IGST Act")),
    ("other", "rule 36(4)", ("rule", "rule 36(4)")),
    ("form", "Form GSTR-3B", ("form", "FORM GSTR-3B")),
])
def test_normalize_target(kind, ref, expected):
    assert normalize_target(kind, ref) == expected


def test_provision_key():
    assert provision_key("rule", "rule 36(4)") == ("cgst_rules", "rule 36(4)")
    assert provision_key(*normalize_target("section", "section 5 of the IGST Act")) == (
        "igst_act", "section 5")
    assert provision_key("notification", "49/2019-Central Tax") is None
