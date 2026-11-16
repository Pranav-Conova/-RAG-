from asof.temporal.refs import chunk_refs


def test_chunk_refs_finds_every_rule_mentioned():
    text = ('In the said rules, in rule 9, in sub-rule (2), for the words "seven working days", '
            'the words "twenty one days" shall be substituted; in rule 36, in sub-rule (4), ...')
    assert chunk_refs(text, "rule 8") == ["rule 36(4)", "rule 8", "rule 9(2)"]


def test_chunk_refs_empty_text():
    assert chunk_refs("", None) == []
