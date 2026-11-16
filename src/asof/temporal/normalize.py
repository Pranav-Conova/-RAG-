"""Normalise references to notifications, circulars and provisions into canonical forms.

Pure functions, no I/O:

- notifications: "No. 49/2019 – Central Tax", "49/2019-CT"  -> "49/2019-Central Tax"
- circulars:     "Circular No. 123/42/2019 - GST"           -> "123/42/2019-GST"
- provisions:    "sub-rule (4) of rule 36"                  -> ("cgst_rules", "rule 36(4)")
                 "section 16(4) of the CGST Act"            -> ("cgst_act", "section 16(4)")
"""

from __future__ import annotations

import re
import unicodedata

Instrument = str  # 'cgst_act' | 'cgst_rules' | 'igst_act' | 'other'

_DASH = r"[-‐-―−_]"

# Order matters: "(Rate)" variants must win over the bare series name.
_SERIES: list[tuple[str, str]] = [
    (r"central\s*tax\s*\(\s*rate\s*\)|c\.?\s*t\.?\s*\(\s*r(?:ate)?\.?\s*\)|ctr",
     "Central Tax (Rate)"),
    (r"central\s*tax|c\.?\s*t\.?(?![a-z])|cgst", "Central Tax"),
    (r"integrated\s*tax\s*\(\s*rate\s*\)|i\.?\s*t\.?\s*\(\s*r(?:ate)?\.?\s*\)|itr",
     "Integrated Tax (Rate)"),
    (r"integrated\s*tax|igst|i\.?\s*t\.?(?![a-z])", "Integrated Tax"),
    (r"union\s*territory\s*tax\s*\(\s*rate\s*\)|utt\s*\(\s*r(?:ate)?\s*\)",
     "Union Territory Tax (Rate)"),
    (r"union\s*territory\s*tax|utgst|utt", "Union Territory Tax"),
    (r"compensation\s*cess\s*\(\s*rate\s*\)|cc\s*\(\s*r(?:ate)?\s*\)", "Compensation Cess (Rate)"),
    (r"compensation\s*cess|cc(?![a-z])", "Compensation Cess"),
]
_SERIES_RE = "|".join(f"(?:{p})" for p, _ in _SERIES)

_NOTIFICATION_RE = re.compile(
    rf"(?<![\d/])(\d{{1,3}})\s*/\s*(\d{{4}}|\d{{2}})(?![\d/])\s*(?:{_DASH}+|,)?\s*\(?\s*"
    rf"({_SERIES_RE})",
    re.IGNORECASE,
)
_CIRCULAR_RE = re.compile(
    rf"(?<![\d/])(\d{{1,4}})\s*/\s*(\d{{1,4}})\s*/\s*(\d{{4}})\s*(?:{_DASH}+)?\s*"
    r"(GST|IGST|UTGST|CGST|Compensation\s*Cess)\b",
    re.IGNORECASE,
)


def clean(s: str) -> str:
    """Unicode-normalise, unify quotes/dashes, collapse whitespace."""
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("“", '"').replace("”", '"').replace("‘", "'")
    s = s.replace("’", "'")
    s = re.sub(r"[‐-―−]", "-", s)
    return re.sub(r"\s+", " ", s).strip()


def _year(y: str) -> str:
    return y if len(y) == 4 else f"20{y}"


def _series(raw: str) -> str:
    for pattern, canonical in _SERIES:
        if re.fullmatch(pattern, raw.strip(), re.IGNORECASE):
            return canonical
    return raw.strip()


def normalize_notification_ref(s: str) -> str | None:
    """Canonical notification number, e.g. '49/2019-Central Tax'; None if not found."""
    m = _NOTIFICATION_RE.search(clean(s))
    if not m:
        return None
    num, year, series = m.groups()
    return f"{int(num)}/{_year(year)}-{_series(series)}"


def find_notification_refs(text: str) -> list[str]:
    out: list[str] = []
    for m in _NOTIFICATION_RE.finditer(clean(text)):
        ref = f"{int(m[1])}/{_year(m[2])}-{_series(m[3])}"
        if ref not in out:
            out.append(ref)
    return out


def normalize_circular_ref(s: str) -> str | None:
    """Canonical circular number, e.g. '123/42/2019-GST'; None if not found."""
    m = _CIRCULAR_RE.search(clean(s))
    if not m:
        return None
    a, b, year, suffix = m.groups()
    suffix = re.sub(r"\s+", " ", suffix).upper()
    if suffix.startswith("COMPENSATION"):
        suffix = "Compensation Cess"
    return f"{int(a)}/{int(b)}/{year}-{suffix}"


# ------------------------------------------------------------------ provisions

_SUBUNIT = r"(?:sub-?\s*rule|sub-?\s*section|sub-?\s*clause|clause|item)"
_PAREN = r"\(\s*([0-9]+[A-Za-z]{0,2}|[a-zA-Z]{1,5})\s*\)"
_MAIN_RE = re.compile(
    r"(?<![\w-])(rule|section)\s*(\d{1,3}[A-Za-z]{0,2})(?![\w])((?:\s*\(\s*(?:[0-9]+[A-Za-z]{0,2}|"
    r"[a-zA-Z]{1,5})\s*\))*)",
    re.IGNORECASE,
)
# "clause (c) of sub-rule (4) of " immediately before "rule 89"
_PRE_RE = re.compile(
    rf"((?:{_SUBUNIT}\s*{_PAREN}\s*,?\s*(?:of|in)\s+(?:the\s+)?)+)$", re.IGNORECASE
)
_PRE_ITEM_RE = re.compile(rf"{_SUBUNIT}\s*{_PAREN}", re.IGNORECASE)
# ", in sub-rule (4)" / ", in sub-rule (4), in clause (a)" right after "rule 36"
_POST_RE = re.compile(
    rf"^((?:\s*,?\s*(?:in|of)?\s*(?:the\s+)?{_SUBUNIT}\s*{_PAREN})+)", re.IGNORECASE
)
_PAREN_RE = re.compile(_PAREN)

_IGST = re.compile(r"\bigst\b|integrated\s+goods", re.IGNORECASE)
_UTGST = re.compile(r"\butgst\b|union\s+territory", re.IGNORECASE)
_OTHER_ACT = re.compile(r"customs|finance\s+act|compensation", re.IGNORECASE)


def _part(p: str) -> str:
    p = p.strip()
    if p[0].isdigit():
        digits = re.match(r"\d+", p)[0]  # type: ignore[index]
        return digits + p[len(digits):].upper()
    return p.lower()


def _canon_main(num: str) -> str:
    digits = re.match(r"\d+", num)[0]  # type: ignore[index]
    return str(int(digits)) + num[len(digits):].upper()


def _instrument(unit: str, context: str) -> Instrument:
    if _IGST.search(context):
        return "igst_act" if unit == "section" else "other"
    if _UTGST.search(context) or _OTHER_ACT.search(context):
        return "other"
    return "cgst_act" if unit == "section" else "cgst_rules"


def _parse_main(text: str, m: re.Match[str], max_depth: int | None) -> str:
    unit, num, trailing = m.group(1).lower(), m.group(2), m.group(3) or ""
    parts = [_part(p) for p in _PAREN_RE.findall(trailing)]
    if not parts:
        pre = _PRE_RE.search(text[max(0, m.start() - 160): m.start()])
        if pre:
            parts = [_part(p) for p in reversed(_PRE_ITEM_RE.findall(pre.group(1)))]
        else:
            post = _POST_RE.match(text[m.end(): m.end() + 120])
            if post:
                parts = [_part(p) for p in _PRE_ITEM_RE.findall(post.group(1))]
    if max_depth is not None:
        parts = parts[:max_depth]
    return f"{unit} {_canon_main(num)}" + "".join(f"({p})" for p in parts)


def normalize_provision_ref(
    s: str, *, max_depth: int | None = None
) -> tuple[Instrument, str] | None:
    """('cgst_rules', 'rule 36(4)') for 'sub-rule (4) of rule 36'; None if no rule/section."""
    text = clean(s)
    m = _MAIN_RE.search(text)
    if not m:
        return None
    ref = _parse_main(text, m, max_depth)
    return _instrument(m.group(1).lower(), text), ref


def find_provision_refs(text: str, *, max_depth: int | None = None) -> list[str]:
    """All canonical provision refs ('rule 36(4)', 'section 16') mentioned in `text`."""
    text = clean(text)
    out: list[str] = []
    for m in _MAIN_RE.finditer(text):
        ref = _parse_main(text, m, max_depth)
        if ref not in out:
            out.append(ref)
    return out


def related_refs(a: str, b: str) -> bool:
    """True when a == b or one is a structural ancestor of the other ('rule 36' ~ 'rule 36(4)')."""
    return a == b or a.startswith(b + "(") or b.startswith(a + "(")


def is_ancestor(ancestor: str, ref: str) -> bool:
    return ref.startswith(ancestor + "(")


# ------------------------------------------------------------------ dispatch

TargetKind = str  # notification | circular | rule | section | form | other


def normalize_target(target_kind: str, target_ref: str) -> tuple[TargetKind, str]:
    """Canonicalise an extracted (target_kind, target_ref), fixing the kind when obvious."""
    ref = clean(target_ref)
    circ = normalize_circular_ref(ref)
    if circ and (target_kind in ("circular", "other") or "circular" in ref.lower()):
        return "circular", circ
    notif = normalize_notification_ref(ref)
    if notif and target_kind in ("notification", "circular", "other"):
        return "notification", notif
    if circ:
        return "circular", circ
    if target_kind in ("rule", "section", "other"):
        prov = normalize_provision_ref(ref)
        if prov:
            instrument, pref = prov
            unit = pref.split(" ", 1)[0]
            if instrument == "igst_act":
                return unit, f"{pref} of the IGST Act"
            if instrument == "other":
                return unit, ref  # keep the act's name so provision_key stays 'other'
            return unit, pref
    if notif:
        return "notification", notif
    if target_kind == "form":
        return "form", re.sub(r"^form\s+", "FORM ", ref, flags=re.IGNORECASE)
    return target_kind, ref


def provision_key(target_kind: str, target_ref: str) -> tuple[Instrument, str] | None:
    """(instrument, ref) for a rule/section target, else None."""
    if target_kind not in ("rule", "section"):
        return None
    return normalize_provision_ref(target_ref)
