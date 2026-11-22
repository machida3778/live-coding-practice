import re
from datetime import date

# \d は全角数字にもマッチする (Unicode の数字)
DATE_RE = re.compile(
    r"(?<!\d)(?:"
    r"(?P<y1>\d{4})年(?P<m1>\d{1,2})月(?P<d1>\d{1,2})日"
    r"|(?P<y2>\d{4})(?P<sep>[/-])(?P<m2>\d{1,2})(?P=sep)(?P<d2>\d{1,2})(?!\d)"
    r")"
)


def _to_iso(m: re.Match) -> str | None:
    if m.group("y1") is not None:
        y, mo, d = m.group("y1", "m1", "d1")
    else:
        y, mo, d = m.group("y2", "m2", "d2")
    try:
        return date(int(y), int(mo), int(d)).isoformat()  # int("０５") == 5
    except ValueError:
        return None


def extract_dates(text: str) -> list[str]:
    return [iso for m in DATE_RE.finditer(text) if (iso := _to_iso(m)) is not None]


def mask_dates(text: str) -> str:
    return DATE_RE.sub(lambda m: "<DATE>" if _to_iso(m) else m.group(0), text)
