import re
import unicodedata

URL_RE = re.compile(r"https?://[a-z0-9\-._~:/?#@&=+%]+")
EMAIL_RE = re.compile(r"[a-z0-9._+\-]+@[a-z0-9\-]+(?:\.[a-z0-9\-]+)+")
DIGIT_RE = re.compile(r"[0-9]+")
SPACE_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = URL_RE.sub("<URL>", text)
    text = EMAIL_RE.sub("<EMAIL>", text)
    text = DIGIT_RE.sub("0", text)
    text = SPACE_RE.sub(" ", text)
    return text.strip()
