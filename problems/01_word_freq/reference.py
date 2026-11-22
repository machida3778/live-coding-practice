import re
from collections import Counter


def top_k_words(text: str, k: int, stopwords: set[str] | None = None) -> list[tuple[str, int]]:
    if k <= 0:
        return []
    stop = stopwords or set()
    counts = Counter(w for w in re.findall(r"[a-z]+", text.lower()) if w not in stop)
    return sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:k]
