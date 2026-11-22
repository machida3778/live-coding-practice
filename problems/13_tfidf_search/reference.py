import math
from collections import Counter


def compute_idf(docs: list[list[str]]) -> dict[str, float]:
    n = len(docs)
    df = Counter()
    for d in docs:
        df.update(set(d))
    return {t: math.log(n / c) + 1 for t, c in df.items()}


def tfidf(doc: list[str], idf: dict[str, float]) -> dict[str, float]:
    if not doc:
        return {}
    length = len(doc)
    return {t: c / length * idf[t] for t, c in Counter(doc).items() if t in idf}


def cosine(u: dict[str, float], v: dict[str, float]) -> float:
    nu = math.sqrt(sum(w * w for w in u.values()))
    nv = math.sqrt(sum(w * w for w in v.values()))
    if nu == 0 or nv == 0:
        return 0.0
    if len(u) > len(v):
        u, v = v, u
    dot = sum(w * v.get(t, 0.0) for t, w in u.items())
    return dot / (nu * nv)


def search(docs: list[list[str]], query: list[str], k: int) -> list[tuple[int, float]]:
    if k <= 0:
        return []
    idf = compute_idf(docs)
    q = tfidf(query, idf)
    if not q:
        return []
    scored = []
    for i, d in enumerate(docs):
        s = cosine(q, tfidf(d, idf))
        if s > 0:
            scored.append((i, s))
    scored.sort(key=lambda x: (-x[1], x[0]))
    return scored[:k]
