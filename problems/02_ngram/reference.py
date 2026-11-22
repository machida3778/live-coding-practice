from collections import Counter


def _check(n: int) -> None:
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")


def word_ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    _check(n)
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def char_ngrams(text: str, n: int, pad: bool = False) -> list[str]:
    _check(n)
    if pad:
        text = "^" * (n - 1) + text + "$" * (n - 1)
    return [text[i:i + n] for i in range(len(text) - n + 1)]


def ngram_counts(sentences: list[list[str]], n: int, min_count: int = 1) -> dict[tuple[str, ...], int]:
    _check(n)
    counts = Counter()
    for sent in sentences:
        counts.update(word_ngrams(sent, n))
    return {g: c for g, c in counts.items() if c >= min_count}
