def word_ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    raise NotImplementedError


def char_ngrams(text: str, n: int, pad: bool = False) -> list[str]:
    raise NotImplementedError


def ngram_counts(sentences: list[list[str]], n: int, min_count: int = 1) -> dict[tuple[str, ...], int]:
    raise NotImplementedError
