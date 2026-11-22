from collections import Counter

EOW = "</w>"


def _merge(symbols: list[str], pair: tuple[str, str]) -> list[str]:
    """symbols 中の pair を左から順に、重ならないように連結する"""
    a, b = pair
    out = []
    i, n = 0, len(symbols)
    while i < n:
        if i + 1 < n and symbols[i] == a and symbols[i + 1] == b:
            out.append(a + b)
            i += 2
        else:
            out.append(symbols[i])
            i += 1
    return out


def train_bpe(corpus: dict[str, int], num_merges: int) -> list[tuple[str, str]]:
    words = [(list(w) + [EOW], freq) for w, freq in corpus.items()]
    merges: list[tuple[str, str]] = []
    for _ in range(num_merges):
        counts: Counter = Counter()
        for syms, freq in words:
            for pair in zip(syms, syms[1:]):
                counts[pair] += freq
        if not counts:
            break
        best = min(counts, key=lambda p: (-counts[p], p))
        merges.append(best)
        words = [(_merge(syms, best), freq) for syms, freq in words]
    return merges


def encode(word: str, merges: list[tuple[str, str]]) -> list[str]:
    syms = list(word) + [EOW]
    for pair in merges:
        if len(syms) == 1:
            break
        syms = _merge(syms, pair)
    return syms
