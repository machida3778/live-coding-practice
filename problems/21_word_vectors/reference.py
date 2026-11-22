import numpy as np


def normalize_rows(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return np.divide(X, norms, out=np.zeros_like(X), where=norms > 0)


class WordVectors:
    def __init__(self, words: list[str], vectors: np.ndarray):
        self.words = list(words)
        self.index = {w: i for i, w in enumerate(self.words)}
        self.unit = normalize_rows(vectors)  # 事前に正規化しておけば内積 = コサイン類似度

    def similarity(self, w1: str, w2: str) -> float:
        return float(self.unit[self.index[w1]] @ self.unit[self.index[w2]])

    def _top(self, query: np.ndarray, exclude: set[int], k: int) -> list[tuple[str, float]]:
        n = min(k, len(self.words) - len(exclude))
        if n <= 0:
            return []
        q = normalize_rows(query[None, :])[0]
        sims = self.unit @ q
        sims[list(exclude)] = -np.inf
        order = np.argsort(-sims, kind="stable")[:n]  # stable: 同値は index の小さい順
        return [(self.words[i], float(sims[i])) for i in order]

    def most_similar(self, word: str, k: int = 5) -> list[tuple[str, float]]:
        i = self.index[word]
        return self._top(self.unit[i], {i}, k)

    def analogy(self, a: str, b: str, c: str, k: int = 1) -> list[tuple[str, float]]:
        ia, ib, ic = self.index[a], self.index[b], self.index[c]
        target = self.unit[ib] - self.unit[ia] + self.unit[ic]
        return self._top(target, {ia, ib, ic}, k)
