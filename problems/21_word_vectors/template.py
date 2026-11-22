import numpy as np


def normalize_rows(X: np.ndarray) -> np.ndarray:
    raise NotImplementedError


class WordVectors:
    def __init__(self, words: list[str], vectors: np.ndarray):
        raise NotImplementedError

    def similarity(self, w1: str, w2: str) -> float:
        raise NotImplementedError

    def most_similar(self, word: str, k: int = 5) -> list[tuple[str, float]]:
        raise NotImplementedError

    def analogy(self, a: str, b: str, c: str, k: int = 1) -> list[tuple[str, float]]:
        raise NotImplementedError
