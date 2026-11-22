import math
from collections import Counter


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        raise NotImplementedError

    def idf(self, term: str) -> float:
        raise NotImplementedError

    def score(self, query: list[str], i: int) -> float:
        raise NotImplementedError

    def search(self, query: list[str], k: int) -> list[tuple[int, float]]:
        raise NotImplementedError
