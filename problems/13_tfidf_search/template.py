import math
from collections import Counter


def compute_idf(docs: list[list[str]]) -> dict[str, float]:
    raise NotImplementedError


def tfidf(doc: list[str], idf: dict[str, float]) -> dict[str, float]:
    raise NotImplementedError


def cosine(u: dict[str, float], v: dict[str, float]) -> float:
    raise NotImplementedError


def search(docs: list[list[str]], query: list[str], k: int) -> list[tuple[int, float]]:
    raise NotImplementedError
