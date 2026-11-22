import math
import re
import unicodedata
from collections import Counter, defaultdict


# ---- Part 1
def normalize(text: str) -> str:
    raise NotImplementedError


def tokenize(text: str) -> list[str]:
    raise NotImplementedError


class SearchEngine:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        raise NotImplementedError

    # ---- Part 2
    def add_document(self, text: str) -> int:
        raise NotImplementedError

    def postings(self, term: str) -> dict[int, int]:
        raise NotImplementedError

    def doc_length(self, doc_id: int) -> int:
        raise NotImplementedError

    # ---- Part 3
    def search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        raise NotImplementedError

    # ---- Part 4
    def snippet(self, doc_id: int, query: str, width: int = 10) -> str:
        raise NotImplementedError
