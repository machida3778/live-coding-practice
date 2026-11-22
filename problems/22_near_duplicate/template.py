def normalize(text: str) -> str:
    raise NotImplementedError


def shingles(text: str, k: int) -> set[str]:
    raise NotImplementedError


def jaccard(a: set[str], b: set[str]) -> float:
    raise NotImplementedError


def find_similar_pairs(docs: list[str], k: int, threshold: float) -> list[tuple[int, int, float]]:
    raise NotImplementedError


class UnionFind:
    def __init__(self, n: int):
        raise NotImplementedError

    def find(self, x: int) -> int:
        raise NotImplementedError

    def union(self, x: int, y: int) -> bool:
        raise NotImplementedError

    def groups(self) -> list[list[int]]:
        raise NotImplementedError


def group_duplicates(docs: list[str], k: int = 3, threshold: float = 0.5) -> list[list[int]]:
    raise NotImplementedError
