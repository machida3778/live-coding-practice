import re
import unicodedata
from itertools import combinations


def normalize(text: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text).lower())


def shingles(text: str, k: int) -> set[str]:
    t = normalize(text)
    if len(t) < k:
        return {t} if t else set()
    return {t[i:i + k] for i in range(len(t) - k + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


def find_similar_pairs(docs: list[str], k: int, threshold: float) -> list[tuple[int, int, float]]:
    sh = [shingles(d, k) for d in docs]  # 各文書の shingle は 1 回だけ計算する
    out = []
    for i, j in combinations(range(len(docs)), 2):
        s = jaccard(sh[i], sh[j])
        if s >= threshold:
            out.append((i, j, s))
    return out


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:  # 経路圧縮
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, x: int, y: int) -> bool:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        if self.size[rx] < self.size[ry]:  # union by size
            rx, ry = ry, rx
        self.parent[ry] = rx
        self.size[rx] += self.size[ry]
        return True

    def groups(self) -> list[list[int]]:
        g: dict[int, list[int]] = {}
        for x in range(len(self.parent)):  # x の昇順に追加するので各グループは昇順・先頭が最小
            g.setdefault(self.find(x), []).append(x)
        return sorted(g.values(), key=lambda m: m[0])


def group_duplicates(docs: list[str], k: int = 3, threshold: float = 0.5) -> list[list[int]]:
    uf = UnionFind(len(docs))
    for i, j, _ in find_similar_pairs(docs, k, threshold):
        uf.union(i, j)
    return [g for g in uf.groups() if len(g) >= 2]
