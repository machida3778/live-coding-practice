import math
from collections import Counter


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.n = len(docs)
        self.doc_len = [len(d) for d in docs]
        self.avgdl = sum(self.doc_len) / self.n
        self.tfs = [Counter(d) for d in docs]
        self.df = Counter()
        for tf in self.tfs:
            self.df.update(tf.keys())
        # 転置インデックス: 単語 -> その単語を含む文書 index のリスト
        self.postings: dict[str, list[int]] = {}
        for i, tf in enumerate(self.tfs):
            for t in tf:
                self.postings.setdefault(t, []).append(i)

    def idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log((self.n - df + 0.5) / (df + 0.5) + 1)

    def score(self, query: list[str], i: int) -> float:
        tf = self.tfs[i]
        norm = self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)
        s = 0.0
        for t in query:
            f = tf.get(t, 0)
            if f:
                s += self.idf(t) * f * (self.k1 + 1) / (f + norm)
        return s

    def search(self, query: list[str], k: int) -> list[tuple[int, float]]:
        if k <= 0:
            return []
        cands = {i for t in query for i in self.postings.get(t, [])}
        scored = [(i, self.score(query, i)) for i in cands]
        scored = [x for x in scored if x[1] > 0]
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored[:k]
