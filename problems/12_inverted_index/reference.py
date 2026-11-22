import re
from collections import defaultdict

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


class InvertedIndex:
    def __init__(self) -> None:
        # term -> {doc_id: [出現位置, ...]}
        self.postings: dict[str, dict[int, list[int]]] = defaultdict(dict)

    def add(self, doc_id: int, text: str) -> None:
        for pos, tok in enumerate(tokenize(text)):
            self.postings[tok].setdefault(doc_id, []).append(pos)

    def _docs(self, term: str) -> set[int]:
        p = self.postings.get(term.lower())
        return set(p) if p else set()

    def _and(self, terms: list[str]) -> set[int]:
        if not terms:
            return set()
        sets = sorted((self._docs(t) for t in terms), key=len)  # 小さい集合から交差
        result = sets[0]
        for s in sets[1:]:
            if not result:
                break
            result = result & s
        return result

    def search_and(self, terms: list[str]) -> list[int]:
        return sorted(self._and(terms))

    def search_or(self, terms: list[str]) -> list[int]:
        result: set[int] = set()
        for t in terms:
            result |= self._docs(t)
        return sorted(result)

    def search_and_not(self, must: list[str], must_not: list[str]) -> list[int]:
        result = self._and(must)
        for t in must_not:
            result -= self._docs(t)
        return sorted(result)

    def search_phrase(self, phrase: str) -> list[int]:
        toks = tokenize(phrase)
        if not toks:
            return []
        hits = []
        for d in self._and(toks):
            later = [set(self.postings[t][d]) for t in toks[1:]]
            if any(all(p + i in later[i - 1] for i in range(1, len(toks)))
                   for p in self.postings[toks[0]][d]):
                hits.append(d)
        return sorted(hits)
