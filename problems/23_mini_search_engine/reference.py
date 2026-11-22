import math
import re
import unicodedata
from collections import Counter, defaultdict

# 日本語文字: 々, ひらがな, カタカナ(・を除く), 長音符, CJK統合漢字
JP_CHARS = "々ぁ-ゖァ-ヺー一-鿿"
RUN_RE = re.compile(rf"[a-z0-9]+|[{JP_CHARS}]+")
ASCII_RE = re.compile(r"[a-z0-9]+")


def normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text).lower()


def _runs(text: str) -> list[str]:
    """正規化後の「英数字の連続」「日本語文字の連続」を出現順に返す"""
    return RUN_RE.findall(normalize(text))


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for run in _runs(text):
        if ASCII_RE.fullmatch(run) or len(run) == 1:
            tokens.append(run)
        else:
            tokens.extend(run[i:i + 2] for i in range(len(run) - 1))
    return tokens


class SearchEngine:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs: list[str] = []
        self.lengths: list[int] = []
        self.index: dict[str, dict[int, int]] = {}  # 転置インデックス: term -> {doc_id: tf}

    # ---- Part 2: インデックス
    def add_document(self, text: str) -> int:
        doc_id = len(self.docs)
        tokens = tokenize(text)
        self.docs.append(text)
        self.lengths.append(len(tokens))
        for term, tf in Counter(tokens).items():
            self.index.setdefault(term, {})[doc_id] = tf
        return doc_id

    def postings(self, term: str) -> dict[int, int]:
        return dict(self.index.get(term, {}))

    def doc_length(self, doc_id: int) -> int:
        return self.lengths[doc_id]

    # ---- Part 3: BM25 検索
    def search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        terms = set(tokenize(query))
        n_docs = len(self.docs)
        if n_docs == 0 or not terms or k <= 0:
            return []
        avgdl = sum(self.lengths) / n_docs
        scores: dict[int, float] = defaultdict(float)
        for term in sorted(terms):
            post = self.index.get(term)
            if not post:
                continue
            df = len(post)
            idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            for doc_id, tf in post.items():
                norm = self.k1 * (1 - self.b + self.b * self.lengths[doc_id] / avgdl)
                scores[doc_id] += idf * tf * (self.k1 + 1) / (tf + norm)
        return sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:k]

    # ---- Part 4: スニペット
    def snippet(self, doc_id: int, query: str, width: int = 10) -> str:
        text = normalize(self.docs[doc_id])
        intervals = []
        for term in set(_runs(query)):
            pos = text.find(term)
            while pos != -1:
                intervals.append((pos, pos + len(term)))
                pos = text.find(term, pos + 1)
        if not intervals:
            return text[:2 * width] + ("…" if len(text) > 2 * width else "")
        intervals.sort()
        merged: list[list[int]] = []
        for s, e in intervals:
            if merged and s <= merged[-1][1]:  # 重なり・隣接はまとめる
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        p, q = merged[0]
        ws, we = max(0, p - width), min(len(text), q + width)
        parts, cur = [], ws
        for s, e in merged:
            if ws <= s and e <= we:  # ウィンドウに完全に含まれるものだけ強調
                parts += [text[cur:s], "【", text[s:e], "】"]
                cur = e
        parts.append(text[cur:we])
        return ("…" if ws > 0 else "") + "".join(parts) + ("…" if we < len(text) else "")
