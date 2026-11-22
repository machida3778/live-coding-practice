UNK_POS = "未知語"


def load_dictionary(path: str) -> dict[str, list[tuple[str, int]]]:
    dic: dict[str, list[tuple[str, int]]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            surface, pos, cost = (x.strip() for x in line.split(","))
            dic.setdefault(surface, []).append((pos, int(cost)))
    return dic


def load_connection(path: str) -> dict[tuple[str, str], int]:
    conn: dict[tuple[str, str], int] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            left, right, cost = line.split()
            conn[(left, right)] = int(cost)
    return conn


class LatticeTokenizer:
    def __init__(self, dictionary: dict[str, list[tuple[str, int]]],
                 connection: dict[tuple[str, str], int], unk_cost: int = 10000):
        self.dictionary = dictionary
        self.connection = connection
        self.unk_cost = unk_cost
        self.max_len = max((len(s) for s in dictionary), default=0)

    def lookup(self, text: str, i: int) -> list[tuple[str, str, int]]:
        found = []
        # 最大語長までの部分文字列だけを辞書引きする (辞書全体を走査しない)
        for length in range(1, min(self.max_len, len(text) - i) + 1):
            surface = text[i:i + length]
            for pos, cost in self.dictionary.get(surface, []):
                found.append((surface, pos, cost))
        if not found:
            found.append((text[i], UNK_POS, self.unk_cost))
        return found

    def _viterbi(self, text: str) -> tuple[int, list[tuple[str, str]]]:
        n = len(text)
        # ends[j] = 位置 j で終わるノードのリスト: (累積コスト, 品詞, 表層, 前ノード)
        ends: list[list[tuple]] = [[] for _ in range(n + 1)]
        ends[0].append((0, "BOS", "", None))
        for i in range(n):
            if not ends[i]:
                continue
            for surface, pos, cost in self.lookup(text, i):
                best = min(ends[i], key=lambda p: p[0] + self.connection.get((p[1], pos), 0))
                total = best[0] + self.connection.get((best[1], pos), 0) + cost
                ends[i + len(surface)].append((total, pos, surface, best))
        last = min(ends[n], key=lambda p: p[0] + self.connection.get((p[1], "EOS"), 0))
        total = last[0] + self.connection.get((last[1], "EOS"), 0)
        path = []
        node = last
        while node[3] is not None:
            path.append((node[2], node[1]))
            node = node[3]
        return total, path[::-1]

    def tokenize(self, text: str) -> list[tuple[str, str]]:
        return self._viterbi(text)[1]

    def best_cost(self, text: str) -> int:
        return self._viterbi(text)[0]
