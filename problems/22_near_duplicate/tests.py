import math
import random

DOCS = ["今日は良い天気ですね。", "今日は 良い天気ですね！", "明日は雨が降るでしょう。",
        "今日はとても良い天気ですね。", "明日は雨が降るでしょう", "ＮＥＸＵＳ Technology", "nexus technology!"]


def _same_pairs(got, exp):
    assert [(i, j) for i, j, _ in got] == [(i, j) for i, j, _ in exp], (got, exp)
    assert all(math.isclose(a, b, rel_tol=1e-9) for (_, _, a), (_, _, b) in zip(got, exp)), (got, exp)


# ---------------------------------------------------------------- public
def test_public_normalize_and_shingles(sol):
    assert sol.normalize("ＡＢ c\td\n") == "abcd", sol.normalize("ＡＢ c\td\n")
    assert sol.shingles("ＡＢ cd", 2) == {"ab", "bc", "cd"}, sol.shingles("ＡＢ cd", 2)


def test_public_jaccard(sol):
    assert math.isclose(sol.jaccard({"a", "b", "c"}, {"b", "c", "d"}), 0.5)
    assert math.isclose(sol.jaccard({"a"}, {"b"}), 0.0)


def test_public_find_similar_pairs_example(sol):
    got = sol.find_similar_pairs(DOCS, 2, 0.5)
    exp = [(0, 1, 9 / 11), (0, 3, 9 / 14), (1, 3, 8 / 15), (2, 4, 10 / 11), (5, 6, 14 / 15)]
    _same_pairs(got, exp)


def test_public_group_duplicates_example(sol):
    assert sol.group_duplicates(DOCS, k=2, threshold=0.5) == [[0, 1, 3], [2, 4], [5, 6]]


# ---------------------------------------------------------------- hidden
def test_hidden_shingles_edge_cases(sol):
    assert sol.shingles("ab", 3) == {"ab"}
    assert sol.shingles("  ", 2) == set()
    assert sol.shingles("", 1) == set()
    assert sol.shingles("aaaa", 2) == {"aa"}
    assert sol.shingles("ｱｲｳ", 2) == {"アイ", "イウ"}  # 半角カナも NFKC で全角に


def test_hidden_jaccard_empty(sol):
    assert sol.jaccard(set(), set()) == 1.0
    assert sol.jaccard(set(), {"a"}) == 0.0


def test_hidden_threshold_inclusive_and_transitive(sol):
    docs = ["abcdefghij", "abcdefghxy", "abcdwvghxy"]  # 0-1: 7/11, 1-2: ちょうど 0.5, 0-2: 4/14
    _same_pairs(sol.find_similar_pairs(docs, 2, 0.5), [(0, 1, 7 / 11), (1, 2, 0.5)])
    assert sol.group_duplicates(docs, k=2, threshold=0.5) == [[0, 1, 2]], "推移的にまとめてください"
    assert sol.group_duplicates(docs, k=2, threshold=0.51) == [[0, 1]]
    assert sol.group_duplicates(docs, k=2, threshold=0.9) == []


def test_hidden_union_find(sol):
    uf = sol.UnionFind(6)
    assert uf.union(4, 2) is True
    assert uf.union(2, 4) is False
    assert uf.union(5, 0) is True
    assert uf.union(0, 2) is True
    assert uf.find(4) == uf.find(5) and uf.find(1) != uf.find(4) and uf.find(3) == 3
    assert uf.groups() == [[0, 2, 4, 5], [1], [3]], uf.groups()


def test_hidden_union_find_long_chain(sol):
    n = 200000
    uf = sol.UnionFind(n)
    for i in range(n - 1):
        uf.union(i, i + 1)
    r = uf.find(0)
    assert all(uf.find(i) == r for i in range(0, n, 997))
    assert len(uf.groups()) == 1


def test_hidden_group_order_and_singletons_excluded(sol):
    docs = ["xyz unique", "hello world", "completely different", "Hello  World", "ｘｙｚ ｕｎｉｑｕｅ", "hello world!!"]
    assert sol.group_duplicates(docs, k=3, threshold=0.6) == [[0, 4], [1, 3, 5]]


def _gen_corpus(seed=0):
    rng = random.Random(seed)
    pool = "あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをん日本語自然言語処理学習機械検索文書重複"
    base = ["".join(rng.choice(pool) for _ in range(rng.randint(80, 120))) for _ in range(120)]
    docs = list(base)
    for b in base[:80]:
        for _ in range(rng.randint(1, 3)):
            s = list(b)
            for _ in range(rng.randint(1, 6)):
                op, p = rng.random(), rng.randrange(len(s))
                if op < 0.4:
                    s[p] = rng.choice(pool)
                elif op < 0.7:
                    s.insert(p, rng.choice(pool + " 　"))
                else:
                    del s[p]
            docs.append("".join(s))
    docs += ["".join(rng.choice(pool) for _ in range(rng.randint(80, 120))) for _ in range(100)]
    rng.shuffle(docs)
    return docs


def _expected_groups(docs, k, th):
    import re
    import unicodedata
    sh = []
    for d in docs:
        t = re.sub(r"\s+", "", unicodedata.normalize("NFKC", d).lower())
        sh.append({t[i:i + k] for i in range(len(t) - k + 1)} if len(t) >= k else ({t} if t else set()))
    parent = list(range(len(docs)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(len(docs)):
        for j in range(i + 1, len(docs)):
            inter = len(sh[i] & sh[j])
            union = len(sh[i]) + len(sh[j]) - inter
            if (inter / union if union else 1.0) >= th:
                parent[find(i)] = find(j)
    g = {}
    for x in range(len(docs)):
        g.setdefault(find(x), []).append(x)
    return sorted([m for m in g.values() if len(m) >= 2], key=lambda m: m[0])


def test_hidden_performance_and_correctness(sol):
    docs = _gen_corpus()
    assert len(docs) > 350
    exp = _expected_groups(docs, 3, 0.6)
    assert len(exp) >= 50
    got = sol.group_duplicates(docs, k=3, threshold=0.6)
    assert got == exp, f"グループ数 {len(got)} (期待 {len(exp)})"
