import itertools
import math
import random

TOL = 1e-9
DOCS = [["cat", "sat", "on", "mat"], ["dog", "sat", "on", "log"], ["cat", "cat", "dog"]]


def _close(a, b):
    return math.isclose(a, b, rel_tol=TOL, abs_tol=TOL)


def _dict_close(got, exp):
    return set(got) == set(exp) and all(_close(got[t], exp[t]) for t in exp)


def _ranking_eq(got, exp):
    """index の並びが完全一致し、スコアが許容誤差内で一致するか"""
    return [i for i, _ in got] == [i for i, _ in exp] and all(_close(g, e) for (_, g), (_, e) in zip(got, exp))


# ---- テスト内で使う独立した参照実装 (大規模テスト用) ----
def _ref_scores(docs, query):
    n = len(docs)
    df = {}
    for d in docs:
        for t in set(d):
            df[t] = df.get(t, 0) + 1
    idf = {t: math.log(n / c) + 1 for t, c in df.items()}

    def vec(d):
        if not d:
            return {}
        out = {}
        for t in d:
            if t in idf:
                out[t] = out.get(t, 0.0) + 1.0
        return {t: c / len(d) * idf[t] for t, c in out.items()}

    q = vec(query)
    qn = math.sqrt(sum(w * w for w in q.values()))
    scores = {}
    for i, d in enumerate(docs):
        v = vec(d)
        vn = math.sqrt(sum(w * w for w in v.values()))
        if qn == 0 or vn == 0:
            continue
        s = sum(w * v.get(t, 0.0) for t, w in q.items()) / (qn * vn)
        if s > 0:
            scores[i] = s
    return scores


def test_public_compute_idf(sol):
    idf = sol.compute_idf(DOCS)
    exp = {"cat": math.log(3 / 2) + 1, "sat": math.log(3 / 2) + 1, "on": math.log(3 / 2) + 1,
           "mat": math.log(3) + 1, "dog": math.log(3 / 2) + 1, "log": math.log(3) + 1}
    assert _dict_close(idf, exp), f"got={idf}, expected={exp}"


def test_public_tfidf_and_cosine(sol):
    idf = sol.compute_idf(DOCS)
    got = sol.tfidf(["cat", "cat", "dog"], idf)
    exp = {"cat": 2 / 3 * (math.log(1.5) + 1), "dog": 1 / 3 * (math.log(1.5) + 1)}
    assert _dict_close(got, exp), f"tfidf: got={got}, expected={exp}"
    got = sol.tfidf(["cat", "bird"], idf)
    exp = {"cat": 0.5 * (math.log(1.5) + 1)}
    assert _dict_close(got, exp), f"tfidf (未知語入り): got={got}, expected={exp}"
    c = sol.cosine({"a": 1.0, "b": 1.0}, {"a": 1.0})
    assert _close(c, 1 / math.sqrt(2)), f"cosine: got={c}"


def test_public_search(sol):
    got = sol.search(DOCS, ["cat"], 2)
    exp = [(2, 0.8944271909999157), (0, 0.43728676490568713)]
    assert _ranking_eq(got, exp), f"got={got}, expected={exp}"
    got = sol.search(DOCS, ["sat", "dog"], 3)
    exp = [(1, 0.6184168735758779), (2, 0.3162277660168379), (0, 0.30920843678793897)]
    assert _ranking_eq(got, exp), f"got={got}, expected={exp}"


def test_hidden_empty_inputs(sol):
    assert sol.compute_idf([]) == {}, "docs が空なら空の辞書"
    assert sol.tfidf([], {"a": 1.0}) == {}, "空の文書は空の辞書"
    assert sol.search([], ["a"], 3) == []


def test_hidden_cosine_zero_norm(sol):
    assert sol.cosine({}, {"a": 1.0}) == 0.0
    assert sol.cosine({"a": 1.0}, {}) == 0.0
    assert sol.cosine({}, {}) == 0.0
    assert _close(sol.cosine({"a": 2.0}, {"b": 3.0}), 0.0)
    assert _close(sol.cosine({"a": 1.0, "b": 2.0}, {"a": 2.0, "b": 4.0}), 1.0)


def test_hidden_empty_doc_counts_in_N(sol):
    docs = [[], ["a", "b"], ["a"]]
    idf = sol.compute_idf(docs)
    assert _close(idf["a"], math.log(3 / 2) + 1), f"N には空の文書も含める: idf[a]={idf['a']}"
    assert _close(idf["b"], math.log(3) + 1)
    got = sol.search(docs, ["a"], 10)
    assert [i for i, _ in got] == [2, 1], f"空の文書は類似度0なので含めない: got={got}"


def test_hidden_query_unknown_words(sol):
    base = sol.search(DOCS, ["cat"], 10)
    got = sol.search(DOCS, ["cat", "unknown", "words"], 10)
    assert _ranking_eq(got, base), f"未知語はベクトルに入らない (コサインは定数倍に不変): got={got}, expected={base}"
    assert sol.search(DOCS, ["unknown"], 10) == []
    assert sol.search(DOCS, [], 10) == []


def test_hidden_query_duplicates(sol):
    got = sol.search(DOCS, ["dog", "dog", "sat"], 10)
    exp = sorted(_ref_scores(DOCS, ["dog", "dog", "sat"]).items(), key=lambda x: (-x[1], x[0]))
    assert _ranking_eq(got, exp), f"got={got}, expected={exp}"


def test_hidden_tie_break(sol):
    docs = [["x", "y"], ["a", "b"], ["x", "y"], ["x", "z"], ["z", "c"]]
    got = sol.search(docs, ["x"], 10)
    assert [i for i, _ in got] == [0, 2, 3], f"同点は index 昇順, 類似度0は除外: got={got}"
    assert _close(got[0][1], got[1][1]) and _close(got[1][1], got[2][1])


def test_hidden_k_bounds(sol):
    assert sol.search(DOCS, ["cat"], 0) == []
    assert sol.search(DOCS, ["cat"], -2) == []
    got = sol.search(DOCS, ["sat", "dog"], 1)
    assert [i for i, _ in got] == [1], got


def test_hidden_large(sol):
    rng = random.Random(42)
    vocab = [f"w{i}" for i in range(2000)]
    cum = list(itertools.accumulate(1 / (r + 1) for r in range(len(vocab))))  # Zipf 風
    docs = [rng.choices(vocab, cum_weights=cum, k=rng.randint(5, 60)) for _ in range(3000)]
    queries = [rng.choices(vocab, cum_weights=cum, k=4) for _ in range(5)] + [["w1500", "w1999"]]
    k = 10
    for q in queries:
        got = sol.search(docs, q, k)
        scores = _ref_scores(docs, q)
        assert len(got) == min(k, len(scores)), f"件数が違います: {len(got)} vs {min(k, len(scores))}"
        for i, s in got:
            assert i in scores and _close(s, scores[i]), f"doc {i} のスコアが違います: {s} vs {scores.get(i)}"
        for (_, a), (_, b) in zip(got, got[1:]):
            assert a >= b - TOL, "降順になっていません"
        if got:
            kth = got[-1][1]
            chosen = {i for i, _ in got}
            assert all(s <= kth + TOL for i, s in scores.items() if i not in chosen), "上位k件になっていません"
