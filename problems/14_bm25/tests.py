import itertools
import math
import random

TOL = 1e-9
DOCS = [
    ["the", "cat", "sat", "on", "the", "mat"],
    ["the", "dog", "sat"],
    ["cats", "and", "dogs"],
    ["the", "cat", "and", "the", "dog"],
]


def _close(a, b):
    return math.isclose(a, b, rel_tol=TOL, abs_tol=TOL)


def _ranking_eq(got, exp):
    return [i for i, _ in got] == [i for i, _ in exp] and all(_close(g, e) for (_, g), (_, e) in zip(got, exp))


# ---- 独立した参照実装 (テスト用) ----
def _ref_index(docs, k1=1.5, b=0.75):
    n = len(docs)
    avgdl = sum(len(d) for d in docs) / n
    tfs, df, postings = [], {}, {}
    for i, d in enumerate(docs):
        tf = {}
        for t in d:
            tf[t] = tf.get(t, 0) + 1
        tfs.append(tf)
        for t in tf:
            df[t] = df.get(t, 0) + 1
            postings.setdefault(t, []).append(i)

    def scores(query):
        out = {}
        for i in {i for t in query for i in postings.get(t, [])}:
            s = 0.0
            for t in query:
                f = tfs[i].get(t, 0)
                if f:
                    idf = math.log((n - df[t] + 0.5) / (df[t] + 0.5) + 1)
                    s += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * len(docs[i]) / avgdl))
            if s > 0:
                out[i] = s
        return out

    return scores


def _ref_scores(docs, query, k1=1.5, b=0.75):
    return _ref_index(docs, k1, b)(query)


def test_public_idf(sol):
    bm = sol.BM25(DOCS)
    assert _close(bm.idf("cat"), math.log(2)), f"idf('cat') = {bm.idf('cat')}, expected ln(2)"
    exp = math.log((4 - 3 + 0.5) / (3 + 0.5) + 1)
    assert _close(bm.idf("the"), exp), f"idf('the') = {bm.idf('the')}, expected {exp}"


def test_public_score(sol):
    bm = sol.BM25(DOCS)
    got = [bm.score(["cat"], i) for i in range(4)]
    exp = [0.5847891845915171, 0.0, 0.0, 0.6421527013361892]
    assert all(_close(g, e) for g, e in zip(got, exp)), f"got={got}, expected={exp}"


def test_public_search(sol):
    bm = sol.BM25(DOCS)
    got = bm.search(["cat", "dog"], 3)
    exp = [(3, 1.2843054026723784), (1, 0.7988814962385811), (0, 0.5847891845915171)]
    assert _ranking_eq(got, exp), f"got={got}, expected={exp}"


def test_hidden_unknown_term_idf(sol):
    bm = sol.BM25(DOCS)
    assert _close(bm.idf("bird"), math.log(4.5 / 0.5 + 1)), f"df=0 の idf: {bm.idf('bird')}"
    assert bm.score(["bird"], 0) == 0.0
    assert bm.search(["bird"], 5) == []
    assert bm.search([], 5) == []


def test_hidden_duplicate_query_terms(sol):
    bm = sol.BM25(DOCS)
    for i in range(4):
        assert _close(bm.score(["cat", "cat"], i), 2 * bm.score(["cat"], i)), "重複した語は重複回数だけ加算"
    exp = sorted(_ref_scores(DOCS, ["dog", "the", "dog"]).items(), key=lambda x: (-x[1], x[0]))
    got = bm.search(["dog", "the", "dog"], 10)
    assert _ranking_eq(got, exp), f"got={got}, expected={exp}"


def test_hidden_avgdl_includes_empty_docs(sol):
    docs = [[], ["a", "b"], ["a"]]
    bm = sol.BM25(docs)
    # avgdl = (0 + 2 + 1) / 3 = 1.0 なので docs[2] の分母の正規化項がちょうど k1 になる
    assert _close(bm.score(["a"], 2), math.log(1.6)), f"avgdl に空の文書も含める: score={bm.score(['a'], 2)}"
    assert bm.score(["a"], 0) == 0.0
    got = bm.search(["a"], 10)
    assert [i for i, _ in got] == [2, 1], f"短い文書ほど高スコア: got={got}"


def test_hidden_custom_params(sol):
    docs = [["a", "x", "x", "x"], ["a"], ["a", "a", "y"]]
    bm = sol.BM25(docs, k1=1.2, b=0.0)
    # b = 0 なら文書長の影響は無い
    assert _close(bm.score(["a"], 0), bm.score(["a"], 1)), "b=0 では tf が同じなら文書長に依らず同スコア"
    for q in (["a"], ["a", "y"], ["x", "a"]):
        exp = _ref_scores(docs, q, k1=1.2, b=0.0)
        for i in range(3):
            assert _close(bm.score(q, i), exp.get(i, 0.0)), f"k1=1.2,b=0: q={q}, i={i}"
    bm2 = sol.BM25(docs, k1=2.0, b=1.0)
    exp = _ref_scores(docs, ["a", "x"], k1=2.0, b=1.0)
    for i in range(3):
        assert _close(bm2.score(["a", "x"], i), exp.get(i, 0.0)), f"k1=2,b=1: i={i}"


def test_hidden_tie_and_k(sol):
    docs = [["p", "q"], ["r"], ["p", "q"], ["q", "p"], ["p", "s", "t"]]
    bm = sol.BM25(docs)
    got = bm.search(["p"], 10)
    assert [i for i, _ in got] == [0, 2, 3, 4], f"同点は index 昇順: got={got}"
    assert [i for i, _ in bm.search(["p"], 2)] == [0, 2]
    assert bm.search(["p"], 0) == []
    assert bm.search(["p"], -1) == []


def test_hidden_large(sol):
    rng = random.Random(7)
    vocab = [f"w{i}" for i in range(5000)]
    cum = list(itertools.accumulate(1 / (r + 1) for r in range(len(vocab))))  # Zipf 風
    docs = [rng.choices(vocab, cum_weights=cum, k=rng.randint(5, 30)) for _ in range(10000)]
    bm = sol.BM25(docs)
    ref = _ref_index(docs)
    k = 10
    for _ in range(20):
        q = rng.choices(vocab, cum_weights=cum, k=3)
        got = bm.search(q, k)
        scores = ref(q)
        assert len(got) == min(k, len(scores)), f"件数が違います: {len(got)} vs {min(k, len(scores))}"
        for i, s in got:
            assert i in scores and _close(s, scores[i]), f"doc {i}: {s} vs {scores.get(i)}"
        for (_, a), (_, b) in zip(got, got[1:]):
            assert a >= b - TOL, "降順になっていません"
        if got:
            kth = got[-1][1]
            chosen = {i for i, _ in got}
            assert all(s <= kth + TOL for i, s in scores.items() if i not in chosen), "上位k件になっていません"
