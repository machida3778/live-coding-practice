import math

import numpy as np

WORDS = ["man", "woman", "king", "queen", "apple", "prince"]
VECS = np.array([[0, 1, 0], [0, -1, 0], [1, 1, 0], [1, -1, 0], [0, 0, 1], [1, 1, 0.2]], dtype=float)


def _cos(u, v):
    nu, nv = math.sqrt(sum(x * x for x in u)), math.sqrt(sum(x * x for x in v))
    if nu == 0 or nv == 0:
        return 0.0
    return sum(a * b for a, b in zip(u, v)) / (nu * nv)


def _unit(u):
    n = math.sqrt(sum(x * x for x in u))
    return [x / n for x in u] if n else [0.0] * len(u)


def _naive_top(words, vecs, q, exclude, k):
    cand = [(i, _cos(vecs[i], q)) for i in range(len(words)) if i not in exclude]
    cand.sort(key=lambda t: (-t[1], t[0]))
    return [(words[i], s) for i, s in cand[:max(k, 0)]]


def _assert_same(got, exp):
    assert [w for w, _ in got] == [w for w, _ in exp], (got, exp)
    assert all(math.isclose(a, b, abs_tol=1e-9) for (_, a), (_, b) in zip(got, exp)), (got, exp)


# ---------------------------------------------------------------- public
def test_public_normalize_rows(sol):
    X = np.array([[3.0, 4.0], [0.0, 0.0], [1.0, 0.0]])
    got = sol.normalize_rows(X)
    assert np.allclose(got, [[0.6, 0.8], [0.0, 0.0], [1.0, 0.0]]), got
    assert np.array_equal(X, [[3.0, 4.0], [0.0, 0.0], [1.0, 0.0]]), "入力 X が変更されています"


def test_public_example(sol):
    wv = sol.WordVectors(WORDS, VECS)
    assert math.isclose(wv.similarity("king", "queen"), 0.0, abs_tol=1e-12)
    _assert_same(wv.most_similar("king", 2), [("prince", 0.9901475429766743), ("man", 0.7071067811865475)])
    got = wv.analogy("man", "king", "woman")
    assert len(got) == 1 and got[0][0] == "queen" and math.isclose(got[0][1], 0.9596829822606673, rel_tol=1e-9), got


# ---------------------------------------------------------------- hidden
def test_hidden_zero_vector_and_ties(sol):
    words = ["a", "b", "c", "d", "e"]
    vecs = np.array([[1, 0], [0, 1], [0, -1], [1, 1], [0, 0]], dtype=float)
    wv = sol.WordVectors(words, vecs)
    assert wv.similarity("a", "e") == 0.0
    _assert_same(wv.most_similar("a", 4), [("d", 1 / math.sqrt(2)), ("b", 0.0), ("c", 0.0), ("e", 0.0)])
    _assert_same(wv.most_similar("e", 2), [("a", 0.0), ("b", 0.0)])


def test_hidden_k_bounds(sol):
    wv = sol.WordVectors(WORDS, VECS)
    assert wv.most_similar("king", 0) == []
    assert len(wv.most_similar("king", 100)) == len(WORDS) - 1
    assert len(wv.analogy("man", "king", "woman", k=100)) == len(WORDS) - 3
    assert wv.analogy("man", "king", "woman", k=0) == []


def test_hidden_unknown_word(sol):
    wv = sol.WordVectors(WORDS, VECS)
    for call in (lambda: wv.similarity("king", "dog"), lambda: wv.most_similar("dog"),
                 lambda: wv.analogy("man", "dog", "woman")):
        try:
            call()
        except KeyError:
            continue
        raise AssertionError("語彙に無い単語で KeyError が送出されていません")


def test_hidden_analogy_excludes_inputs_and_uses_unit_vectors(sol):
    # ノルムがばらばらなベクトル: 正規化してから足し引きしないと答えが変わる
    rng = np.random.default_rng(5)
    words = [f"w{i}" for i in range(40)]
    vecs = rng.normal(size=(40, 6)) * rng.uniform(0.1, 10, size=(40, 1))
    wv = sol.WordVectors(words, vecs)
    for a, b, c in [(0, 1, 2), (3, 4, 5), (7, 7, 8), (10, 20, 30)]:
        ua, ub, uc = _unit(vecs[a]), _unit(vecs[b]), _unit(vecs[c])
        target = [y - x + z for x, y, z in zip(ua, ub, uc)]
        exp = _naive_top(words, vecs, target, {a, b, c}, 5)
        _assert_same(wv.analogy(words[a], words[b], words[c], k=5), exp)


def test_hidden_most_similar_random(sol):
    rng = np.random.default_rng(6)
    words = [f"w{i}" for i in range(200)]
    vecs = rng.normal(size=(200, 8))
    vecs[17] = 0.0
    wv = sol.WordVectors(words, vecs)
    for i in (0, 17, 199):
        _assert_same(wv.most_similar(words[i], 7), _naive_top(words, vecs, vecs[i], {i}, 7))
    assert math.isclose(wv.similarity("w3", "w4"), _cos(vecs[3], vecs[4]), abs_tol=1e-12)


def test_hidden_performance(sol):
    rng = np.random.default_rng(7)
    V, d = 50000, 100
    words = [f"w{i}" for i in range(V)]
    vecs = rng.normal(size=(V, d))
    wv = sol.WordVectors(words, vecs)
    unit = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    for t in range(200):
        i = (t * 7919) % V
        got = wv.most_similar(words[i], 10)
        if t % 50 == 0:
            sims = unit @ unit[i]
            sims[i] = -np.inf
            exp_idx = np.argsort(-sims, kind="stable")[:10]
            assert [w for w, _ in got] == [words[j] for j in exp_idx]
    for t in range(50):
        wv.analogy(words[t], words[t + 1], words[t + 2], k=5)
