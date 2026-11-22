import math

STATES = ["DET", "NOUN", "VERB"]
START = {"DET": 0.6, "NOUN": 0.3, "VERB": 0.1}
TRANS = {
    "DET": {"DET": 0.05, "NOUN": 0.9, "VERB": 0.05},
    "NOUN": {"DET": 0.1, "NOUN": 0.3, "VERB": 0.6},
    "VERB": {"DET": 0.5, "NOUN": 0.3, "VERB": 0.2},
}
EMIT = {
    "DET": {"the": 0.7, "a": 0.3},
    "NOUN": {"dog": 0.4, "cat": 0.3, "runs": 0.1, "walk": 0.2},
    "VERB": {"runs": 0.5, "walk": 0.4, "sees": 0.1},
}


def _run(sol, obs, states=STATES, start=START, trans=TRANS, emit=EMIT):
    path, logp = sol.viterbi(list(obs), states, start, trans, emit)
    return list(path), logp


def _check(got, exp_path, exp_logp):
    path, logp = got
    assert path == exp_path, f"状態列: 期待 {exp_path}, 実際 {path}"
    assert math.isclose(logp, exp_logp, rel_tol=1e-6, abs_tol=1e-9), f"対数確率: 期待 {exp_logp}, 実際 {logp}"


def test_public_example(sol):
    got = _run(sol, ["the", "dog", "runs"])
    _check(got, ["DET", "NOUN", "VERB"], math.log(0.6 * 0.7 * 0.9 * 0.4 * 0.6 * 0.5))


def test_public_context_dependent(sol):
    # 出力確率だけなら walk は VERB (0.4 > 0.2) だが、DET の後なので NOUN
    got = _run(sol, ["a", "walk"])
    _check(got, ["DET", "NOUN"], math.log(0.6 * 0.3 * 0.9 * 0.2))
    got = _run(sol, ["dog", "walk", "the", "walk"])
    _check(got, ["NOUN", "VERB", "DET", "NOUN"], math.log(0.3 * 0.4 * 0.6 * 0.4 * 0.5 * 0.7 * 0.9 * 0.2))


def test_public_empty(sol):
    path, logp = _run(sol, [])
    assert path == [] and logp == 0.0, (path, logp)


def test_hidden_unknown_word(sol):
    got = _run(sol, ["the", "xyz", "runs"])
    _check(got, ["DET", "NOUN", "VERB"], math.log(0.6 * 0.7 * 0.9 * 1e-6 * 0.6 * 0.5))


def test_hidden_single_word(sol):
    got = _run(sol, ["runs"])
    _check(got, ["VERB"], math.log(0.1 * 0.5))


def test_hidden_longer_sentence(sol):
    got = _run(sol, ["the", "dog", "sees", "the", "cat", "walk"])
    exp = 0.6 * 0.7 * 0.9 * 0.4 * 0.6 * 0.1 * 0.5 * 0.7 * 0.9 * 0.3 * 0.6 * 0.4
    _check(got, ["DET", "NOUN", "VERB", "DET", "NOUN", "VERB"], math.log(exp))


def test_hidden_missing_transitions(sol):
    # B は先頭に来られず、A→A / B→B の遷移も無い → A B A しかありえない
    states = ["A", "B"]
    start = {"A": 1.0}
    trans = {"A": {"B": 1.0}, "B": {"A": 1.0}}
    emit = {"A": {"x": 0.5, "y": 0.5}, "B": {"x": 0.9, "y": 0.1}}
    got = _run(sol, ["x", "x", "x"], states, start, trans, emit)
    _check(got, ["A", "B", "A"], math.log(0.5 * 0.9 * 0.5))


def test_hidden_no_possible_path(sol):
    states = ["A", "B"]
    start = {"A": 1.0}
    trans = {"A": {}, "B": {"A": 1.0}}
    emit = {"A": {"x": 0.5}, "B": {"x": 0.5}}
    path, logp = _run(sol, ["x", "x"], states, start, trans, emit)
    assert path == [] and logp == float("-inf"), f"期待 ([], -inf), 実際 ({path}, {logp})"


def test_hidden_underflow_long(sol):
    # 確率の積は 0.2^4000 ≈ 1e-2796 になり float ではアンダーフローする
    import random
    rng = random.Random(0)
    states = [f"S{k}" for k in range(5)]
    start = {s: 0.2 for s in states}
    trans = {s: {t: 0.2 for t in states} for s in states}
    emit = {f"S{k}": {f"w{k}": 0.2, f"z{k}": 0.8} for k in range(5)}
    T = 2000
    ks = [rng.randrange(5) for _ in range(T)]
    obs = [f"w{k}" for k in ks]
    got = _run(sol, obs, states, start, trans, emit)
    _check(got, [f"S{k}" for k in ks], 2 * T * math.log(0.2))


def test_hidden_large_random(sol):
    import hashlib
    import random
    rng = random.Random(3)
    S, T, V = 40, 500, 100
    states = [f"s{i}" for i in range(S)]

    def norm(xs):
        t = sum(xs)
        return [x / t for x in xs]

    start = dict(zip(states, norm([rng.random() + 0.01 for _ in states])))
    trans = {s: dict(zip(states, norm([rng.random() + 0.01 for _ in states]))) for s in states}
    words = [f"w{j}" for j in range(V)]
    emit = {s: dict(zip(words, norm([rng.random() + 0.01 for _ in words]))) for s in states}
    obs = [rng.choice(words) for _ in range(T)]
    path, logp = _run(sol, obs, states, start, trans, emit)
    assert math.isclose(logp, -3527.20042375674, rel_tol=1e-6), f"対数確率: 期待 -3527.2004..., 実際 {logp}"
    assert len(path) == T, len(path)
    assert path[:5] == ["s26", "s4", "s23", "s33", "s33"], path[:5]
    assert hashlib.md5(",".join(path).encode()).hexdigest() == "1e81c6934c231e3e104c954aca29817e", "状態列が期待と異なります"
