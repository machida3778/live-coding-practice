import math


def _check(got, exp):
    got = tuple(got)
    assert len(got) == 3, f"(P, R, F) の3要素を返してください: {got}"
    assert all(math.isclose(g, e, rel_tol=1e-9, abs_tol=1e-9) for g, e in zip(got, exp)), f"期待値 {exp}, 実際 {got}"


def test_public_lcs_example(sol):
    got = sol.lcs_length(list("ABCBDAB"), list("BDCABA"))
    assert got == 4, got


def test_public_rouge_example(sol):
    got = sol.rouge_l("the cat sat on the mat".split(), "the cat is on the mat".split())
    _check(got, (5 / 6, 5 / 6, 5 / 6))


def test_public_word_order(sol):
    got = sol.rouge_l("police killed the gunman".split(), "the gunman police killed".split())
    _check(got, (0.5, 0.5, 0.5))


def test_hidden_empty_inputs(sol):
    _check(sol.rouge_l([], ["a"]), (0.0, 0.0, 0.0))
    _check(sol.rouge_l(["a"], []), (0.0, 0.0, 0.0))
    _check(sol.rouge_l([], []), (0.0, 0.0, 0.0))
    assert sol.lcs_length([], []) == 0
    assert sol.lcs_length(["a"], []) == 0


def test_hidden_no_overlap(sol):
    assert sol.lcs_length(["a", "b"], ["c", "d"]) == 0
    _check(sol.rouge_l(["a", "b"], ["c", "d"]), (0.0, 0.0, 0.0))


def test_hidden_identical(sol):
    x = "to be or not to be".split()
    _check(sol.rouge_l(x, x), (1.0, 1.0, 1.0))


def test_hidden_non_contiguous(sol):
    assert sol.lcs_length("a x b y c".split(), "a b c".split()) == 3
    assert sol.lcs_length("a b c".split(), "a x b y c".split()) == 3


def test_hidden_repeated_tokens(sol):
    assert sol.lcs_length(["a", "a", "a"], ["a", "a"]) == 2
    assert sol.lcs_length(list("aab"), list("aba")) == 2
    assert sol.lcs_length(list("abab"), list("baba")) == 3


def test_hidden_uneven_lengths(sol):
    _check(sol.rouge_l(["a", "b"], ["a", "x", "b", "y"]), (1.0, 0.5, 2 / 3))
    _check(sol.rouge_l(["a", "x", "b", "y"], ["a", "b"]), (0.5, 1.0, 2 / 3))


def test_hidden_beta(sol):
    _check(sol.rouge_l(["a", "b"], ["a", "b", "c", "d"], beta=2.0), (1.0, 0.5, 2.5 / 4.5))
    _check(sol.rouge_l(["a", "b"], ["a", "x", "b", "y"], beta=0.5), (1.0, 0.5, 1.25 * 0.5 / 0.75))


def test_hidden_large(sol):
    # a の部分列 s に、a に現れないトークン "<unk>" を挿入したものを b とする → LCS(a, b) = len(s)
    import random
    rng = random.Random(42)
    vocab = [f"w{i}" for i in range(30)]
    a = [rng.choice(vocab) for _ in range(1500)]
    s = [t for t in a if rng.random() > 0.2]
    b = []
    for t in s:
        if rng.random() < 0.25:
            b.append("<unk>")
        b.append(t)
    got = sol.lcs_length(a, b)
    assert got == len(s), f"期待値 {len(s)}, 実際 {got}"
    p, r, f = sol.rouge_l(a, b)
    _check((p, r, f), (len(s) / len(a), len(s) / len(b), 2 * len(s) / (len(a) + len(b))))
