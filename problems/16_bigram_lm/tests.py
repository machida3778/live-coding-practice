import itertools
import math
import random

TOL = 1e-9
TRAIN = [["I", "am", "Sam"],
         ["Sam", "I", "am"],
         ["I", "do", "not", "like", "green", "eggs", "and", "ham"]]
VOCAB = sorted({w for s in TRAIN for w in s} | {"</s>", "<unk>"})  # |V| = 12


def _close(a, b):
    return math.isclose(a, b, rel_tol=TOL, abs_tol=1e-12)


def test_public_prob(sol):
    lm = sol.BigramLM(k=1.0).fit(TRAIN)
    cases = [("I", "<s>", 3 / 15), ("am", "I", 3 / 15), ("Sam", "am", 2 / 14), ("</s>", "Sam", 2 / 14),
             ("do", "I", 2 / 15), ("ham", "I", 1 / 15)]
    for w, prev, exp in cases:
        got = lm.prob(w, prev)
        assert _close(got, exp), f"P({w} | {prev}) = {got}, expected {exp}"


def test_public_sentence_logprob_and_perplexity(sol):
    lm = sol.BigramLM(k=1.0).fit(TRAIN)
    exp = 2 * math.log(0.2) + 2 * math.log(1 / 7)
    got = lm.sentence_logprob(["I", "am", "Sam"])
    assert _close(got, exp), f"logprob = {got}, expected {exp}"
    got = lm.perplexity([["I", "am", "Sam"]])
    assert _close(got, math.exp(-exp / 4)), f"perplexity = {got}, expected {math.exp(-exp / 4)}"


def test_public_unknown_word(sol):
    lm = sol.BigramLM(k=1.0).fit(TRAIN)
    assert _close(lm.prob("cat", "I"), 1 / 15), f"未知語は <unk>: {lm.prob('cat', 'I')}"
    assert _close(lm.prob("cat", "I"), lm.prob("<unk>", "I"))


def test_hidden_fit_returns_self(sol):
    lm = sol.BigramLM()
    assert lm.fit(TRAIN) is lm, "fit は self を返す"


def test_hidden_distribution_sums_to_one(sol):
    lm = sol.BigramLM(k=0.5).fit(TRAIN)
    for prev in ["<s>", "I", "am", "Sam", "ham", "unknown_prev"]:
        total = math.fsum(lm.prob(w, prev) for w in VOCAB)
        assert _close(total, 1.0), f"Σ_w P(w | {prev}) = {total} (V は学習語彙 + </s> + <unk>、<s> は含まない)"


def test_hidden_unknown_context(sol):
    lm = sol.BigramLM(k=1.0).fit(TRAIN)
    # 未知の prev は <unk> (C(<unk>) = 0) なので一様分布 1/|V|
    assert _close(lm.prob("am", "cat"), 1 / 12), f"P(am | 未知語) = {lm.prob('am', 'cat')}"
    assert _close(lm.prob("dog", "cat"), 1 / 12)
    exp = math.log(3 / 15) + math.log(1 / 15) + math.log(1 / 12) + math.log(1 / 12)
    got = lm.sentence_logprob(["I", "cat", "dog"])
    assert _close(got, exp), f"logprob(I cat dog) = {got}, expected {exp}"


def test_hidden_empty_sentence(sol):
    lm = sol.BigramLM(k=1.0).fit(TRAIN)
    exp = math.log(1 / 15)  # P(</s> | <s>) = (0 + 1) / (3 + 12)
    assert _close(lm.sentence_logprob([]), exp), f"空文: {lm.sentence_logprob([])}"
    assert _close(lm.perplexity([[]]), 15.0), "空文1つ: M = 1"


def test_hidden_perplexity_multiple_and_k(sol):
    lm = sol.BigramLM(k=0.1).fit(TRAIN)
    V = 12

    def p(c_bigram, c_prev):
        return (c_bigram + 0.1) / (c_prev + 0.1 * V)

    s1 = ["Sam", "I", "am"]          # <s>→Sam(1/3), Sam→I(1/2), I→am(2/3), am→</s>(1/2)
    s2 = ["I", "like", "ham"]        # <s>→I(2/3), I→like(0/3), like→ham(0/1), ham→</s>(1/1)
    lp1 = math.log(p(1, 3)) + math.log(p(1, 2)) + math.log(p(2, 3)) + math.log(p(1, 2))
    lp2 = math.log(p(2, 3)) + math.log(p(0, 3)) + math.log(p(0, 1)) + math.log(p(1, 1))
    assert _close(lm.sentence_logprob(s1), lp1), f"{lm.sentence_logprob(s1)} vs {lp1}"
    assert _close(lm.sentence_logprob(s2), lp2), f"{lm.sentence_logprob(s2)} vs {lp2}"
    exp = math.exp(-(lp1 + lp2) / 8)
    got = lm.perplexity([s1, s2])
    assert _close(got, exp), f"perplexity = {got}, expected {exp} (M は </s> を含むトークン数)"


def test_hidden_large(sol):
    rng = random.Random(5)
    vocab = [f"w{i}" for i in range(3000)]
    cum = list(itertools.accumulate(1 / (r + 1) for r in range(len(vocab))))  # Zipf 風
    train = [rng.choices(vocab, cum_weights=cum, k=rng.randint(0, 25)) for _ in range(20000)]
    test = [rng.choices(vocab, cum_weights=cum, k=rng.randint(1, 25)) + ["oov_word"] for _ in range(3000)]
    lm = sol.BigramLM(k=0.01).fit(train)

    # 参照値 (独立実装)
    big, ctx = {}, {}
    V = len({w for s in train for w in s}) + 2
    seen = {w for s in train for w in s}
    for s in train:
        toks = ["<s>"] + s + ["</s>"]
        for a, b in zip(toks, toks[1:]):
            big[(a, b)] = big.get((a, b), 0) + 1
            ctx[a] = ctx.get(a, 0) + 1
    total, m = 0.0, 0
    for s in test:
        toks = ["<s>"] + [w if w in seen else "<unk>" for w in s] + ["</s>"]
        for a, b in zip(toks, toks[1:]):
            total += math.log((big.get((a, b), 0) + 0.01) / (ctx.get(a, 0) + 0.01 * V))
        m += len(s) + 1
    exp = math.exp(-total / m)
    got = lm.perplexity(test)
    assert math.isclose(got, exp, rel_tol=1e-7), f"perplexity = {got}, expected {exp}"
