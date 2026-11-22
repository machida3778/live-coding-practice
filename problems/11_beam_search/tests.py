import math


def _lm(table, eos="</s>"):
    """確率の表から next_token_logprobs を作る。eos で終わった仮説を展開したらエラー"""
    def fn(prefix):
        assert isinstance(prefix, tuple), f"prefix はタプルで渡してください: {prefix!r}"
        assert not prefix or prefix[-1] != eos, f"eos で終わった仮説を展開しています: {prefix}"
        return {t: math.log(p) for t, p in table[prefix].items()}
    return fn


def _check(got, exp_seq, exp_score):
    seq, score = got
    assert tuple(seq) == tuple(exp_seq), f"系列: 期待 {exp_seq}, 実際 {tuple(seq)}"
    assert math.isclose(score, exp_score, rel_tol=1e-9, abs_tol=1e-9), f"スコア: 期待 {exp_score}, 実際 {score}"


P = {
    (): {"A": 0.6, "B": 0.4},
    ("A",): {"x": 0.4, "y": 0.35, "</s>": 0.25},
    ("B",): {"w": 0.9, "</s>": 0.1},
    ("A", "x"): {"</s>": 1.0},
    ("A", "y"): {"</s>": 1.0},
    ("B", "w"): {"</s>": 1.0},
}


def test_public_greedy(sol):
    _check(sol.beam_search(_lm(P), 1, 10), ("A", "x", "</s>"), math.log(0.6) + math.log(0.4) + math.log(1.0))


def test_public_beam2_beats_greedy(sol):
    _check(sol.beam_search(_lm(P), 2, 10), ("B", "w", "</s>"), math.log(0.4) + math.log(0.9) + math.log(1.0))


def test_public_max_len(sol):
    always = lambda prefix: {"a": math.log(0.9), "</s>": math.log(0.1)}
    _check(sol.beam_search(always, 2, 3), ("a", "a", "a"), 3 * math.log(0.9))


def test_hidden_wide_beam(sol):
    _check(sol.beam_search(_lm(P), 10, 10), ("B", "w", "</s>"), math.log(0.4) + math.log(0.9))


def test_hidden_max_len_one(sol):
    always = lambda prefix: {"a": math.log(0.9), "</s>": math.log(0.1)}
    _check(sol.beam_search(always, 1, 1), ("a",), math.log(0.9))
    _check(sol.beam_search(always, 3, 1), ("a",), math.log(0.9))


LN = {
    (): {"</s>": 0.4, "a": 0.6},
    ("a",): {"</s>": 0.3, "b": 0.7},
    ("a", "b"): {"</s>": 0.8, "c": 0.2},
    ("a", "b", "c"): {"</s>": 1.0},
}


def test_hidden_length_normalization(sol):
    _check(sol.beam_search(_lm(LN), 5, 10, alpha=0.0), ("</s>",), math.log(0.4))
    lp = math.log(0.6) + math.log(0.7) + math.log(0.8)
    _check(sol.beam_search(_lm(LN), 5, 10, alpha=1.0), ("a", "b", "</s>"), lp / 3)
    _check(sol.beam_search(_lm(LN), 5, 10, alpha=0.5), ("a", "b", "</s>"), lp / 3 ** 0.5)


def test_hidden_finished_consumes_slot(sol):
    # ステップ1の上位2つは "</s>"(完了) と "a" → "b" はここで枠から落ちる
    SL = {
        (): {"</s>": 0.5, "a": 0.3, "b": 0.2},
        ("a",): {"</s>": 0.4, "c": 0.6},
        ("b",): {"d": 1.0},
        ("a", "c"): {"</s>": 1.0},
        ("b", "d"): {"</s>": 1.0},
    }
    lp = math.log(0.3) + math.log(0.6)
    _check(sol.beam_search(_lm(SL), 2, 10, alpha=1.0), ("a", "c", "</s>"), lp / 3)
    _check(sol.beam_search(_lm(SL), 3, 10, alpha=1.0), ("b", "d", "</s>"), math.log(0.2) / 3)


def test_hidden_tie_break(sol):
    TIE = {(): {"b": 0.5, "a": 0.5}, ("a",): {"</s>": 1.0}, ("b",): {"</s>": 1.0}}
    _check(sol.beam_search(_lm(TIE), 1, 5), ("a", "</s>"), math.log(0.5))
    _check(sol.beam_search(_lm(TIE), 2, 5), ("a", "</s>"), math.log(0.5))


def test_hidden_custom_eos(sol):
    table = {
        (): {"x": 0.7, "<eos>": 0.3},
        ("x",): {"<eos>": 0.9, "y": 0.1},
        ("x", "y"): {"<eos>": 1.0},
    }
    _check(sol.beam_search(_lm(table, eos="<eos>"), 3, 10, eos="<eos>"),
           ("x", "<eos>"), math.log(0.7) + math.log(0.9))


def test_hidden_large(sol):
    import random
    import zlib

    def make_lm(vocab_size=300, eos="</s>"):
        vocab = [f"t{i}" for i in range(vocab_size)] + [eos]

        def lm(prefix):
            assert isinstance(prefix, tuple), f"prefix はタプルで渡してください: {prefix!r}"
            assert not prefix or prefix[-1] != eos, f"eos で終わった仮説を展開しています: {prefix}"
            rng = random.Random(zlib.crc32(("|".join(prefix) + "#0").encode()))
            logits = [rng.gauss(0, 2) for _ in vocab]
            logits[-1] += 2.0
            m = max(logits)
            z = math.log(sum(math.exp(x - m) for x in logits)) + m
            return {t: x - z for t, x in zip(vocab, logits)}
        return lm

    _check(sol.beam_search(make_lm(), 25, 40, alpha=0.7),
           ("t98", "t111", "t60", "</s>"), -2.880828674983835)
    _check(sol.beam_search(make_lm(), 25, 40, alpha=0.0), ("</s>",), -3.890450215624171)
