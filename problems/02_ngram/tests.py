from collections import Counter


def test_public_word_ngrams(sol):
    got = sol.word_ngrams(["I", "love", "NLP"], 2)
    assert got == [("I", "love"), ("love", "NLP")], got


def test_public_char_ngrams(sol):
    got = sol.char_ngrams("自然言語", 2)
    assert got == ["自然", "然言", "言語"], got
    got = sol.char_ngrams("abc", 2, pad=True)
    assert got == ["^a", "ab", "bc", "c$"], got


def test_public_ngram_counts(sol):
    got = sol.ngram_counts([["a", "b", "a", "b"], ["b", "a"]], 2)
    assert dict(got) == {("a", "b"): 2, ("b", "a"): 2}, got


def test_hidden_unigram_trigram_tuple(sol):
    got = sol.word_ngrams(["a", "b", "c"], 1)
    assert got == [("a",), ("b",), ("c",)], got
    got = sol.word_ngrams(["a", "b", "c"], 3)
    assert got == [("a", "b", "c")], got
    assert all(isinstance(g, tuple) for g in got), "n-gram はタプルで返してください"


def test_hidden_too_short(sol):
    assert sol.word_ngrams(["a", "b"], 3) == []
    assert sol.word_ngrams([], 1) == []
    assert sol.char_ngrams("ab", 3) == []
    assert sol.char_ngrams("", 1) == []


def test_hidden_padding(sol):
    got = sol.char_ngrams("ab", 3, pad=True)
    assert got == ["^^a", "^ab", "ab$", "b$$"], got
    got = sol.char_ngrams("ab", 1, pad=True)
    assert got == ["a", "b"], f"n=1 ではパディングは0個: {got}"
    got = sol.char_ngrams("", 2, pad=True)
    assert got == ["^$"], got


def test_hidden_invalid_n(sol):
    calls = [
        lambda: sol.word_ngrams(["a"], 0),
        lambda: sol.char_ngrams("abc", 0),
        lambda: sol.char_ngrams("abc", -1, pad=True),
        lambda: sol.ngram_counts([["a"]], 0),
    ]
    for call in calls:
        try:
            call()
        except ValueError:
            continue
        raise AssertionError("n < 1 のとき ValueError を送出してください")


def test_hidden_counts_no_cross_and_min_count(sol):
    sents = [["the", "cat", "sat"], ["the", "cat"], ["cat"], []]
    got = sol.ngram_counts(sents, 2)
    assert dict(got) == {("the", "cat"): 2, ("cat", "sat"): 1}, got
    got = sol.ngram_counts(sents, 2, min_count=2)
    assert dict(got) == {("the", "cat"): 2}, got
    got = sol.ngram_counts(sents, 1, min_count=3)
    assert dict(got) == {("cat",): 3}, got


def test_hidden_large(sol):
    import random
    rng = random.Random(1)
    vocab = [f"w{i}" for i in range(50)]
    sents = [[rng.choice(vocab) for _ in range(rng.randint(0, 30))] for _ in range(20000)]
    got = dict(sol.ngram_counts(sents, 3, min_count=2))
    exp = Counter()
    for s in sents:
        for i in range(len(s) - 2):
            exp[tuple(s[i:i + 3])] += 1
    exp = {g: c for g, c in exp.items() if c >= 2}
    assert got == exp, f"件数 got={len(got)} expected={len(exp)}"

    text = "".join(rng.choice("あいうえお") for _ in range(300000))
    got = sol.char_ngrams(text, 3, pad=True)
    assert len(got) == 300002, len(got)
    assert got[0] == "^^" + text[0] and got[-1] == text[-1] + "$$"
