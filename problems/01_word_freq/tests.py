def test_public_example1(sol):
    got = sol.top_k_words("The cat and the hat. The CAT!", 2)
    assert got == [("the", 3), ("cat", 2)], got


def test_public_tie_break(sol):
    got = sol.top_k_words("b a c a b", 3)
    assert got == [("a", 2), ("b", 2), ("c", 1)], got


def test_public_stopwords(sol):
    got = sol.top_k_words("The cat and the hat", 2, stopwords={"the", "and"})
    assert got == [("cat", 1), ("hat", 1)], got


def test_hidden_apostrophe_digits(sol):
    got = sol.top_k_words("don't stop 2023 don't", 10)
    assert got == [("don", 2), ("t", 2), ("stop", 1)], got


def test_hidden_k_larger_than_vocab(sol):
    assert sol.top_k_words("one two", 5) == [("one", 1), ("two", 1)]


def test_hidden_k_zero_and_negative(sol):
    assert sol.top_k_words("a a b", 0) == []
    assert sol.top_k_words("a a b", -1) == []


def test_hidden_empty(sol):
    assert sol.top_k_words("", 3) == []
    assert sol.top_k_words("123 !!! ...", 3) == []


def test_hidden_large(sol):
    import random
    rng = random.Random(0)
    words = [f"w{chr(97 + i % 26)}{chr(97 + i // 26 % 26)}" for i in range(500)]
    words = ["".join(c for c in w if c.isalpha()) for w in words]
    text = " ".join(rng.choice(words) for _ in range(300000))
    got = sol.top_k_words(text, 5)
    from collections import Counter
    exp = sorted(Counter(text.split()).items(), key=lambda x: (-x[1], x[0]))[:5]
    assert got == exp, (got, exp)
