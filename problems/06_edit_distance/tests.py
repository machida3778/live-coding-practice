VOCAB = ["spelling", "spewing", "sapling", "peeling", "spell"]


def test_public_levenshtein(sol):
    assert sol.levenshtein("kitten", "sitting") == 3
    assert sol.levenshtein("", "abc") == 3
    assert sol.levenshtein("abc", "abc") == 0


def test_public_suggest(sol):
    got = sol.suggest("speling", VOCAB, 3)
    assert got == [("spelling", 1), ("spewing", 1), ("peeling", 2)], got


def test_public_suggest_max_distance(sol):
    got = sol.suggest("speling", VOCAB, 10, max_distance=1)
    assert got == [("spelling", 1), ("spewing", 1)], got


def test_hidden_levenshtein_cases(sol):
    cases = [
        ("", "", 0),
        ("abc", "", 3),
        ("flaw", "lawn", 2),
        ("intention", "execution", 5),
        ("sunday", "saturday", 3),
        ("自然言語処理", "自然言語理解", 2),
        ("a", "b", 1),
        ("ab", "ba", 2),
    ]
    for a, b, d in cases:
        got = sol.levenshtein(a, b)
        assert got == d, f"levenshtein({a!r}, {b!r}) = {got}, expected {d}"
        got = sol.levenshtein(b, a)
        assert got == d, f"levenshtein({b!r}, {a!r}) = {got}, expected {d} (対称性)"


def test_hidden_suggest_duplicates_and_self(sol):
    got = sol.suggest("cat", ["cat", "cat", "bat", "bat", "cut", "dog"], 10)
    assert got == [("cat", 0), ("bat", 1), ("cut", 1), ("dog", 3)], got


def test_hidden_suggest_edge(sol):
    assert sol.suggest("cat", ["bat"], 0) == []
    assert sol.suggest("cat", ["bat"], -1) == []
    assert sol.suggest("cat", [], 3) == []
    assert sol.suggest("cat", ["dog", "elephant"], 5, max_distance=0) == []
    got = sol.suggest("cat", ["cart", "act", "at"], 5, max_distance=2)
    assert got == [("at", 1), ("cart", 1), ("act", 2)], got


def test_hidden_levenshtein_long(sol):
    import random
    assert sol.levenshtein("ab" * 500, "ba" * 500) == 2
    rng = random.Random(0)
    s = "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(600))
    assert sol.levenshtein(s, s[1:] + "x") == 2


def test_hidden_suggest_large_vocab(sol):
    import random
    rng = random.Random(42)
    letters = "bcdfhijkmopqrtvwxyz"  # "language(s)" に含まれる文字を除外 → 遠い単語
    vocab = ["".join(rng.choice(letters) for _ in range(rng.randint(6, 10))) for _ in range(5000)]
    vocab += ["language", "languages", "languages", "lenguage", "langage", "langauge"]
    rng.shuffle(vocab)
    got = sol.suggest("language", vocab, 5)
    exp = [("language", 0), ("langage", 1), ("languages", 1), ("lenguage", 1), ("langauge", 2)]
    assert got == exp, got
