CORPUS = {"low": 5, "lower": 2, "newest": 6, "widest": 3}


def _pairs(merges):
    return [tuple(m) for m in merges]


def test_public_train_first3(sol):
    got = _pairs(sol.train_bpe(dict(CORPUS), 3))
    assert got == [("e", "s"), ("es", "t"), ("est", "</w>")], got


def test_public_train5_and_encode(sol):
    merges = _pairs(sol.train_bpe(dict(CORPUS), 5))
    assert merges == [("e", "s"), ("es", "t"), ("est", "</w>"), ("l", "o"), ("lo", "w")], merges
    got = list(sol.encode("lowest", merges))
    assert got == ["low", "est</w>"], got


def test_public_encode_unknown_and_order(sol):
    merges = [("e", "s"), ("es", "t"), ("est", "</w>"), ("l", "o"), ("lo", "w")]
    assert list(sol.encode("xyz", merges)) == ["x", "y", "z", "</w>"]
    got = list(sol.encode("abc", [("b", "c"), ("a", "b")]))
    assert got == ["a", "bc", "</w>"], got


def test_hidden_train10_full(sol):
    got = _pairs(sol.train_bpe(dict(CORPUS), 10))
    exp = [("e", "s"), ("es", "t"), ("est", "</w>"), ("l", "o"), ("lo", "w"), ("e", "w"),
           ("ew", "est</w>"), ("n", "ewest</w>"), ("low", "</w>"), ("d", "est</w>")]
    assert got == exp, f"期待値 {exp}\n実際 {got}"


def test_hidden_encode_training_words(sol):
    merges = [("e", "s"), ("es", "t"), ("est", "</w>"), ("l", "o"), ("lo", "w"), ("e", "w"),
              ("ew", "est</w>"), ("n", "ewest</w>"), ("low", "</w>"), ("d", "est</w>")]
    exp = {"low": ["low</w>"], "lower": ["low", "e", "r", "</w>"],
           "newest": ["newest</w>"], "widest": ["w", "i", "dest</w>"]}
    got = {w: list(sol.encode(w, merges)) for w in exp}
    assert got == exp, f"期待値 {exp}\n実際 {got}"


def test_hidden_early_stop(sol):
    got = _pairs(sol.train_bpe(dict(CORPUS), 100))
    assert len(got) == 15, f"全単語が1シンボルになったら終了 (15回で終わるはず): {len(got)}回"
    assert got[-1] == ("low", "er</w>"), got[-1]
    assert _pairs(sol.train_bpe({"a": 3}, 5)) == [("a", "</w>")]


def test_hidden_zero_merges(sol):
    assert _pairs(sol.train_bpe(dict(CORPUS), 0)) == []
    assert list(sol.encode("abc", [])) == ["a", "b", "c", "</w>"]


def test_hidden_frequency_weighting(sol):
    got = _pairs(sol.train_bpe({"ab": 1, "cd": 5}, 1))
    assert got == [("c", "d")], f"頻度で重み付けして数えてください: {got}"


def test_hidden_overlapping_count(sol):
    # "aaa" の ("a","a") は2回と数える。1回と数えると ("a","</w>") などと同点になってしまう
    got = _pairs(sol.train_bpe({"aaa": 1, "bc": 1}, 1))
    assert got == [("a", "a")], got
    assert list(sol.encode("aaa", [("a", "a")])) == ["aa", "a", "</w>"], "左から順に重ならないようにマージ"


def test_hidden_eow_tie_break(sol):
    got = _pairs(sol.train_bpe({"aaaa": 1}, 10))
    assert got == [("a", "a"), ("aa", "</w>"), ("aa", "aa</w>")], got


def test_hidden_empty_word(sol):
    assert list(sol.encode("", [("a", "b")])) == ["</w>"]


def test_hidden_japanese(sol):
    corpus = {"東京都": 3, "京都府": 2, "東京": 4, "京都": 5}
    merges = _pairs(sol.train_bpe(corpus, 4))
    assert merges == [("京", "都"), ("京都", "</w>"), ("京", "</w>"), ("東", "京</w>")], merges
    assert list(sol.encode("東京都庁", merges)) == ["東", "京都", "庁", "</w>"]


def test_hidden_large(sol):
    import hashlib
    import random
    rng = random.Random(7)
    corpus = {}
    while len(corpus) < 1500:
        w = "".join(rng.choice("abcdefghij") for _ in range(rng.randint(3, 10)))
        corpus[w] = rng.randint(1, 50)
    merges = _pairs(sol.train_bpe(corpus, 150))
    assert len(merges) == 150, len(merges)
    assert merges[:3] == [("f", "</w>"), ("g", "</w>"), ("c", "</w>")], merges[:3]
    assert hashlib.md5(repr(merges).encode()).hexdigest() == "87c81b53141df0bc8396c0b2266b5334", \
        f"マージ列が期待と異なります (末尾3つ: {merges[-3:]}, 期待: [('f', 'f</w>'), ('fi', 'h'), ('a', 'ba')])"
    total = 0
    for w in corpus:
        syms = list(sol.encode(w, merges))
        assert "".join(syms) == w + "</w>", (w, syms)
        total += len(syms)
    assert total == 6005, f"encode 後の総シンボル数: 期待 6005, 実際 {total}"
