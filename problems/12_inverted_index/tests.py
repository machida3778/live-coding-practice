def _example(sol):
    idx = sol.InvertedIndex()
    idx.add(1, "New York is a big city.")
    idx.add(2, "I love new books. York is old.")
    idx.add(3, "The city of York")
    return idx


def _eq(got, exp):
    assert list(got) == exp, f"期待 {exp}, 実際 {got}"


def test_public_and_or(sol):
    idx = _example(sol)
    _eq(idx.search_and(["york", "new"]), [1, 2])
    _eq(idx.search_or(["books", "CITY"]), [1, 2, 3])


def test_public_and_not_phrase(sol):
    idx = _example(sol)
    _eq(idx.search_and_not(["york"], ["new"]), [3])
    _eq(idx.search_phrase("new york"), [1])


def test_public_tokenize(sol):
    idx = sol.InvertedIndex()
    idx.add(5, "Hello, World! 2024")
    _eq(idx.search_and(["hello", "world", "2024"]), [5])
    _eq(idx.search_and(["tokyo"]), [])


def test_hidden_empty_queries(sol):
    idx = _example(sol)
    _eq(idx.search_and([]), [])
    _eq(idx.search_or([]), [])
    _eq(idx.search_and_not([], ["new"]), [])
    _eq(idx.search_phrase(""), [])
    _eq(idx.search_phrase("!!! ..."), [])


def test_hidden_missing_terms(sol):
    idx = _example(sol)
    _eq(idx.search_and(["york", "tokyo"]), [])
    _eq(idx.search_or(["tokyo", "books"]), [2])
    _eq(idx.search_and_not(["york"], ["tokyo"]), [1, 2, 3])
    _eq(idx.search_phrase("new tokyo"), [])


def test_hidden_case_insensitive_query(sol):
    idx = _example(sol)
    _eq(idx.search_and(["YORK", "New"]), [1, 2])
    _eq(idx.search_and_not(["York"], ["NEW", "CITY"]), [])
    _eq(idx.search_phrase("NEW York"), [1])


def test_hidden_sorted_unique(sol):
    idx = sol.InvertedIndex()
    idx.add(10, "apple apple banana")
    idx.add(3, "banana apple")
    idx.add(7, "cherry apple apple apple")
    _eq(idx.search_and(["apple"]), [3, 7, 10])
    _eq(idx.search_or(["apple", "banana", "cherry"]), [3, 7, 10])
    _eq(idx.search_phrase("apple apple"), [7, 10])


def test_hidden_phrase_order_and_adjacency(sol):
    idx = sol.InvertedIndex()
    idx.add(1, "a a b")
    idx.add(2, "a b a")
    idx.add(3, "b x a")
    idx.add(4, "New, York!")
    _eq(idx.search_phrase("a b"), [1, 2])
    _eq(idx.search_phrase("a a"), [1])
    _eq(idx.search_phrase("b a"), [2])
    _eq(idx.search_phrase("a a b"), [1])
    _eq(idx.search_phrase("b a b"), [])
    _eq(idx.search_phrase("x"), [3])
    _eq(idx.search_phrase("new-york"), [4])


def test_hidden_large(sol):
    import random
    rng = random.Random(1)
    V = 2000
    vocab = [f"w{i}" for i in range(V)]
    cum, acc = [], 0.0
    for i in range(V):
        acc += 1 / (i + 1)
        cum.append(acc)
    docs = [rng.choices(vocab, cum_weights=cum, k=30) for _ in range(20000)]
    idx = sol.InvertedIndex()
    for d, toks in enumerate(docs):
        text = " ".join(toks)
        idx.add(d * 3 + 1, text.upper() if d % 2 else text)

    qrng = random.Random(2)
    and_qs = [qrng.sample(vocab[:300], 2) for _ in range(300)]
    phr_qs = []
    for _ in range(100):
        toks = docs[qrng.randrange(len(docs))]
        i = qrng.randrange(len(toks) - 1)
        phr_qs.append(f"{toks[i]} {toks[i + 1]}")
    res_and = [list(idx.search_and(q)) for q in and_qs]
    res_or = [list(idx.search_or(q)) for q in and_qs[:100]]
    res_phr = [list(idx.search_phrase(q)) for q in phr_qs]

    # 一部のクエリだけ全走査で答え合わせ
    sets = [set(t) for t in docs]
    for q, got in list(zip(and_qs, res_and))[:20]:
        exp = [d * 3 + 1 for d, s in enumerate(sets) if all(t in s for t in q)]
        assert got == exp, f"AND {q}: 件数 期待 {len(exp)}, 実際 {len(got)}"
    for q, got in list(zip(and_qs, res_or))[:10]:
        exp = [d * 3 + 1 for d, s in enumerate(sets) if any(t in s for t in q)]
        assert got == exp, f"OR {q}: 件数 期待 {len(exp)}, 実際 {len(got)}"
    joined = [" " + " ".join(t) + " " for t in docs]
    for q, got in list(zip(phr_qs, res_phr))[:20]:
        exp = [d * 3 + 1 for d, j in enumerate(joined) if f" {q} " in j]
        assert got == exp, f"フレーズ '{q}': 件数 期待 {len(exp)}, 実際 {len(got)}"
