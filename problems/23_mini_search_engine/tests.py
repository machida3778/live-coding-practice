import math
import random
import unicodedata
from collections import Counter

EX_DOCS = ["東京タワーは港区にある電波塔です。", "東京スカイツリーは墨田区にあります。",
           "Tokyo Tower is in Minato, Tokyo.", "大阪城は大阪市にある城です。"]
SNIP_DOC = "東京タワーの営業時間は9時から23時までです。東京タワーは港区にあります。"

CORPUS = [
    "東京タワーは東京都港区にある電波塔です。",
    "Tokyo Tower is a communications tower in Minato, Tokyo.",
    "東京スカイツリーは墨田区にあり、高さは634メートルです。",
    "大阪城は大阪市中央区にある城です。",
    "京都の清水寺は世界遺産に登録されています。",
    "自然言語処理（NLP）はAIの一分野です。",
    "NEXUS Technologyは自然言語処理や画像認識のAIを開発しています。",
    "BM25は検索エンジンで広く使われるランキング関数です。",
    "検索エンジンは転置インデックスを使って高速に文書を検索します。",
    "ＴＯＫＹＯ　ＴＯＷＥＲ　の夜景はきれいです。",
    "",
    "東京 東京 東京 タワー",
]


# ---- テスト側の独立実装 ----
def _is_jp(ch):
    o = ord(ch)
    return o == 0x3005 or 0x3041 <= o <= 0x3096 or 0x30A1 <= o <= 0x30FA or o == 0x30FC or 0x4E00 <= o <= 0x9FFF


def _is_an(ch):
    return "a" <= ch <= "z" or "0" <= ch <= "9"


def _runs(text):
    t = unicodedata.normalize("NFKC", text).lower()
    runs, cur, kind = [], "", None
    for ch in t:
        k = "jp" if _is_jp(ch) else "an" if _is_an(ch) else None
        if k != kind and cur:
            runs.append((kind, cur))
            cur = ""
        kind = k
        if k:
            cur += ch
    if cur:
        runs.append((kind, cur))
    return runs


def _tok(text):
    out = []
    for kind, r in _runs(text):
        if kind == "an" or len(r) == 1:
            out.append(r)
        else:
            out += [r[i:i + 2] for i in range(len(r) - 1)]
    return out


def _bm25(docs, query, k=10, k1=1.5, b=0.75):
    toks = [Counter(_tok(d)) for d in docs]
    lens = [sum(c.values()) for c in toks]
    n = len(docs)
    if n == 0:
        return []
    avgdl = sum(lens) / n
    res = []
    q = set(_tok(query))
    dfs = {t: sum(1 for cc in toks if t in cc) for t in q}
    for i, c in enumerate(toks):
        s, hit = 0.0, False
        for t in q:
            df = dfs[t]
            if t in c:
                hit = True
                idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
                s += idf * c[t] * (k1 + 1) / (c[t] + k1 * (1 - b + b * lens[i] / avgdl))
        if hit:
            res.append((i, s))
    res.sort(key=lambda x: (-x[1], x[0]))
    return res[:k]


def _engine(sol, docs, **kw):
    e = sol.SearchEngine(**kw)
    for d in docs:
        e.add_document(d)
    return e


def _same_results(got, exp):
    assert [i for i, _ in got] == [i for i, _ in exp], (got, exp)
    assert all(math.isclose(a, b, rel_tol=1e-9) for (_, a), (_, b) in zip(got, exp)), (got, exp)


# ================================================================ public
def test_public_part1_normalize(sol):
    assert sol.normalize("ＴＯＫＹＯ　ﾀﾜｰ") == "tokyo タワー", sol.normalize("ＴＯＫＹＯ　ﾀﾜｰ")


def test_public_part1_tokenize_example(sol):
    got = sol.tokenize("東京タワーとTokyo Tower 2024年")
    assert got == ["東京", "京タ", "タワ", "ワー", "ーと", "tokyo", "tower", "2024", "年"], got


def test_public_part2_index_example(sol):
    e = sol.SearchEngine()
    ids = [e.add_document(d) for d in EX_DOCS]
    assert ids == [0, 1, 2, 3], ids
    assert e.postings("東京") == {0: 1, 1: 1}, e.postings("東京")
    assert e.postings("tokyo") == {2: 2}, e.postings("tokyo")
    assert e.doc_length(0) == 15 and e.doc_length(2) == 6, (e.doc_length(0), e.doc_length(2))


def test_public_part3_search_example(sol):
    e = _engine(sol, EX_DOCS)
    got = e.search("東京タワー")
    assert [(i, round(s, 4)) for i, s in got] == [(0, 3.9101), (1, 0.6092)], got
    _same_results(got, _bm25(EX_DOCS, "東京タワー"))
    got = e.search("TOKYO")
    assert [(i, round(s, 4)) for i, s in got] == [(2, 2.0574)], got


def test_public_part4_snippet_example(sol):
    e = _engine(sol, EX_DOCS)
    doc = e.add_document(SNIP_DOC)
    assert e.snippet(doc, "東京タワー 営業時間", width=5) == "【東京タワー】の【営業時間】…", e.snippet(doc, "東京タワー 営業時間", width=5)
    assert e.snippet(doc, "港区", width=3) == "…ワーは【港区】にあり…", e.snippet(doc, "港区", width=3)
    assert e.snippet(doc, "スカイツリー", width=5) == "東京タワーの営業時間…", e.snippet(doc, "スカイツリー", width=5)


# ================================================================ hidden
def test_hidden_part1_tokenize_char_classes(sol):
    cases = {
        "ﾀﾜｰ・ＡＩ　々時々!": ["タワ", "ワー", "ai", "々時", "時々"],
        "a": ["a"],
        "日": ["日"],
        "ＡＢＣ１２３漢字かな": ["abc123", "漢字", "字か", "かな"],
        "東京、大阪。": ["東京", "大阪"],
        "x-ray C++ e-mail": ["x", "ray", "c", "e", "mail"],
        "ヴァイオリンとピアノ": ["ヴァ", "ァイ", "イオ", "オリ", "リン", "ンと", "とピ", "ピア", "アノ"],
    }
    for text, exp in cases.items():
        got = sol.tokenize(text)
        assert got == exp, (text, got, exp)


def test_hidden_part1_tokenize_empty(sol):
    assert sol.tokenize("") == []
    assert sol.tokenize("!!! 。、 ・ ()") == []


def test_hidden_part2_postings_counts(sol):
    e = sol.SearchEngine()
    assert e.add_document("ももももも") == 0
    assert e.add_document("Tokyo tokyo TOKYO ｔｏｋｙｏ") == 1
    assert e.add_document("") == 2
    assert e.postings("もも") == {0: 4}, e.postings("もも")
    assert e.postings("tokyo") == {1: 4}, e.postings("tokyo")
    assert e.postings("存在しない") == {}
    assert [e.doc_length(i) for i in range(3)] == [4, 4, 0]


def test_hidden_part3_bm25_corpus(sol):
    e = _engine(sol, CORPUS)
    for q in ["東京タワー", "tokyo tower", "自然言語処理 AI", "検索エンジン", "大阪", "城", "ＮＬＰ", "東京", "634"]:
        _same_results(e.search(q), _bm25(CORPUS, q))


def test_hidden_part3_bm25_parameters(sol):
    for kw in ({"k1": 1.2, "b": 0.0}, {"k1": 2.0, "b": 1.0}):
        e = _engine(sol, CORPUS, **kw)
        for q in ["東京タワー", "検索 AI"]:
            _same_results(e.search(q), _bm25(CORPUS, q, **kw))


def test_hidden_part3_ties_k_and_duplicate_query_terms(sol):
    docs = ["猫が好き", "犬が好き", "猫が好き", "猫と犬"]
    e = _engine(sol, docs)
    got = e.search("猫が")
    assert [i for i, _ in got] == [0, 2], got
    assert math.isclose(got[0][1], got[1][1]), got
    assert len(e.search("好き", k=1)) == 1 and e.search("好き", k=1)[0][0] == 0
    assert e.search("好き", k=0) == []
    _same_results(e.search("猫 猫 猫"), e.search("猫"))


def test_hidden_part3_empty_cases(sol):
    assert sol.SearchEngine().search("東京") == []
    e = _engine(sol, CORPUS)
    assert e.search("!!!") == []
    assert e.search("zzz") == []
    assert e.search("") == []


def test_hidden_part4_snippet_merge_and_boundaries(sol):
    e = _engine(sol, EX_DOCS + ["aaa bbb ccc"])
    assert e.snippet(0, "東京 タワー", width=3) == "【東京タワー】は港区…", "隣接区間はまとめる"
    assert e.snippet(0, "東京タワー タワー", width=3) == "【東京タワー】は港区…", "重なる区間はまとめる"
    assert e.snippet(4, "aaa ccc", width=5) == "【aaa】 bbb …", e.snippet(4, "aaa ccc", width=5)
    assert e.snippet(4, "aaa bbb ccc", width=5) == "【aaa】 【bbb】 …", e.snippet(4, "aaa bbb ccc", width=5)
    assert e.snippet(4, "aaa cc", width=6) == "【aaa】 bbb c…", "はみ出す区間は囲まない"
    assert e.snippet(4, "ccc", width=2) == "…b 【ccc】", e.snippet(4, "ccc", width=2)


def test_hidden_part4_snippet_case_and_no_match(sol):
    e = _engine(sol, ["Tokyo Tower is tall", "短い文", "ＡＢＣＤＥＦＧＨＩＪＫ"])
    assert e.snippet(0, "TOWER", width=3) == "…yo 【tower】 is…", e.snippet(0, "TOWER", width=3)
    assert e.snippet(1, "東京", width=5) == "短い文"
    assert e.snippet(2, "xyz", width=3) == "abcdef…", e.snippet(2, "xyz", width=3)
    assert e.snippet(2, "ｃｄ", width=1) == "…b【cd】e…", e.snippet(2, "ｃｄ", width=1)


def test_hidden_performance(sol):
    rng = random.Random(23)
    jp = ["自然", "言語", "処理", "検索", "東京", "大阪", "機械", "学習", "文書", "分類", "要約", "翻訳", "対話", "音声"]
    en = ["search", "engine", "model", "data", "tokyo", "neural", "rank", "query", "index", "vector"]
    docs = []
    for _ in range(3000):
        parts = [rng.choice(jp) if rng.random() < 0.6 else " " + rng.choice(en) + " " for _ in range(rng.randint(10, 40))]
        docs.append("".join(parts) + "。")
    e = _engine(sol, docs)
    queries = [rng.choice(jp) + rng.choice(jp) + " " + rng.choice(en) for _ in range(200)]
    results = [e.search(q, k=5) for q in queries]
    for qi in (0, 100, 199):
        _same_results(results[qi], _bm25(docs, queries[qi], k=5))
