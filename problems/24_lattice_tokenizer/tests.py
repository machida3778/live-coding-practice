import os
import random
import tempfile
from pathlib import Path

HERE = Path(__file__).parent
DICT = HERE / "dict.csv"
MATRIX = HERE / "matrix.txt"
UNK = "未知語"


# ---- テスト側の独立実装 (解答の load_* に依存しない) ----
def _D():
    d = {}
    for line in DICT.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            s, p, c = [x.strip() for x in line.split(",")]
            d.setdefault(s, []).append((p, int(c)))
    return d


def _C():
    c = {}
    for line in MATRIX.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            l, r, v = line.split()
            c[(l, r)] = int(v)
    return c


def _maxlen(D):
    return max((len(s) for s in D), default=0)


def _nodes(D, text, i, unk, maxlen):
    out = [(text[i:j], p, c) for j in range(i + 1, min(len(text), i + maxlen) + 1) for p, c in D.get(text[i:j], [])]
    return out or [(text[i], UNK, unk)]


def _dp_best(D, C, text, unk=10000):
    """状態 = (位置, 直前の品詞) の DP で最小コストを求める"""
    n, maxlen = len(text), _maxlen(D)
    dp = [dict() for _ in range(n + 1)]
    dp[0]["BOS"] = 0
    for i in range(n):
        if not dp[i]:
            continue
        for s, pos, c in _nodes(D, text, i, unk, maxlen):
            v = min(val + C.get((p, pos), 0) for p, val in dp[i].items()) + c
            j = i + len(s)
            if v < dp[j].get(pos, float("inf")):
                dp[j][pos] = v
    return min(val + C.get((p, "EOS"), 0) for p, val in dp[n].items())


def _path_cost(D, C, text, path, unk=10000):
    assert "".join(s for s, _ in path) == text, f"表層を連結しても元の文になりません: {path}"
    cost, prev, i, maxlen = 0, "BOS", 0, _maxlen(D)
    for s, pos in path:
        cands = [c for ss, pp, c in _nodes(D, text, i, unk, maxlen) if ss == s and pp == pos]
        assert cands, f"位置 {i} にノード {s}/{pos} はあり得ません"
        cost += C.get((prev, pos), 0) + min(cands)
        prev, i = pos, i + len(s)
    return cost + C.get((prev, "EOS"), 0)


EXPECTED = {
    "すもももももももものうち": [("すもも", "名詞"), ("も", "助詞"), ("もも", "名詞"), ("も", "助詞"), ("もも", "名詞"),
                       ("の", "助詞"), ("うち", "名詞")],
    "外国人参政権": [("外国", "名詞"), ("人", "接尾辞"), ("参政権", "名詞")],
    "東京都に住む": [("東京都", "名詞"), ("に", "助詞"), ("住む", "動詞")],
    "京都に行った": [("京都", "名詞"), ("に", "助詞"), ("行っ", "動詞"), ("た", "助動詞")],
    "今日はいい天気です。": [("今日", "名詞"), ("は", "助詞"), ("いい", "形容詞"), ("天気", "名詞"), ("です", "助動詞"),
                   ("。", "記号")],
    "私は学校に行く": [("私", "名詞"), ("は", "助詞"), ("学校", "名詞"), ("に", "助詞"), ("行く", "動詞")],
    "東京都の人参": [("東京都", "名詞"), ("の", "助詞"), ("人参", "名詞")],
}


# ================================================================ public
def test_public_part1_load_dictionary(sol):
    D = sol.load_dictionary(str(DICT))
    assert D["も"] == [("助詞", 300), ("名詞", 2500)], D.get("も")
    assert D["すもも"] == [("名詞", 1000)], D.get("すもも")
    assert D == _D(), "辞書の内容が期待と一致しません (コメント行・空行・型を確認)"


def test_public_part1_load_connection(sol):
    C = sol.load_connection(str(MATRIX))
    assert C[("BOS", "名詞")] == 0 and C[("名詞", "助詞")] == -300, (C.get(("BOS", "名詞")), C.get(("名詞", "助詞")))
    assert C == _C(), "連接コスト表が期待と一致しません (タブ区切りの行・コメント行を確認)"


def test_public_part2_lookup(sol):
    tok = sol.LatticeTokenizer(_D(), _C())
    assert tok.lookup("すもももも", 0) == [("す", "名詞", 2000), ("すもも", "名詞", 1000)], tok.lookup("すもももも", 0)
    assert tok.lookup("パクシャ", 0) == [("パ", UNK, 10000)], tok.lookup("パクシャ", 0)


def test_public_part3_tokenize_sumomo(sol):
    tok = sol.LatticeTokenizer(_D(), _C())
    text = "すもももももももものうち"
    assert tok.tokenize(text) == EXPECTED[text], tok.tokenize(text)


def test_public_part3_best_cost_sumomo(sol):
    tok = sol.LatticeTokenizer(_D(), _C())
    assert tok.best_cost("すもももももももものうち") == 3600, tok.best_cost("すもももももももものうち")


# ================================================================ hidden
def test_hidden_part1_whitespace_negative_and_comments(sol):
    with tempfile.TemporaryDirectory() as td:
        dp, mp = os.path.join(td, "d.csv"), os.path.join(td, "m.txt")
        with open(dp, "w", encoding="utf-8") as f:
            f.write("# comment\n\n 外国 , 名詞 , -100 \n外国,接尾辞,5\n  \n#外国,名詞,1\nあ,感動詞,0\n")
        with open(mp, "w", encoding="utf-8") as f:
            f.write("#x\n名詞 \t 助詞   -7\n\n  BOS  名詞 3  \n")
        assert sol.load_dictionary(dp) == {"外国": [("名詞", -100), ("接尾辞", 5)], "あ": [("感動詞", 0)]}, \
            sol.load_dictionary(dp)
        assert sol.load_connection(mp) == {("名詞", "助詞"): -7, ("BOS", "名詞"): 3}, sol.load_connection(mp)


def test_hidden_part2_lookup_order_and_positions(sol):
    tok = sol.LatticeTokenizer(_D(), _C())
    assert tok.lookup("ももの", 0) == [("も", "助詞", 300), ("も", "名詞", 2500), ("もも", "名詞", 1000)]
    assert tok.lookup("すもも", 2) == [("も", "助詞", 300), ("も", "名詞", 2500)]
    assert tok.lookup("外国人参政権", 3) == [("参政", "名詞", 1800), ("参政権", "名詞", 2000)]
    assert tok.lookup("パクシャ", 3) == [("ャ", UNK, 10000)]
    tok2 = sol.LatticeTokenizer(_D(), _C(), unk_cost=123)
    assert tok2.lookup("xもも", 0) == [("x", UNK, 123)]


def test_hidden_part3_fixed_sentences(sol):
    tok = sol.LatticeTokenizer(_D(), _C())
    D, C = _D(), _C()
    for text, exp in EXPECTED.items():
        got = tok.tokenize(text)
        assert got == exp, (text, got)
        assert tok.best_cost(text) == _dp_best(D, C, text), text


def test_hidden_part3_unknown_words(sol):
    tok = sol.LatticeTokenizer(_D(), _C())
    text = "パクシャは自然言語処理"
    exp = [("パ", UNK), ("ク", UNK), ("シ", UNK), ("ャ", UNK), ("は", "助詞"), ("自然言語", "名詞"), ("処理", "名詞")]
    assert tok.tokenize(text) == exp, tok.tokenize(text)
    assert tok.best_cost(text) == 42700, tok.best_cost(text)
    tok1 = sol.LatticeTokenizer(_D(), _C(), unk_cost=1)
    assert tok1.best_cost("パクシャ") == 500 + 4 * 1 + 3 * (-300) + 0, tok1.best_cost("パクシャ")


def test_hidden_part3_empty_text(sol):
    tok = sol.LatticeTokenizer(_D(), _C())
    assert tok.tokenize("") == []
    assert tok.best_cost("") == 0
    C = _C()
    C[("BOS", "EOS")] = 7
    assert sol.LatticeTokenizer(_D(), C).best_cost("") == 7


def test_hidden_part3_missing_connection_is_zero_and_negative_costs(sol):
    D = {"a": [("X", 1)], "b": [("Y", 1)], "ab": [("Z", 5)]}
    tok = sol.LatticeTokenizer(D, {("X", "Y"): 10})
    assert tok.tokenize("ab") == [("ab", "Z")] and tok.best_cost("ab") == 5
    tok = sol.LatticeTokenizer(D, {("X", "Y"): -10})
    assert tok.tokenize("ab") == [("a", "X"), ("b", "Y")] and tok.best_cost("ab") == -8
    tok = sol.LatticeTokenizer(D, {("BOS", "Z"): 100, ("X", "Y"): 0})
    assert tok.tokenize("abab") == [("a", "X"), ("b", "Y"), ("a", "X"), ("b", "Y")], tok.tokenize("abab")


def test_hidden_part3_random_consistency(sol):
    D, C = _D(), _C()
    tok = sol.LatticeTokenizer(D, C)
    rng = random.Random(24)
    pieces = list(D) + ["パ", "x", "っ"]
    for _ in range(40):
        text = "".join(rng.choice(pieces) for _ in range(rng.randint(1, 8)))
        best = _dp_best(D, C, text)
        assert tok.best_cost(text) == best, (text, tok.best_cost(text), best)
        assert _path_cost(D, C, text, tok.tokenize(text)) == best, (text, tok.tokenize(text))


def test_hidden_performance(sol):
    rng = random.Random(2024)
    D, C = _D(), _C()
    pool = "あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをん言語処理自然学習機械"
    for _ in range(50000):
        s = "".join(rng.choice(pool) for _ in range(rng.randint(2, 7)))
        D.setdefault(s, []).append((rng.choice(["名詞", "動詞", "助詞", "形容詞"]), rng.randint(500, 3000)))
    surfaces = list(D)
    parts, n = [], 0
    while n < 4000:
        w = rng.choice(surfaces) if rng.random() < 0.95 else rng.choice("アイウエオ")
        parts.append(w)
        n += len(w)
    text = "".join(parts)
    tok = sol.LatticeTokenizer(D, C)
    path = tok.tokenize(text)
    best = _dp_best(D, C, text)
    assert _path_cost(D, C, text, path) == best, "最小コストの分割になっていません"
    assert tok.best_cost(text) == best
