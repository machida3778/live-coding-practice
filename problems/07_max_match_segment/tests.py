SUMOMO = ["すもも", "もも", "も", "の", "うち"]
TOKYO = {"東京", "東京都", "京都", "都知事", "知事"}


def test_public_sumomo(sol):
    got = sol.segment("すもももももももものうち", SUMOMO)
    assert got == ["すもも", "もも", "もも", "もも", "の", "うち"], got


def test_public_forward_vs_backward(sol):
    got = sol.segment("東京都知事", TOKYO)
    assert got == ["東京都", "知事"], got
    got = sol.segment_backward("東京都知事", TOKYO)
    assert got == ["東京", "都知事"], got


def test_public_unknown_chars(sol):
    got = sol.segment("私はNLPを学ぶ", {"私", "は", "を", "学ぶ"})
    assert got == ["私", "は", "N", "L", "P", "を", "学ぶ"], got


def test_hidden_empty_text(sol):
    assert sol.segment("", SUMOMO) == []
    assert sol.segment_backward("", SUMOMO) == []


def test_hidden_empty_dictionary(sol):
    assert sol.segment("abc", []) == ["a", "b", "c"]
    assert sol.segment_backward("abc", set()) == ["a", "b", "c"]


def test_hidden_empty_string_in_dictionary(sol):
    assert sol.segment("abc", ["", "ab"]) == ["ab", "c"]
    assert sol.segment_backward("abc", {"", "bc"}) == ["a", "bc"]


def test_hidden_longest_preference(sol):
    d = ["a", "ab", "abc", "cd", "ab"]  # 重複あり
    got = sol.segment("abcd", d)
    assert got == ["abc", "d"], got
    got = sol.segment_backward("abcd", d)
    assert got == ["ab", "cd"], got


def test_hidden_backward_sumomo(sol):
    got = sol.segment_backward("すもももももももものうち", SUMOMO)
    assert got == ["すもも", "もも", "もも", "もも", "の", "うち"], got


def test_hidden_long_word_not_in_text(sol):
    d = {"外国", "外国人", "人参", "参政", "参政権", "政権", "外国人参政権反対運動の歴史"}
    got = sol.segment("外国人参政権", d)
    assert got == ["外国人", "参政権"], got
    got = sol.segment("外国人参", d)
    assert got == ["外国人", "参"], got
    got = sol.segment("外人参政権", d)
    assert got == ["外", "人参", "政権"], got
    got = sol.segment_backward("外人参政権", d)
    assert got == ["外", "人", "参政権"], got


def _ref_forward(text, words, max_len):
    out, i = [], 0
    while i < len(text):
        for l in range(min(max_len, len(text) - i), 0, -1):
            if text[i:i + l] in words:
                break
        else:
            l = 1
        out.append(text[i:i + l])
        i += l
    return out


def _ref_backward(text, words, max_len):
    out, j = [], len(text)
    while j > 0:
        for l in range(min(max_len, j), 0, -1):
            if text[j - l:j] in words:
                break
        else:
            l = 1
        out.append(text[j - l:j])
        j -= l
    return out[::-1]


def test_hidden_large(sol):
    import random
    rng = random.Random(7)
    chars = "あいうえおかきくけこ"
    words = {"".join(rng.choice(chars) for _ in range(rng.randint(2, 8))) for _ in range(3000)}
    text = "".join(rng.choice(chars) for _ in range(100000))
    max_len = max(map(len, words))
    got = sol.segment(text, list(words))
    assert got == _ref_forward(text, words, max_len), "前向き最長一致の結果が一致しません"
    got = sol.segment_backward(text, words)
    assert got == _ref_backward(text, words, max_len), "後ろ向き最長一致の結果が一致しません"
