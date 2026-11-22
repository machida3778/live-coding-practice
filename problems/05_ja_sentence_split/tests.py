def _check(sol, src, exp):
    got = sol.split_sentences(src)
    assert got == exp, f"入力 {src!r}\n期待 {exp!r}\n実際 {got!r}"


def test_public_basic(sol):
    _check(sol, "今日は晴れ。明日は雨！本当？", ["今日は晴れ。", "明日は雨！", "本当？"])


def test_public_brackets(sol):
    _check(sol, "彼は「はい。わかりました。」と言った。次へ。",
           ["彼は「はい。わかりました。」と言った。", "次へ。"])


def test_public_consecutive_terminals(sol):
    _check(sol, "えっ！？嘘でしょ!!", ["えっ！？", "嘘でしょ!!"])


def test_public_newline(sol):
    _check(sol, "一行目\n二行目。三文目", ["一行目", "二行目。", "三文目"])


def test_hidden_nested_brackets(sol):
    _check(sol, "「彼女は『無理。』と言った。」と聞いた。本当だ。",
           ["「彼女は『無理。』と言った。」と聞いた。", "本当だ。"])


def test_hidden_parentheses(sol):
    _check(sol, "結果は良好（詳細は後述。）である。(注: 重要!)と書く。",
           ["結果は良好（詳細は後述。）である。", "(注: 重要!)と書く。"])


def test_hidden_strip_and_empty_lines(sol):
    _check(sol, "  　前置き。  \n\n\t後書き。　", ["前置き。", "後書き。"])
    _check(sol, "A。 B。", ["A。", "B。"])


def test_hidden_empty(sol):
    _check(sol, "", [])
    _check(sol, "\n\n  \n　", [])


def test_hidden_terminal_runs(sol):
    _check(sol, "。。。", ["。。。"])
    _check(sol, "まじ！？！？それな。", ["まじ！？！？", "それな。"])
    _check(sol, "え。。。そう？", ["え。。。", "そう？"])
    _check(sol, "Really?Yes!OK", ["Really?", "Yes!", "OK"])


def test_hidden_extra_closing_bracket(sol):
    _check(sol, "」終わり。始まり。", ["」終わり。", "始まり。"])
    _check(sol, "a)）b。c。", ["a)）b。", "c。"])


def test_hidden_unclosed_bracket(sol):
    _check(sol, "「未完。の文\n次の行。続き", ["「未完。の文", "次の行。", "続き"])


def test_hidden_no_terminal(sol):
    _check(sol, "句点なし", ["句点なし"])


def test_hidden_large(sol):
    import random
    rng = random.Random(3)
    pieces = [
        ("これはテスト", "。"),
        ("彼は「了解。」と答えた", "！"),
        ("（注：省略。）以上", "？！"),
        ("本当", "!?"),
        ("『入れ子「です。」』だ", "。"),
    ]
    exp, parts = [], []
    for _ in range(40000):
        body, term = rng.choice(pieces)
        exp.append(body + term)
        parts.append(body + term + rng.choice(["", " ", "\n", "　"]))
    got = sol.split_sentences("".join(parts))
    assert got == exp, f"件数 got={len(got)} expected={len(exp)}"
