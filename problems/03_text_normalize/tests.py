def _check(sol, src, exp):
    got = sol.normalize(src)
    assert got == exp, f"入力 {src!r}\n期待 {exp!r}\n実際 {got!r}"


def test_public_nfkc_lower(sol):
    _check(sol, "Ｈｅｌｌｏ　ＷＯＲＬＤ！！", "hello world!!")


def test_public_url(sol):
    _check(sol, "詳細はhttps://example.com/a?b=1を参照", "詳細は<URL>を参照")


def test_public_email(sol):
    _check(sol, "連絡先: Taro.Yamada@Example.COM まで", "連絡先: <EMAIL> まで")


def test_public_digits_spaces(sol):
    _check(sol, "価格は１，９８０円\n\n（税込）", "価格は0,0円 (税込)")


def test_hidden_fullwidth_url(sol):
    # NFKC → 小文字化 → URL置換 の順序でないと置換できない
    _check(sol, "ＨＴＴＰＳ：／／ＥＸＡＭＰＬＥ．ＣＯＭ／ｘ　です", "<URL> です")
    _check(sol, "Visit HTTP://Example.com/Path?Q=1 now", "visit <URL> now")


def test_hidden_halfwidth_kana(sol):
    _check(sol, "ｶﾀｶﾅのﾃﾞｰﾀ", "カタカナのデータ")


def test_hidden_multiple_url_email(sol):
    _check(sol, "a@b.co と https://x.jp と c.d@e.f.g", "<EMAIL> と <URL> と <EMAIL>")
    _check(sol, "連絡はtaro_01@mail.co.jpまで。", "連絡は<EMAIL>まで。")


def test_hidden_not_email(sol):
    _check(sol, "@user さん、a@b", "@user さん、a@b")


def test_hidden_url_boundary(sol):
    _check(sol, "見て→https://a.com/b。", "見て→<URL>。")
    _check(sol, "https://a.com/1 は 2 番目", "<URL> は 0 番目")
    _check(sol, "mail: x@example.com.", "mail: <EMAIL>.")


def test_hidden_url_contains_at(sol):
    # URL の置換が先なので、URL 中の @ はメールとして扱われない
    _check(sol, "https://user@example.com/p と u@example.com", "<URL> と <EMAIL>")


def test_hidden_digit_variants(sol):
    _check(sol, "第Ⅲ章①、3.14と1e10", "第iii章0、0.0と0e0")


def test_hidden_whitespace_and_empty(sol):
    _check(sol, " \t a \n\n b 　 ", "a b")
    _check(sol, "", "")
    _check(sol, "  　\n ", "")


def test_hidden_large(sol):
    line = "ＩＤ１２３　ｈｔｔｐ://ex.com/12 mail:a1@b.jp"
    norm = "id0 <URL> mail:<EMAIL>"
    n = 30000
    got = sol.normalize("\n".join([line] * n))
    exp = " ".join([norm] * n)
    assert got == exp, f"長さ got={len(got)} expected={len(exp)}"
