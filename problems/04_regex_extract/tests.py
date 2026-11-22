def test_public_basic(sol):
    got = sol.extract_dates("会議は2024年1月5日、締切は2024/02/10です。")
    assert got == ["2024-01-05", "2024-02-10"], got


def test_public_invalid_date(sol):
    got = sol.extract_dates("2023-2-29 は無効、2024-2-29 は有効")
    assert got == ["2024-02-29"], got


def test_public_mask(sol):
    got = sol.mask_dates("期間: 2024/1/5〜2024/13/1")
    assert got == "期間: <DATE>〜2024/13/1", got


def test_hidden_order_across_formats(sol):
    got = sol.extract_dates("2022-1-1、2024/03/01、2023年12月31日、2021/5/5")
    assert got == ["2022-01-01", "2024-03-01", "2023-12-31", "2021-05-05"], got


def test_hidden_mixed_separator(sol):
    got = sol.extract_dates("2024/1-5 と 2024-1/5")
    assert got == [], got


def test_hidden_digit_boundary(sol):
    got = sol.extract_dates("12024/1/5 2024/1/155 12024年1月5日 電話03-1234-5678")
    assert got == [], got


def test_hidden_kanji_followed_by_digits(sol):
    got = sol.extract_dates("2024年1月5日10時に集合")
    assert got == ["2024-01-05"], got


def test_hidden_fullwidth_digits(sol):
    got = sol.extract_dates("２０２４年１２月２５日と２０２５/１/１")
    assert got == ["2024-12-25", "2025-01-01"], got


def test_hidden_invalid_variants(sol):
    got = sol.extract_dates("2024/0/10 2024/1/0 2024-4-31 2023年2月29日 2100-2-29 2024年13月1日")
    assert got == [], got
    got = sol.extract_dates("2000/2/29")
    assert got == ["2000-02-29"], got


def test_hidden_duplicates(sol):
    got = sol.extract_dates("2024/1/1 と 2024-01-01 と 2024年1月1日")
    assert got == ["2024-01-01"] * 3, got


def test_hidden_empty(sol):
    assert sol.extract_dates("") == []
    assert sol.mask_dates("") == ""
    assert sol.extract_dates("日付はありません") == []


def test_hidden_mask_keeps_other_chars(sol):
    src = "（第２回）２０２４年１月５日、次回は2024/2/30（未定）"
    got = sol.mask_dates(src)
    assert got == "（第２回）<DATE>、次回は2024/2/30（未定）", got
    got = sol.mask_dates("2024-01-05から2024年2月3日まで")
    assert got == "<DATE>から<DATE>まで", got


def test_hidden_large(sol):
    from datetime import date, timedelta
    base = date(1990, 1, 1)
    parts, masked, exp = [], [], []
    for i in range(20000):
        d = base + timedelta(days=i * 3)
        fmt = i % 3
        if fmt == 0:
            s = f"{d.year}年{d.month}月{d.day}日"
        elif fmt == 1:
            s = f"{d.year}/{d.month:02d}/{d.day:02d}"
        else:
            s = f"{d.year}-{d.month}-{d.day}"
        parts.append(f"第{i}項は{s}で、")
        masked.append(f"第{i}項は<DATE>で、")
        exp.append(d.isoformat())
    text = "".join(parts)
    got = sol.extract_dates(text)
    assert got == exp, f"件数 got={len(got)} expected={len(exp)}"
    assert sol.mask_dates(text) == "".join(masked)
