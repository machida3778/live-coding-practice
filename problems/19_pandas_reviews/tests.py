import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
CSV = HERE / "reviews.csv"


def _raw_small():
    return pd.DataFrame({
        "product": ["A", "B", "A", "A", "B", "C", "A"],
        "rating": [5, 2, None, 4, 2, 1, 4],
        "text": ["Great phone!", "Bad battery. Bad.", "ok", "  Nice  ", "Bad battery. Bad.", "   ", "Nice"],
        "date": ["2024-01-05", "2024-01-20", "2024-02-01", "2024-03-10", "2024-01-20", "2024-03-11", "2024-03-10"],
    })


def _clean_small():
    return pd.DataFrame({
        "product": ["A", "B", "A"],
        "rating": [5, 2, 4],
        "text": ["Great phone!", "Bad battery. Bad.", "Nice"],
        "date": pd.to_datetime(["2024-01-05", "2024-01-20", "2024-03-10"]),
    })


def _expected_clean_rows():
    """CSV から期待値を素の Python で計算 (product, rating, text, 'YYYY-MM-DD')"""
    seen, out = set(), []
    with open(CSV, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            text = row["text"].strip()
            if text == "" or row["rating"] == "":
                continue
            key = (row["product"], int(float(row["rating"])), text, row["date"])
            if key in seen:
                continue
            seen.add(key)
            out.append(key)
    return out


def _rows_of(df):
    return list(zip(df["product"], [int(x) for x in df["rating"]], df["text"],
                    pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")))


def _cleaned_full_ref():
    rows = _expected_clean_rows()
    return pd.DataFrame({
        "product": [r[0] for r in rows],
        "rating": pd.Series([r[1] for r in rows], dtype="int64"),
        "text": [r[2] for r in rows],
        "date": pd.to_datetime([r[3] for r in rows]),
    })


# ---------------------------------------------------------------- public
def test_public_clean_example(sol):
    got = sol.clean(_raw_small())
    assert list(got.columns) == ["product", "rating", "text", "date"], list(got.columns)
    assert _rows_of(got) == _rows_of(_clean_small()), _rows_of(got)
    assert list(got.index) == [0, 1, 2], list(got.index)


def test_public_product_summary_example(sol):
    got = sol.product_summary(_clean_small())
    exp = pd.DataFrame({"n_reviews": [2, 1], "mean_rating": [4.5, 2.0], "mean_length": [8.0, 17.0]},
                       index=pd.Index(["A", "B"], name="product"))
    pd.testing.assert_frame_equal(got, exp, check_dtype=False)


def test_public_monthly_counts_example(sol):
    got = sol.monthly_counts(_clean_small())
    exp = pd.Series([2, 0, 1], index=pd.Index(["2024-01", "2024-02", "2024-03"], name="month"), name="count")
    pd.testing.assert_series_equal(got, exp, check_dtype=False)


def test_public_sentiment_words_example(sol):
    got = sol.sentiment_words(_clean_small(), 2)
    exp = {"positive": [("great", 1), ("nice", 1)], "negative": [("bad", 1), ("battery", 1)]}
    assert got == exp, got


# ---------------------------------------------------------------- hidden
def test_hidden_clean_full_csv(sol):
    raw = pd.read_csv(CSV)
    got = sol.clean(raw)
    exp = _expected_clean_rows()
    assert len(got) == len(exp), f"行数 {len(got)} != 期待 {len(exp)}"
    assert _rows_of(got) == exp, "clean の結果が期待と一致しません (処理順序・重複判定を確認)"
    assert list(got.index) == list(range(len(exp))), "index が 0 からの連番になっていません"


def test_hidden_clean_types_and_no_mutation(sol):
    raw = pd.read_csv(CSV)
    before = raw.copy()
    got = sol.clean(raw)
    pd.testing.assert_frame_equal(raw, before, obj="引数の df が変更されています")
    assert pd.api.types.is_datetime64_any_dtype(got["date"]), got["date"].dtype
    assert pd.api.types.is_integer_dtype(got["rating"]), got["rating"].dtype
    assert list(got.columns) == ["product", "rating", "text", "date"], list(got.columns)


def test_hidden_product_summary_full(sol):
    df = _cleaned_full_ref()
    got = sol.product_summary(df)
    agg = defaultdict(list)
    for p, r, t in zip(df["product"], df["rating"], df["text"]):
        agg[p].append((r, len(t)))
    items = sorted(agg.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    exp = pd.DataFrame({
        "n_reviews": [len(v) for _, v in items],
        "mean_rating": [sum(r for r, _ in v) / len(v) for _, v in items],
        "mean_length": [sum(l for _, l in v) / len(v) for _, v in items],
    }, index=pd.Index([k for k, _ in items], name="product"))
    pd.testing.assert_frame_equal(got, exp, check_dtype=False)
    assert pd.api.types.is_integer_dtype(got["n_reviews"]), got["n_reviews"].dtype


def test_hidden_product_summary_tie_break(sol):
    df = pd.DataFrame({"product": ["b", "a", "c", "a", "b"], "rating": [5, 3, 4, 1, 4],
                       "text": ["xx", "yyyy", "z", "ww", "vvvvvv"],
                       "date": pd.to_datetime(["2024-01-01"] * 5)})
    got = sol.product_summary(df)
    exp = pd.DataFrame({"n_reviews": [2, 2, 1], "mean_rating": [2.0, 4.5, 4.0], "mean_length": [3.0, 4.0, 1.0]},
                       index=pd.Index(["a", "b", "c"], name="product"))
    pd.testing.assert_frame_equal(got, exp, check_dtype=False)


def test_hidden_monthly_counts_full(sol):
    df = _cleaned_full_ref()
    got = sol.monthly_counts(df)
    c = Counter(d.strftime("%Y-%m") for d in df["date"])
    months, (y, m) = [], (2023, 11)
    while (y, m) <= (2024, 10):
        months.append(f"{y:04d}-{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    exp = pd.Series([c.get(k, 0) for k in months], index=pd.Index(months, name="month"), name="count")
    assert exp["2024-06"] == 0
    pd.testing.assert_series_equal(got, exp, check_dtype=False)


def test_hidden_monthly_counts_year_boundary(sol):
    df = pd.DataFrame({"product": ["a"] * 4, "rating": [1, 2, 3, 4], "text": ["x"] * 4,
                       "date": pd.to_datetime(["2024-02-29", "2023-12-31", "2024-02-01", "2023-12-01"])})
    got = sol.monthly_counts(df)
    exp = pd.Series([2, 0, 2], index=pd.Index(["2023-12", "2024-01", "2024-02"], name="month"), name="count")
    pd.testing.assert_series_equal(got, exp, check_dtype=False)


def test_hidden_monthly_counts_empty(sol):
    df = pd.DataFrame({"product": [], "rating": [], "text": [], "date": pd.to_datetime([])})
    got = sol.monthly_counts(df)
    assert isinstance(got, pd.Series) and len(got) == 0, got


def test_hidden_sentiment_words_doc_freq_and_stopwords(sol):
    df = pd.DataFrame({
        "product": ["a"] * 5,
        "rating": [5, 4, 1, 2, 3],
        "text": ["Great battery. Great screen!", "Great price", "Bad battery", "battery bad, BAD support", "great but bad"],
        "date": pd.to_datetime(["2024-01-01"] * 5),
    })
    got = sol.sentiment_words(df, 3)
    assert got == {"positive": [("great", 2), ("battery", 1), ("price", 1)],
                   "negative": [("bad", 2), ("battery", 2), ("support", 1)]}, got
    got = sol.sentiment_words(df, 5, stopwords=frozenset({"battery"}))
    assert got == {"positive": [("great", 2), ("price", 1), ("screen", 1)],
                   "negative": [("bad", 2), ("support", 1)]}, got
    assert sol.sentiment_words(df, 0) == {"positive": [], "negative": []}


def test_hidden_sentiment_words_full(sol):
    df = _cleaned_full_ref()
    got = sol.sentiment_words(df, 5, stopwords=frozenset({"the", "it", "is", "and"}))
    stop = {"the", "it", "is", "and"}

    def top(cond):
        c = Counter()
        for r, t in zip(df["rating"], df["text"]):
            if cond(r):
                c.update(set(re.findall(r"[a-z]+", t.lower())) - stop)
        return sorted(c.items(), key=lambda x: (-x[1], x[0]))[:5]

    exp = {"positive": top(lambda r: r >= 4), "negative": top(lambda r: r <= 2)}
    assert got == exp, (got, exp)
