# 19 pandasでレビューデータ分析  (目標 20分 / ★★)

商品レビューの DataFrame を分析する 4 つの関数を実装してください。
動作確認用に `reviews.csv` (解答ファイルと同じディレクトリ) があります: `pd.read_csv("reviews.csv")`

入力 DataFrame の列: `product` (str), `rating` (1〜5, 欠損あり), `text` (str, 欠損あり), `date` ("YYYY-MM-DD" の文字列)

## 1. `clean(df) -> pd.DataFrame`
次の順に処理した **新しい** DataFrame を返す (引数の df は変更しない):
1. `text` の前後の空白を除去
2. `text` が欠損または空文字の行、`rating` が欠損の行を削除
3. 全列が一致する重複行は最初の 1 行だけ残す (1. の後の値で判定)
4. `date` を datetime 型 (`datetime64`) に、`rating` を整数型 (`int64`) に変換
5. index を 0 からの連番に振り直す (列の並びは元のまま)

## 2. `product_summary(df) -> pd.DataFrame`  (入力は clean 済み)
- index: 商品名 (index 名 `"product"`)
- 列: `["n_reviews", "mean_rating", "mean_length"]`
  - `n_reviews`: レビュー件数 (整数)
  - `mean_rating`: 平均評価 (float)
  - `mean_length`: `text` の平均文字数 (float)
- 並び順: `n_reviews` の降順、同数なら商品名の昇順

## 3. `monthly_counts(df) -> pd.Series`  (入力は clean 済み)
- 月ごとのレビュー件数。index は `"YYYY-MM"` 形式の **文字列** (index 名 `"month"`)、Series 名は `"count"`
- データ中の最も古い月から最も新しい月まで、**レビューが 0 件の月も 0 として含める**。月の昇順
- 空の df なら空の Series

## 4. `sentiment_words(df, n, stopwords=frozenset()) -> dict`  (入力は clean 済み)
- 高評価 (`rating >= 4`) / 低評価 (`rating <= 2`) のレビューそれぞれで、
  **その単語を含むレビューの件数** (1件の中で何度出ても 1) の上位 n 語を求める
- 単語 = `text` を小文字化して `[a-z]+` にマッチする部分。`stopwords` に含まれる単語は除外
- 戻り値: `{"positive": [(単語, 件数), ...], "negative": [...]}`
  - 件数の降順、同数なら単語の昇順。`n <= 0` なら空リスト

## 例
```python
>>> raw = pd.DataFrame({
...     "product": ["A", "B", "A", "A", "B", "C", "A"],
...     "rating":  [5, 2, None, 4, 2, 1, 4],
...     "text":    ["Great phone!", "Bad battery. Bad.", "ok", "  Nice  ", "Bad battery. Bad.", "   ", "Nice"],
...     "date":    ["2024-01-05", "2024-01-20", "2024-02-01", "2024-03-10", "2024-01-20", "2024-03-11", "2024-03-10"],
... })
>>> df = clean(raw)
>>> df
  product  rating               text       date
0       A       5       Great phone! 2024-01-05
1       B       2  Bad battery. Bad. 2024-01-20
2       A       4               Nice 2024-03-10
>>> product_summary(df)
         n_reviews  mean_rating  mean_length
product
A                2          4.5          8.0
B                1          2.0         17.0
>>> monthly_counts(df)
month
2024-01    2
2024-02    0
2024-03    1
Name: count, dtype: int64
>>> sentiment_words(df, 2)
{'positive': [('great', 1), ('nice', 1)], 'negative': [('bad', 1), ('battery', 1)]}
```

## 面接官が聞きそうなこと
- `inplace=True` や `SettingWithCopyWarning` について説明できる?
- `groupby().agg()` と `apply` の違い・速度は? 行ごとの `iterrows` を避けるべき理由は?
- 数百万行・メモリに載らない CSV ならどうする? (`chunksize`, dtype 指定, Parquet, DuckDB など)
