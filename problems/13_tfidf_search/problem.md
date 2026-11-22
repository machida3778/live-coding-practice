# 13 TF-IDF 類似文書検索  (目標 25分 / ★★)

トークン化済みの文書群に対して TF-IDF を **自作** し、クエリとのコサイン類似度で文書を検索する関数群を実装してください。
(sklearn の `TfidfVectorizer` は使用禁止。標準ライブラリ・numpy は使用可)

## 仕様
文書数を `N` (空の文書も数える)、単語 `t` を含む文書の数を `df(t)` とします。対数は自然対数 (`math.log`) です。

### 1. `compute_idf(docs: list[list[str]]) -> dict[str, float]`
- docs に1回以上現れる全単語 `t` について `idf(t) = ln(N / df(t)) + 1` を返す
- docs が空リストなら空の辞書

### 2. `tfidf(doc: list[str], idf: dict[str, float]) -> dict[str, float]`
- `tf(t, d) = (d における t の出現回数) / len(d)`
  - `len(d)` は idf に **無い** 単語も含めた総トークン数
- 重み `w(t) = tf(t, d) * idf(t)` を値とする辞書 (疎ベクトル) を返す
- idf に無い単語はキーに含めない。`doc` が空なら空の辞書

### 3. `cosine(u: dict[str, float], v: dict[str, float]) -> float`
- 疎ベクトル (dict) 同士のコサイン類似度 `u·v / (|u| |v|)`
- どちらかのノルムが 0 (空の辞書を含む) なら `0.0`

### 4. `search(docs: list[list[str]], query: list[str], k: int) -> list[tuple[int, float]]`
- docs から idf を計算し、各文書の TF-IDF ベクトルと query の TF-IDF ベクトル (同じ idf を使う) のコサイン類似度を求める
- `(文書のindex, 類似度)` のリストを返す
  - 類似度の **降順**。同点の場合は文書 index の **昇順**
  - 類似度が `0` の文書は含めない
  - 上位 `k` 件まで。`k <= 0` なら空リスト

テストでは浮動小数点を許容誤差 `1e-9` で比較します。

## 例
```python
>>> docs = [["cat", "sat", "on", "mat"], ["dog", "sat", "on", "log"], ["cat", "cat", "dog"]]
>>> idf = compute_idf(docs)
>>> idf["cat"], idf["mat"]          # ln(3/2)+1, ln(3/1)+1
(1.4054651081081644, 2.09861228866811)
>>> tfidf(["cat", "cat", "dog"], idf)
{'cat': 0.9369767387387762, 'dog': 0.4684883693693881}
>>> tfidf(["cat", "bird"], idf)     # bird は idf に無いので無視 (ただし len には数える)
{'cat': 0.7027325540540822}
>>> cosine({"a": 1.0, "b": 1.0}, {"a": 1.0})
0.7071067811865475
>>> search(docs, ["cat"], 2)
[(2, 0.8944271909999157), (0, 0.43728676490568713)]
>>> search(docs, ["sat", "dog"], 3)
[(1, 0.6184168735758779), (2, 0.3162277660168379), (0, 0.30920843678793897)]
>>> search(docs, ["bird"], 3)
[]
```

## 面接官が聞きそうなこと
- tf を文書長で割る正規化は、コサイン類似度の結果に影響する? しない? それはなぜ?
- 文書が100万件になったらどう高速化する? (転置インデックス、`scipy.sparse`、事前計算したノルム、近似最近傍探索)
- sklearn の `TfidfVectorizer` の既定 (`smooth_idf=True`, `sublinear_tf=False`, L2正規化) との違いは?
- TF-IDF の弱点は? (語順や同義語を扱えない → BM25 / 埋め込みベクトル検索 / ハイブリッド検索)
