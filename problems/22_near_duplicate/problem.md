# 22 Jaccard類似度による重複文書検出  (目標 30分 / ★★★)

学習データのクリーニングでよく行う「ほぼ重複 (near-duplicate) した文書の検出」を実装します。
文字 k-shingle の Jaccard 類似度で似た文書のペアを見つけ、Union-Find で重複グループにまとめます。

## 1. `normalize(text) -> str`
- `unicodedata.normalize("NFKC", text)` → 小文字化 → **空白文字 (`\s`) をすべて除去**

## 2. `shingles(text, k) -> set[str]`
- `normalize(text)` の長さ k の連続部分文字列 (文字 k-gram) の集合
- 正規化後の長さが k 未満なら、空でなければ `{正規化後の文字列}`、空なら空集合 (k >= 1 とする)

## 3. `jaccard(a, b) -> float`
- `|a ∩ b| / |a ∪ b|`。**両方とも空集合なら 1.0**

## 4. `find_similar_pairs(docs, k, threshold) -> list[tuple[int, int, float]]`
- `i < j` の全ペアについて shingle 集合の Jaccard 類似度を計算し、**`threshold` 以上** のものを
  `(i, j, 類似度)` で返す。`(i, j)` の昇順

## 5. `class UnionFind(n)`  (要素は 0 〜 n-1。n は 20 万程度まであり得る)
- `find(x) -> int`: x が属するグループの代表元
- `union(x, y) -> bool`: x と y のグループを併合。新たに併合したら True、既に同じグループなら False
- `groups() -> list[list[int]]`: 全グループ (要素 1 個のグループも含む)。各グループは昇順、グループ同士は最小要素の昇順

## 6. `group_duplicates(docs, k=3, threshold=0.5) -> list[list[int]]`
- `find_similar_pairs` で見つかったペアを Union-Find で併合し (**推移的にまとめる**: A≈B, B≈C なら A,B,C は同じグループ)、
  **要素が 2 個以上のグループだけ** を返す。並び順は `groups()` と同じ

## 例
```python
>>> sorted(shingles("ＡＢ cd", 2))
['ab', 'bc', 'cd']
>>> jaccard({"a", "b", "c"}, {"b", "c", "d"})
0.5
>>> docs = ["今日は良い天気ですね。", "今日は 良い天気ですね！", "明日は雨が降るでしょう。",
...         "今日はとても良い天気ですね。", "明日は雨が降るでしょう", "ＮＥＸＵＳ Technology", "nexus technology!"]
>>> [(i, j, round(s, 3)) for i, j, s in find_similar_pairs(docs, 2, 0.5)]
[(0, 1, 0.818), (0, 3, 0.643), (1, 3, 0.533), (2, 4, 0.909), (5, 6, 0.933)]
>>> group_duplicates(docs, k=2, threshold=0.5)
[[0, 1, 3], [2, 4], [5, 6]]
```

## 面接官が聞きそうなこと
- 全ペア比較は O(n²)。文書が 1 億件あったらどうする? (MinHash で Jaccard を近似 + LSH で候補ペアを絞る)
- MinHash で Jaccard 類似度が推定できるのはなぜ?
- Union-Find の経路圧縮・union by rank (size) をすると計算量はどうなる?
- k (shingle 長) を大きく/小さくするとどうなる? 日本語と英語で適切な k は違う?
