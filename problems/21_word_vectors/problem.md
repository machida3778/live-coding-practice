# 21 単語ベクトルの類似検索とアナロジー  (目標 20分 / ★★)

学習済みの単語ベクトル (word2vec のようなもの) を使って、類似語検索とアナロジー (king - man + woman ≈ queen) を
numpy で実装してください。語彙数 5 万程度でも高速に動くよう **ベクトル化** すること (性能テストあり)。

## 1. `normalize_rows(X) -> np.ndarray`
- 2 次元配列 `X` の各行を L2 ノルムで割ったものを返す (`X` 自体は変更しない)
- **ノルムが 0 の行は 0 のまま** (nan や警告を出さない)

## 2. `class WordVectors(words, vectors)`
- `words`: 単語のリスト (重複なし), `vectors`: 形状 `(len(words), d)` の配列。`vectors[i]` が `words[i]` のベクトル
- 語彙に無い単語が渡されたメソッドは `KeyError` を送出する
- コサイン類似度 `cos(u, v) = u·v / (|u||v|)`。**どちらかがゼロベクトルなら 0.0** とする

### `similarity(w1, w2) -> float`
- 2 語のコサイン類似度

### `most_similar(word, k=5) -> list[tuple[str, float]]`
- `word` 以外の全単語とのコサイン類似度を計算し、上位 k 件を `(単語, 類似度)` で返す
- 類似度の降順。**同じ類似度なら `words` での出現順 (index の小さい順)**
- 候補が k 件未満ならあるだけ返す。`k <= 0` なら空リスト

### `analogy(a, b, c, k=1) -> list[tuple[str, float]]`
- 「a : b = c : ?」を解く。各ベクトルを L2 正規化したもの (`unit(x)`) を使って
  `target = unit(b) - unit(a) + unit(c)` とし、`target` とのコサイン類似度上位 k 件を返す
- **a, b, c 自身は候補から除外**。並び順・k の扱いは `most_similar` と同じ

## 例
```python
>>> words = ["man", "woman", "king", "queen", "apple", "prince"]
>>> vecs = np.array([[0, 1, 0], [0, -1, 0], [1, 1, 0], [1, -1, 0], [0, 0, 1], [1, 1, 0.2]], dtype=float)
>>> wv = WordVectors(words, vecs)
>>> round(wv.similarity("king", "queen"), 4)
0.0
>>> [(w, round(s, 4)) for w, s in wv.most_similar("king", 2)]
[('prince', 0.9901), ('man', 0.7071)]
>>> [(w, round(s, 4)) for w, s in wv.analogy("man", "king", "woman")]
[('queen', 0.9597)]
```

## 面接官が聞きそうなこと
- 内積ではなくコサイン類似度を使うのはなぜ? 事前に正規化しておくと何が嬉しい?
- 上位 k 件だけ欲しいとき `argsort` 全体より速い方法は? (`np.argpartition`, ヒープ)
- 語彙が数億件になったら? (近似最近傍探索: Faiss, HNSW, ScaNN など)
- word2vec の分散表現と BERT 等の文脈化された埋め込みの違いは?
