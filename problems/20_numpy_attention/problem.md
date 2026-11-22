# 20 numpyでAttentionを実装  (目標 25分 / ★★★)

Transformer の Attention を numpy だけで実装してください。
**Python の for ループで要素を回さず、ベクトル化 (行列演算・ブロードキャスト) で書くこと** を推奨します
(大きめの入力での性能テストがあります)。

## 1. `softmax(x, axis=-1) -> np.ndarray`
- 指定 axis 方向の softmax。**数値的に安定** であること (例: `[1000, 1001, 1002]` で nan/inf にならない)

## 2. `causal_mask(L) -> np.ndarray`
- 形状 `(L, L)` の bool 配列。`mask[i, j] = (j <= i)` (自分より未来の位置は見ない)

## 3. `padding_mask(lengths, max_len) -> np.ndarray`
- `lengths`: 各系列の長さのリスト (長さ B)。形状 `(B, max_len)` の bool 配列で `mask[b, j] = (j < lengths[b])`

## 4. `scaled_dot_product_attention(Q, K, V, mask=None) -> (output, weights)`
- `Q`: `(..., Lq, d_k)`, `K`: `(..., Lk, d_k)`, `V`: `(..., Lk, d_v)`  (`...` は任意個のバッチ次元)
- `scores = Q @ K^T / sqrt(d_k)` (形状 `(..., Lq, Lk)`)
- `mask` は `scores` にブロードキャスト可能な配列 (bool または 0/1)。
  **mask が False/0 の位置のスコアを `-1e9` で置き換える** (加算ではなく置換。`-inf` も使わない)。
  その結果、全位置が mask された行の重みは一様分布になる
- `weights = softmax(scores, axis=-1)`, `output = weights @ V` (形状 `(..., Lq, d_v)`)

## 5. `multi_head_attention(X, W_q, W_k, W_v, W_o, n_heads, mask=None) -> np.ndarray`
- `X`: `(B, L, d_model)`, `W_q, W_k, W_v, W_o`: `(d_model, d_model)`
- `d_model` が `n_heads` で割り切れなければ `ValueError`。`d_head = d_model // n_heads`
- 手順:
  1. `Q = X @ W_q`, `K = X @ W_k`, `V = X @ W_v`  (各 `(B, L, d_model)`)
  2. 最後の次元を `(n_heads, d_head)` に分割し `(B, n_heads, L, d_head)` に並べ替える
     (ヘッド h は最後の次元の `h*d_head : (h+1)*d_head` 列を使う)
  3. 各ヘッドで `scaled_dot_product_attention` (mask は `(B, n_heads, L, L)` にブロードキャスト可能な配列をそのまま渡す)
  4. ヘッドの出力を `(B, L, d_model)` に戻して連結し、`W_o` を掛けて返す

## 例
```python
>>> softmax(np.array([1000.0, 1001.0, 1002.0]))
array([0.09003057, 0.24472847, 0.66524096])
>>> causal_mask(3)
array([[ True, False, False],
       [ True,  True, False],
       [ True,  True,  True]])
>>> padding_mask([2, 3], 4)
array([[ True,  True, False, False],
       [ True,  True,  True, False]])
>>> Q = K = np.eye(2); V = np.array([[1.0, 0.0], [0.0, 1.0]])
>>> out, w = scaled_dot_product_attention(Q, K, V, mask=causal_mask(2))
>>> w
array([[1.        , 0.        ],
       [0.33023845, 0.66976155]])
```

## 面接官が聞きそうなこと
- なぜ `sqrt(d_k)` で割る? 割らないと何が起きる?
- softmax で最大値を引くと何が嬉しい? 結果は変わらないのはなぜ?
- Self-Attention の計算量・メモリは系列長 L に対してどうなる? 長文ではどう対処する?
- causal mask と padding mask はそれぞれどんな場面で使う? (デコーダ / バッチ内の長さの違い)
