import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    z = x - np.max(x, axis=axis, keepdims=True)  # 最大値を引いてオーバーフローを防ぐ
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def causal_mask(L: int) -> np.ndarray:
    return np.tril(np.ones((L, L), dtype=bool))


def padding_mask(lengths: list[int], max_len: int) -> np.ndarray:
    return np.arange(max_len)[None, :] < np.asarray(lengths)[:, None]


def scaled_dot_product_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                                 mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    d_k = Q.shape[-1]
    scores = Q @ np.swapaxes(K, -1, -2) / np.sqrt(d_k)
    if mask is not None:
        scores = np.where(np.asarray(mask).astype(bool), scores, -1e9)
    weights = softmax(scores, axis=-1)
    return weights @ V, weights


def multi_head_attention(X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray, W_o: np.ndarray,
                         n_heads: int, mask: np.ndarray | None = None) -> np.ndarray:
    B, L, d_model = X.shape
    if d_model % n_heads != 0:
        raise ValueError("d_model must be divisible by n_heads")
    d_head = d_model // n_heads

    def split(T: np.ndarray) -> np.ndarray:  # (B, L, d_model) -> (B, h, L, d_head)
        return T.reshape(B, L, n_heads, d_head).transpose(0, 2, 1, 3)

    Q, K, V = split(X @ W_q), split(X @ W_k), split(X @ W_v)
    heads, _ = scaled_dot_product_attention(Q, K, V, mask)
    concat = heads.transpose(0, 2, 1, 3).reshape(B, L, d_model)
    return concat @ W_o
