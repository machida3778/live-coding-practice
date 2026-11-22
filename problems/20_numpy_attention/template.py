import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    raise NotImplementedError


def causal_mask(L: int) -> np.ndarray:
    raise NotImplementedError


def padding_mask(lengths: list[int], max_len: int) -> np.ndarray:
    raise NotImplementedError


def scaled_dot_product_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                                 mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    raise NotImplementedError


def multi_head_attention(X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray, W_o: np.ndarray,
                         n_heads: int, mask: np.ndarray | None = None) -> np.ndarray:
    raise NotImplementedError
