import math

import numpy as np


# ---- テスト用の素朴な (ループによる) 参照実装 ----
def _naive_attention_2d(Q, K, V, mask=None):
    Lq, d = Q.shape
    Lk = K.shape[0]
    W = np.zeros((Lq, Lk))
    for i in range(Lq):
        s = [float(Q[i] @ K[j]) / math.sqrt(d) for j in range(Lk)]
        if mask is not None:
            s = [s[j] if mask[i][j] else -1e9 for j in range(Lk)]
        m = max(s)
        e = [math.exp(v - m) for v in s]
        W[i] = [v / sum(e) for v in e]
    return W @ V, W


def _naive_mha(X, Wq, Wk, Wv, Wo, h, mask=None):
    B, L, D = X.shape
    dh = D // h
    out = np.zeros((B, L, D))
    for b in range(B):
        Q, K, V = X[b] @ Wq, X[b] @ Wk, X[b] @ Wv
        heads = []
        for k in range(h):
            sl = slice(k * dh, (k + 1) * dh)
            m = None
            if mask is not None:
                m = np.broadcast_to(mask, (B, h, L, L))[b, k]
            o, _ = _naive_attention_2d(Q[:, sl], K[:, sl], V[:, sl], m)
            heads.append(o)
        out[b] = np.concatenate(heads, axis=1) @ Wo
    return out


def _rand(*shape, seed=0):
    return np.random.default_rng(seed).normal(size=shape)


# ---------------------------------------------------------------- public
def test_public_softmax_basic(sol):
    got = sol.softmax(np.array([1.0, 2.0, 3.0]))
    exp = np.exp([1.0, 2.0, 3.0]) / np.exp([1.0, 2.0, 3.0]).sum()
    assert np.allclose(got, exp), got


def test_public_causal_mask(sol):
    got = np.asarray(sol.causal_mask(3))
    exp = np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1]], dtype=bool)
    assert got.shape == (3, 3) and np.array_equal(got.astype(bool), exp), got


def test_public_attention_example(sol):
    Q = K = np.eye(2)
    V = np.array([[1.0, 0.0], [0.0, 1.0]])
    out, w = sol.scaled_dot_product_attention(Q, K, V, mask=np.array([[True, False], [True, True]]))
    exp_w = np.array([[1.0, 0.0], [0.33023845, 0.66976155]])
    assert np.allclose(w, exp_w, atol=1e-7), w
    assert np.allclose(out, exp_w @ V), out


def test_public_mha_matches_naive(sol):
    X = _rand(2, 5, 8, seed=1)
    Ws = [_rand(8, 8, seed=s) * 0.3 for s in (2, 3, 4, 5)]
    got = sol.multi_head_attention(X, *Ws, n_heads=2)
    exp = _naive_mha(X, *Ws, 2)
    assert got.shape == (2, 5, 8), got.shape
    assert np.allclose(got, exp, atol=1e-8), "naive 実装と一致しません"


# ---------------------------------------------------------------- hidden
def test_hidden_softmax_stability(sol):
    got = sol.softmax(np.array([1000.0, 1001.0, 1002.0]))
    assert np.all(np.isfinite(got)), got
    assert np.allclose(got, [0.09003057, 0.24472847, 0.66524096]), got
    got = sol.softmax(np.array([-1e9, 0.0, -1e9]))
    assert np.allclose(got, [0.0, 1.0, 0.0]), got


def test_hidden_softmax_axis_and_ndim(sol):
    x = _rand(3, 4, 5, seed=7) * 50
    for axis in (0, 1, -1):
        got = sol.softmax(x, axis=axis)
        e = np.exp(x - x.max(axis=axis, keepdims=True))
        assert got.shape == x.shape
        assert np.allclose(got, e / e.sum(axis=axis, keepdims=True)), f"axis={axis}"


def test_hidden_padding_mask(sol):
    got = np.asarray(sol.padding_mask([2, 3, 0, 4], 4))
    exp = np.array([[1, 1, 0, 0], [1, 1, 1, 0], [0, 0, 0, 0], [1, 1, 1, 1]], dtype=bool)
    assert got.shape == (4, 4) and np.array_equal(got.astype(bool), exp), got


def test_hidden_attention_causal_vs_naive(sol):
    Q, K, V = _rand(6, 4, seed=10), _rand(6, 4, seed=11), _rand(6, 3, seed=12)
    m = np.tril(np.ones((6, 6), dtype=bool))
    out, w = sol.scaled_dot_product_attention(Q, K, V, m)
    eo, ew = _naive_attention_2d(Q, K, V, m)
    assert np.allclose(w, ew, atol=1e-10) and np.allclose(out, eo, atol=1e-10)
    assert np.allclose(np.triu(w, 1), 0.0), "未来の位置に重みがあります"


def test_hidden_attention_rectangular_no_mask(sol):
    Q, K, V = _rand(3, 5, seed=20), _rand(7, 5, seed=21), _rand(7, 2, seed=22)
    out, w = sol.scaled_dot_product_attention(Q, K, V)
    eo, ew = _naive_attention_2d(Q, K, V)
    assert out.shape == (3, 2) and w.shape == (3, 7), (out.shape, w.shape)
    assert np.allclose(out, eo, atol=1e-10) and np.allclose(w, ew, atol=1e-10)


def test_hidden_attention_fully_masked_row_uniform(sol):
    Q, K, V = _rand(2, 4, seed=30), _rand(3, 4, seed=31), _rand(3, 2, seed=32)
    m = np.array([[0, 0, 0], [1, 1, 0]])  # 0/1 の int マスク
    out, w = sol.scaled_dot_product_attention(Q, K, V, m)
    assert np.allclose(w[0], [1 / 3] * 3), f"全て mask された行は一様分布になるはず: {w[0]}"
    assert np.isclose(w[1, 2], 0.0) and np.isclose(w[1].sum(), 1.0), w[1]


def test_hidden_attention_batched_broadcast_mask(sol):
    B, H, L, d = 3, 2, 5, 4
    Q, K, V = _rand(B, H, L, d, seed=40), _rand(B, H, L, d, seed=41), _rand(B, H, L, d, seed=42)
    pm = np.arange(L)[None, :] < np.array([5, 3, 1])[:, None]
    mask = pm[:, None, None, :]  # (B, 1, 1, L)
    out, w = sol.scaled_dot_product_attention(Q, K, V, mask)
    assert out.shape == (B, H, L, d) and w.shape == (B, H, L, L)
    for b in range(B):
        for h in range(H):
            eo, ew = _naive_attention_2d(Q[b, h], K[b, h], V[b, h], np.broadcast_to(pm[b], (L, L)))
            assert np.allclose(out[b, h], eo, atol=1e-10) and np.allclose(w[b, h], ew, atol=1e-10), (b, h)


def test_hidden_mha_with_masks(sol):
    B, L, D, h = 3, 6, 12, 3
    X = _rand(B, L, D, seed=50)
    Ws = [_rand(D, D, seed=s) * 0.2 for s in (51, 52, 53, 54)]
    causal = np.tril(np.ones((L, L), dtype=bool))
    got = sol.multi_head_attention(X, *Ws, n_heads=h, mask=causal)
    assert np.allclose(got, _naive_mha(X, *Ws, h, causal), atol=1e-8), "causal mask 付きで不一致"
    pm = (np.arange(L)[None, :] < np.array([6, 4, 2])[:, None])[:, None, None, :]
    got = sol.multi_head_attention(X, *Ws, n_heads=h, mask=pm)
    assert np.allclose(got, _naive_mha(X, *Ws, h, pm), atol=1e-8), "padding mask 付きで不一致"


def test_hidden_mha_single_head_equals_attention(sol):
    X = _rand(1, 4, 6, seed=60)
    Ws = [_rand(6, 6, seed=s) for s in (61, 62, 63)] + [np.eye(6)]
    got = sol.multi_head_attention(X, *Ws, n_heads=1)
    exp, _ = _naive_attention_2d(X[0] @ Ws[0], X[0] @ Ws[1], X[0] @ Ws[2])
    assert np.allclose(got[0], exp, atol=1e-10)


def test_hidden_mha_invalid_heads(sol):
    X = _rand(1, 3, 10, seed=70)
    Ws = [np.eye(10)] * 4
    try:
        sol.multi_head_attention(X, *Ws, n_heads=3)
    except ValueError:
        return
    raise AssertionError("d_model が n_heads で割り切れないときは ValueError")


def test_hidden_performance(sol):
    B, L, D, h = 16, 128, 256, 8
    X = _rand(B, L, D, seed=80)
    Ws = [_rand(D, D, seed=s) * 0.05 for s in (81, 82, 83, 84)]
    mask = np.tril(np.ones((L, L), dtype=bool))
    got = sol.multi_head_attention(X, *Ws, n_heads=h, mask=mask)
    assert got.shape == (B, L, D)
    # 1 バッチ分だけ naive と照合
    exp = _naive_mha(X[:1, :16], *Ws, h, mask[:16, :16])
    got_small = sol.multi_head_attention(X[:1, :16], *Ws, n_heads=h, mask=mask[:16, :16])
    assert np.allclose(got_small, exp, atol=1e-8)
