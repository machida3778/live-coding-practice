# 11 ビームサーチによる系列デコーディング  (目標 25分 / ★★★)

言語モデルの「次トークンの対数確率」を返す関数が与えられたとき、**ビームサーチ** で生成系列を求める関数を実装してください。

```python
from typing import Callable

def beam_search(
    next_token_logprobs: Callable[[tuple[str, ...]], dict[str, float]],
    beam_width: int,
    max_len: int,
    eos: str = "</s>",
    alpha: float = 0.0,
) -> tuple[tuple[str, ...], float]
```

## 仕様
- `next_token_logprobs(prefix)` は、これまでに生成したトークン列 `prefix` (**タプル**) を受け取り、
  `{次のトークン: 対数確率}` の辞書を返す (空の辞書は返さない)
- 仮説 = (トークン列, 累積対数確率)。初期ビームは `[((), 0.0)]` の1つだけ
- 以下を **最大 `max_len` ステップ** 繰り返す:
  1. ビーム中の各仮説 `(seq, lp)` について `next_token_logprobs(seq)` を呼び、各トークン `t` について候補 `(seq + (t,), lp + logprob)` を作る
  2. 全候補を **累積対数確率の降順** に並べ、上位 `beam_width` 個だけを残す
     (同点はトークン列タプルの辞書順で小さい方を優先)
  3. 残った候補のうち、末尾が `eos` のものは **完了仮説** に移す (以後は展開しない)。それ以外が次のステップのビームになる
     - つまり完了した仮説もそのステップのビーム幅の枠を1つ消費する (ビームはだんだん小さくなりうる)
  4. ビームが空になったら終了
- `max_len` ステップ終えた時点でビームに残っている (eos で終わっていない) 仮説も、完了仮説として扱う
- 完了仮説の中から **スコア = 累積対数確率 / (len(seq) ** alpha)** が最大のものを返す
  - `len(seq)` は eos を含むトークン数。`alpha = 0` ならスコア = 累積対数確率
  - 同点はトークン列タプルの辞書順で小さい方
- 戻り値: `(トークン列のタプル, スコア)`。eos で終わった系列は eos を含む
- `beam_width >= 1`, `max_len >= 1` は保証される。`beam_width = 1` は貪欲法 (greedy) と同じ
- テストではスコアは誤差 1e-9 まで許容して比較する

## 例
```python
>>> import math
>>> P = {
...     (): {"A": 0.6, "B": 0.4},
...     ("A",): {"x": 0.4, "y": 0.35, "</s>": 0.25},
...     ("B",): {"w": 0.9, "</s>": 0.1},
...     ("A", "x"): {"</s>": 1.0},
...     ("A", "y"): {"</s>": 1.0},
...     ("B", "w"): {"</s>": 1.0},
... }
>>> lm = lambda prefix: {t: math.log(p) for t, p in P[prefix].items()}
>>> seq, score = beam_search(lm, beam_width=1, max_len=10)     # 貪欲法: 最初に確率の高い A を選んでしまう
>>> seq, round(math.exp(score), 4)
(('A', 'x', '</s>'), 0.24)
>>> seq, score = beam_search(lm, beam_width=2, max_len=10)     # B の後の w が高確率なので逆転
>>> seq, round(math.exp(score), 4)
(('B', 'w', '</s>'), 0.36)
>>> always = lambda prefix: {"a": math.log(0.9), "</s>": math.log(0.1)}
>>> beam_search(always, beam_width=2, max_len=3)[0]           # max_len に達した未完了仮説も候補
('a', 'a', 'a')
```

## 面接官が聞きそうなこと
- 計算量は? (ビーム幅 k, 語彙 V, 最大長 T) / 上位 k 個の選択を効率化するには? (`heapq.nlargest`, 部分ソート)
- 長さ正規化 (alpha) が無いとどんな系列が選ばれやすい? なぜ?
- 貪欲法・ビームサーチ・サンプリング (temperature / top-k / top-p) はそれぞれどんな場面で使う?
- ビーム幅を大きくすれば常に良い出力になる? (繰り返し・短すぎる出力などの問題)
- 実際の Transformer で使うとき、高速化のために何をする? (バッチ化, KV キャッシュ)
