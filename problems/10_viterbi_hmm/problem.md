# 10 HMM 品詞タグ付け (Viterbi アルゴリズム)  (目標 30分 / ★★★)

隠れマルコフモデル (HMM) のパラメータが与えられたとき、観測された単語列に対して
**最も確率の高い品詞 (状態) 列** を Viterbi アルゴリズムで求めてください。

```python
def viterbi(
    observations: list[str],
    states: list[str],
    start_p: dict[str, float],
    trans_p: dict[str, dict[str, float]],
    emit_p: dict[str, dict[str, float]],
) -> tuple[list[str], float]
```

## 仕様
- `start_p[s]`: 先頭の状態が `s` である確率
- `trans_p[s][t]`: 状態 `s` から状態 `t` への遷移確率
- `emit_p[s][w]`: 状態 `s` が単語 `w` を出力する確率
- 与えられる確率はすべて **通常の確率 (log ではない)** で、0 より大きい値
- 辞書にキーが無い場合:
  - `start_p` に無い状態 / `trans_p[s]` に無い遷移先 → 確率 **0** (その経路は不可能)
  - `emit_p[s]` に無い単語 → 確率 **1e-6** (未知語のスムージング)
  - `trans_p` と `emit_p` は `states` の全状態をキーに持つ (ただし中の辞書は欠けていることがある)
- 経路の確率 = `start_p[s1] * emit_p[s1][w1] * Π_{i>=2} trans_p[s_{i-1}][s_i] * emit_p[s_i][w_i]`
- 戻り値: `(最尤の状態列 (list), その経路の対数確率 (自然対数, float))`
- **対数空間で計算すること**。長い系列 (数千語) では確率の積がアンダーフローして 0 になるため
- 同点の場合は `states` で先に現れる状態を優先する (テストでは同点が起きないようにしてある)
- `observations` が空なら `([], 0.0)` を返す
- 可能な経路が1つも無い (全経路の確率が 0) なら `([], float("-inf"))` を返す
- テストでは対数確率は相対誤差 1e-6 まで許容して比較する

## 例
```python
>>> states = ["DET", "NOUN", "VERB"]
>>> start_p = {"DET": 0.6, "NOUN": 0.3, "VERB": 0.1}
>>> trans_p = {
...     "DET":  {"DET": 0.05, "NOUN": 0.9, "VERB": 0.05},
...     "NOUN": {"DET": 0.1,  "NOUN": 0.3, "VERB": 0.6},
...     "VERB": {"DET": 0.5,  "NOUN": 0.3, "VERB": 0.2},
... }
>>> emit_p = {
...     "DET":  {"the": 0.7, "a": 0.3},
...     "NOUN": {"dog": 0.4, "cat": 0.3, "runs": 0.1, "walk": 0.2},
...     "VERB": {"runs": 0.5, "walk": 0.4, "sees": 0.1},
... }
>>> path, logp = viterbi(["the", "dog", "runs"], states, start_p, trans_p, emit_p)
>>> path
['DET', 'NOUN', 'VERB']
>>> round(math.exp(logp), 5)     # 0.6*0.7 * 0.9*0.4 * 0.6*0.5
0.04536
>>> viterbi(["dog", "walk", "the", "walk"], states, start_p, trans_p, emit_p)[0]
['NOUN', 'VERB', 'DET', 'NOUN']     # 同じ "walk" でも文脈で品詞が変わる
>>> viterbi([], states, start_p, trans_p, emit_p)
([], 0.0)
```

## 面接官が聞きそうなこと
- 計算量は? (系列長 T, 状態数 S) / 全経路を列挙すると?
- なぜ対数を取るのか? log(0) はどう扱った?
- 前向きアルゴリズム (Forward) との違いは? (max と sum)
- 遷移確率・出力確率はどうやって推定する? (教師ありなら頻度の最尤推定 + スムージング、教師なしなら Baum-Welch)
- 未知語の扱いを改善するには? (接尾辞・文字種などの特徴) / CRF や BiLSTM-CRF との関係は?
