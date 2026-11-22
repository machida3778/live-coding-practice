# 08 ROUGE-L (LCSベースの要約評価指標)  (目標 20分 / ★★)

要約や翻訳の自動評価に使われる **ROUGE-L** を実装してください。
ROUGE-L は候補文 (candidate) と参照文 (reference) の **最長共通部分列 (LCS: Longest Common Subsequence)** の長さに基づく指標です。

次の2つの関数を実装してください。

```python
def lcs_length(a: list[str], b: list[str]) -> int
def rouge_l(candidate: list[str], reference: list[str], beta: float = 1.0) -> tuple[float, float, float]
```

## 仕様
### `lcs_length(a, b)`
- トークン列 `a`, `b` の最長共通部分列の長さを返す
- 部分列は **連続していなくてよい** (順序だけ保てばよい)。部分文字列 (連続) ではない点に注意
  - 例: `["a","x","b","y","c"]` と `["a","b","c"]` の LCS は `["a","b","c"]` で長さ 3
- どちらかが空なら 0

### `rouge_l(candidate, reference, beta=1.0)`
- `L = lcs_length(candidate, reference)` として
  - 適合率 `P = L / len(candidate)`
  - 再現率 `R = L / len(reference)`
  - F値 `F = (1 + beta**2) * P * R / (R + beta**2 * P)`  (beta=1 なら通常の F1)
- 戻り値は `(P, R, F)` の float のタプル
- `candidate` か `reference` のどちらかが空、または `L == 0` の場合は `(0.0, 0.0, 0.0)` を返す (**ゼロ除算で落ちないこと**)
- 入力はトークン化済みのリスト。トークンは完全一致で比較する (大文字小文字の正規化などは不要)
- テストでは浮動小数点は誤差 1e-9 まで許容して比較する

## 例
```python
>>> lcs_length(list("ABCBDAB"), list("BDCABA"))
4
>>> rouge_l("the cat sat on the mat".split(), "the cat is on the mat".split())   # LCS = the cat on the mat
(0.8333333333333334, 0.8333333333333334, 0.8333333333333334)
>>> rouge_l("police killed the gunman".split(), "the gunman police killed".split())   # 語順が違う
(0.5, 0.5, 0.5)
>>> rouge_l(["a", "b"], ["a", "b", "c", "d"], beta=2.0)   # 再現率を重視
(1.0, 0.5, 0.5555555555555556)
>>> rouge_l([], ["a"])
(0.0, 0.0, 0.0)
```

## 面接官が聞きそうなこと
- 時間計算量・空間計算量は? 空間を O(min(n, m)) に減らすには?
- LCS の長さだけでなく、部分列そのものを復元するには?
- BLEU (n-gram の適合率) と比べた ROUGE-L の長所・短所は? (言い換え・同義語には弱い 等)
- 日本語の要約を評価するならトークン化をどうする? (文字単位 / 形態素単位)
