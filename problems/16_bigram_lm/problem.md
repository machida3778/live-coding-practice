# 16 バイグラム言語モデルとパープレキシティ  (目標 25分 / ★★★)

単語バイグラム言語モデルを実装し、文の対数確率とパープレキシティを計算するクラス `BigramLM` を作ってください。

## 仕様
```python
class BigramLM:
    def __init__(self, k: float = 1.0): ...
    def fit(self, sentences: list[list[str]]) -> "BigramLM": ...
    def prob(self, word: str, prev: str) -> float: ...
    def sentence_logprob(self, sentence: list[str]) -> float: ...
    def perplexity(self, sentences: list[list[str]]) -> float: ...
```
- 特殊記号: 文頭 `"<s>"`、文末 `"</s>"`、未知語 `"<unk>"`
- `fit(sentences)`: 学習して **`self` を返す**
  - 各文を `["<s>"] + sentence + ["</s>"]` に拡張し、隣接する2単語の組 (バイグラム) を数える
  - 語彙 `V` = (学習データに現れる単語の集合) ∪ `{"</s>", "<unk>"}`
    - **`"<s>"` は V に含めない** (予測される対象にならないため)
- `prob(word, prev)`: add-k 平滑化した条件付き確率 `P(word | prev)`

  ```
  P(w | prev) = (C(prev, w) + k) / (C(prev) + k * |V|)
  ```
  - `C(prev, w)`: バイグラム `(prev, w)` の学習データ中の出現回数
  - `C(prev)`: `prev` で始まるバイグラムの総数 (= Σ_w C(prev, w))
  - `word` / `prev` が V に無い単語なら (`prev == "<s>"` の場合を除き) `"<unk>"` に置き換えてから計算する
- `sentence_logprob(sentence)`: 拡張した文 `<s> w1 ... wn </s>` について
  `Σ_{i=1}^{n+1} ln P(w_i | w_{i-1})` (自然対数。`</s>` の予測も含む)
- `perplexity(sentences)`: `exp( - (Σ_s sentence_logprob(s)) / M )`
  - `M` = 予測したトークンの総数 = `Σ_s (len(s) + 1)` (`</s>` は含め、`<s>` は含めない)
- `k > 0` と仮定してよい。空の文 `[]` も扱えること (`<s> </s>` になる)

テストでは浮動小数点を許容誤差 `1e-9` (相対誤差) で比較します。

## 例
```python
>>> train = [["I", "am", "Sam"],
...          ["Sam", "I", "am"],
...          ["I", "do", "not", "like", "green", "eggs", "and", "ham"]]
>>> lm = BigramLM(k=1.0).fit(train)
>>> # |V| = 学習語彙10語 + </s> + <unk> = 12
>>> lm.prob("I", "<s>")      # (2 + 1) / (3 + 12)
0.2
>>> lm.prob("Sam", "am")     # (1 + 1) / (2 + 12)
0.14285714285714285
>>> lm.prob("cat", "I")      # cat は未知語 → P(<unk> | I) = (0 + 1) / (3 + 12)
0.06666666666666667
>>> lm.sentence_logprob(["I", "am", "Sam"])   # ln(0.2) + ln(0.2) + ln(1/7) + ln(1/7)
-7.110696122978828
>>> lm.perplexity([["I", "am", "Sam"]])       # exp(7.1106.../4)
5.916079783099617
```

## 面接官が聞きそうなこと
- パープレキシティの直感的な意味は? 値が小さいほど良いのはなぜ?
- add-k 平滑化の問題点は? より良い手法は? (Kneser-Ney、バックオフ、補間)
- 学習データに `<unk>` が出現しないこの設計の問題点は? (低頻度語を `<unk>` に置換して学習する等)
- Σ_w P(w | prev) = 1 になることを確認するには? (`w` は V 全体を走る)
- ニューラル言語モデル (RNN / Transformer) と比べた n-gram の長所・短所は?
