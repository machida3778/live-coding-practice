# 15 多項ナイーブベイズ分類器  (目標 25分 / ★★★)

トークン化済み文書のテキスト分類器として **多項ナイーブベイズ (Multinomial Naive Bayes)** をスクラッチで実装してください。
(sklearn の `MultinomialNB` は使用禁止。標準ライブラリ・numpy は使用可)

## 仕様
```python
class NaiveBayes:
    def __init__(self, alpha: float = 1.0): ...
    def fit(self, docs: list[list[str]], labels: list[str]) -> "NaiveBayes": ...
    def predict_log_proba(self, doc: list[str]) -> dict[str, float]: ...
    def predict(self, doc: list[str]) -> str: ...
```
- `alpha > 0` はラプラス (加算) 平滑化のパラメータ
- `fit(docs, labels)`: 学習して **`self` を返す** (`NaiveBayes().fit(X, y).predict(d)` と書けるように)
  - クラス集合 `C` = labels に現れるラベル全体
  - 事前確率 `P(c) = (ラベル c の文書数) / (全文書数)` (事前確率は平滑化しない)
  - 語彙 `V` = 学習データ全体に現れる単語の集合 (全クラス共通)
  - `count(w, c)` = クラス c の全文書における単語 w の出現回数の合計
  - 尤度 `P(w | c) = (count(w, c) + alpha) / (Σ_{w' ∈ V} count(w', c) + alpha * |V|)`
- 文書 `d` に対するクラス c の同時対数確率:
  `J(c) = ln P(c) + Σ_{w ∈ d, w ∈ V} ln P(w | c)`
  - 同じ単語が複数回出現すれば、その回数だけ加える
  - **V に無い単語 (未知語) は無視する**
- `predict_log_proba(doc)`: 全クラスについて **正規化された** 対数事後確率 `ln P(c | d)` を返す
  - `ln P(c | d) = J(c) - ln Σ_{c'} exp(J(c'))`
  - 長い文書では `exp(J(c))` がアンダーフローして 0 になるので、数値的に安定な計算 (log-sum-exp) をすること
- `predict(doc)`: `J(c)` が最大のクラスを返す。同点の場合は **ラベルの辞書順で最小** のもの

テストでは浮動小数点を許容誤差 `1e-9` で比較します。

## 例 (IR の教科書でおなじみの例)
```python
>>> docs = [["Chinese", "Beijing", "Chinese"],
...         ["Chinese", "Chinese", "Shanghai"],
...         ["Chinese", "Macao"],
...         ["Tokyo", "Japan", "Chinese"]]
>>> labels = ["china", "china", "china", "japan"]
>>> nb = NaiveBayes(alpha=1.0).fit(docs, labels)
>>> test = ["Chinese", "Chinese", "Chinese", "Tokyo", "Japan"]
>>> nb.predict(test)
'china'
>>> lp = nb.predict_log_proba(test)
>>> {c: round(v, 4) for c, v in lp.items()}
{'china': -0.3714, 'japan': -1.1704}
>>> # 検算: J(china) = ln(3/4) + 3 ln(6/14) + 2 ln(1/14),  J(japan) = ln(1/4) + 3 ln(2/9) + 2 ln(2/9)
>>> nb.predict(["Tokyo", "Japan", "Osaka"])    # Osaka は未知語なので無視
'japan'
```

## 面接官が聞きそうなこと
- なぜ確率の積ではなく対数の和で計算する? log-sum-exp はなぜ安定?
- 平滑化しないと何が起きる? `alpha` を大きくするとどうなる?
- 「ナイーブ」な仮定とは何か? その仮定が崩れる例は?
- ロジスティック回帰と比べた長所・短所は? (生成モデル vs 識別モデル、少量データでの強さ)
- 未知語を無視するのではなく、`<unk>` として扱う設計にするとしたら?
