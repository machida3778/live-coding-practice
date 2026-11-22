# 17 BIOタグからの固有表現抽出と F1 評価  (目標 25分 / ★★)

固有表現認識 (NER) の出力は、トークンごとの **BIO タグ** で表されます。
タグ列からエンティティのスパンを取り出す関数と、正解と予測を **エンティティ単位 (完全一致)** で評価する関数を実装してください。
(seqeval 等の評価ライブラリは使用禁止)

## 仕様
### 1. `extract_spans(tags: list[str]) -> list[tuple[str, int, int]]`
- タグは `"O"`、`"B-<TYPE>"`、`"I-<TYPE>"` のいずれか
  - タイプ名に `-` が含まれることがある (例: `"B-WORK-OF-ART"` のタイプは `"WORK-OF-ART"`)
- `(タイプ, 開始index, 終了index)` のリストを開始 index の昇順で返す。**終了 index は含まない** (スライスと同じ)
- 解釈のルール (conlleval 方式)
  - `B-X`: 常に新しいタイプ X のエンティティを開始する (直前のエンティティは閉じる)
  - `I-X`: 直前のトークンが同じタイプ X のエンティティの中 (`B-X` か `I-X`) なら、それを延長する。
    そうでなければ (文頭、`O` の直後、別タイプの直後) **新しいタイプ X のエンティティを開始する**
  - `O`: 直前のエンティティを閉じる

### 2. `evaluate(gold: list[list[str]], pred: list[list[str]]) -> dict[str, dict]`
- `gold[i]` と `pred[i]` は i 番目の文の正解タグ列と予測タグ列
  - 文の数、または対応する文のタグ列の長さが一致しない場合は `ValueError` を送出する
- 予測スパンは、**同じ文** の正解スパンに `(タイプ, 開始, 終了)` が完全一致するものがあるときだけ正解 (TP)
- タイプ X について
  - `precision = TP(X) / 予測スパン数(X)`、`recall = TP(X) / 正解スパン数(X)`
  - `f1 = 2 * precision * recall / (precision + recall)`
  - 分母が 0 になる場合、その値は `0.0`
- 戻り値の辞書
  - キー `"micro"`: 全タイプの TP・予測数・正解数を合計して計算した値
  - 正解または予測に1回以上現れた各タイプ名をキーとする値
  - 各値は `{"precision": float, "recall": float, "f1": float, "support": int}` (`support` は正解スパン数)

テストでは浮動小数点を許容誤差 `1e-9` で比較します。

## 例
```python
>>> extract_spans(["B-PER", "I-PER", "O", "B-LOC"])
[('PER', 0, 2), ('LOC', 3, 4)]
>>> extract_spans(["B-PER", "I-PER", "O", "I-LOC", "I-LOC", "B-LOC", "I-PER"])
[('PER', 0, 2), ('LOC', 3, 5), ('LOC', 5, 6), ('PER', 6, 7)]
>>> gold = [["B-PER", "I-PER", "O", "B-LOC"], ["B-ORG", "O"]]
>>> pred = [["B-PER", "I-PER", "O", "B-ORG"], ["B-ORG", "O"]]
>>> r = evaluate(gold, pred)
>>> r["micro"]
{'precision': 0.6666666666666666, 'recall': 0.6666666666666666, 'f1': 0.6666666666666666, 'support': 3}
>>> r["PER"]
{'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'support': 1}
>>> r["LOC"]
{'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'support': 1}
>>> r["ORG"]
{'precision': 0.5, 'recall': 1.0, 'f1': 0.6666666666666666, 'support': 1}
```

## 面接官が聞きそうなこと
- トークン単位の正解率ではなくエンティティ単位の F1 で評価するのはなぜ?
- micro 平均と macro 平均の違いは? クラス不均衡があるときどちらを見る?
- 部分一致 (境界が1トークンずれた) を評価したい場合はどうする?
- BIO 以外のタグ付け方式 (BIOES など) の利点は? 予測が不正な系列 (`O` の直後の `I-X`) を出さないようにするには? (CRF、制約付きデコード)
