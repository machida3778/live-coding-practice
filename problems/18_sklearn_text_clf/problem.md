# 18 sklearnで日本語ニュース分類  (目標 25分 / ★★)

日本語ニュース記事を 3 クラス (`sports` / `economy` / `technology`) に分類するモデルを
scikit-learn で作ります。データは `data.tsv` (解答ファイルと同じディレクトリにあります)。

## データ形式 (`data.tsv`)
- タブ区切り・1行目はヘッダ `split	label	text`
- `split` は `train` か `test`
- 前後に余計な空白が付いた text や、**空 (または空白のみ) の text** の行が混ざっています

## 実装する関数

### 1. `load_data(path) -> (train_texts, train_labels, test_texts, test_labels)`
- 4 つとも `list[str]`。ファイル中の出現順を保つ
- text の前後の空白は除去する。除去後に空になった行は捨てる

### 2. `build_pipeline() -> sklearn.pipeline.Pipeline`
- ステップ名は必ず `"tfidf"` (`TfidfVectorizer`) と `"clf"` (分類器) の 2 つ
- 分類器は `coef_` 属性 (形状 `(クラス数, 特徴量数)`) を持つ線形モデル
  (`LogisticRegression`, `LinearSVC`, `SGDClassifier` など)
- 未学習の Pipeline を返す。パラメータは自由に工夫してよい

### 3. `train_and_evaluate(train_texts, train_labels, test_texts, test_labels) -> (pipeline, macro_f1)`
- `build_pipeline()` で作った Pipeline を train で学習し、test の **macro-F1** (float) を計算して
  学習済み Pipeline と一緒に返す
- **合格ライン: macro-F1 >= 0.9** (別に用意した非公開データでも 0.9 以上が必要)

### 4. `top_features(pipeline, label, n) -> list[str]`
- 学習済み Pipeline について、クラス `label` の係数 (`coef_`) が **大きい順** に特徴量名を n 個返す
  - 特徴量名は `TfidfVectorizer.get_feature_names_out()` のもの
  - 係数が同じ場合は特徴量名の昇順。`n <= 0` なら空リスト
- `label` が学習したクラスに無ければ `ValueError`

## 例
```python
>>> tr_x, tr_y, te_x, te_y = load_data("data.tsv")
>>> len(tr_x), len(te_x)
(450, 150)
>>> pipe, f1 = train_and_evaluate(tr_x, tr_y, te_x, te_y)
>>> f1 >= 0.9
True
>>> top_features(pipe, "sports", 5)   # 設定次第で変わる
['選手', '勝利', ...]
```

## ヒントになりそうな注意
- 日本語は単語の間に空白がありません。`TfidfVectorizer()` のデフォルト設定で何がトークンになるか確認してみましょう

## 面接官が聞きそうなこと
- `TfidfVectorizer` のデフォルトの `token_pattern` だと日本語で何が起きる? 形態素解析器を使う場合との比較は?
- accuracy ではなく macro-F1 を見るのはなぜ? クラス不均衡があると何が問題?
- `fit` を test データに対して呼んではいけない理由 (データリーク) は?
- ハイパーパラメータはどう決める? (`GridSearchCV`, 交差検証)
