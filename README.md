# NLP ライブコーディング対策ハーネス (Python)

## 基本の流れ
```bash
python harness.py list            # 問題一覧と成績
python harness.py start           # 次の問題を出題 → タイマー開始 (ID指定: start 09 / カテゴリ指定: start --category C / ランダム: --random)
#   → workspace/<番号>_<問題>/solution.py を編集
python harness.py test            # 公開テスト (何度でも実行可。タイマーは止まらない)
python harness.py status          # 経過時間
python harness.py submit          # 提出: タイマー停止 + 非公開テスト込みで自動採点
```
提出したら Claude に **「採点して」** と言えば、コードレビュー (review.md) と品質点が付き、次の問題に進みます。
`python harness.py stats` でカテゴリ別の通過率や目標時間比を確認できます。

## 採点 (100点)
- テスト 60: 公開+非公開テストの通過率
- 時間 20: 目標時間以内で満点、目標の2倍で0点
- コード品質 20: Claude のレビュー (正しさ・計算量・可読性・Pythonらしさ)
- ヒント1回 -3点 (Claude にヒントを頼むと記録されます)

## 出題範囲
| カテゴリ | 内容 |
|---|---|
| A Python基礎・文字列処理 | 頻度集計, n-gram, 正規化, 正規表現, 日本語文分割 |
| B アルゴリズム | 編集距離, 最長一致分割, ROUGE-L, BPE, Viterbi, ビームサーチ, 転置インデックス |
| C 古典NLP・情報検索 | TF-IDF, BM25, ナイーブベイズ, n-gram言語モデル, NER評価 |
| D ライブラリ活用 | sklearn分類, pandas集計, numpy Attention, 単語ベクトル, 重複検出 |
| E 模擬本番 (60分) | 検索エンジン総合問題, ラティス形態素解析 |

`problems/*/reference.py` は参考解答、`tests.py` の `test_hidden_*` は非公開テストなので、解く前には見ないでください。
