# 06 編集距離とスペル訂正  (目標 20分 / ★★)

検索クエリのタイポ訂正や表記ゆれ検出で使われる **編集距離** を実装し、
それを使って簡単なスペル訂正候補の提示を行ってください。

## 仕様
### `levenshtein(a: str, b: str) -> int`
- 文字列 `a` を `b` に変換するのに必要な最小の編集回数 (レーベンシュタイン距離) を返す
- 操作は **挿入・削除・置換** の3種類で、コストはいずれも 1
- 長さ 1000 程度の文字列同士でも数秒以内に計算できること (再帰の深さ制限に注意)

### `suggest(word: str, vocab: list[str], k: int, max_distance: int | None = None) -> list[tuple[str, int]]`
- 語彙 `vocab` の中から `word` に近い単語を最大 `k` 個、`(単語, 距離)` のタプルのリストで返す
- 並び順は距離の **昇順**、同じ距離なら単語の **辞書順 (昇順)**
- `vocab` に重複があっても、結果に同じ単語は1回だけ
- `max_distance` が指定されたら、距離が `max_distance` **以下** のものだけを返す
- `word` 自身が `vocab` にあれば距離 0 として含める
- `k <= 0` なら空リスト

## 例
```python
>>> levenshtein("kitten", "sitting")
3
>>> levenshtein("", "abc")
3
>>> vocab = ["spelling", "spewing", "sapling", "peeling", "spell"]
>>> suggest("speling", vocab, 3)
[('spelling', 1), ('spewing', 1), ('peeling', 2)]
>>> suggest("speling", vocab, 10, max_distance=1)
[('spelling', 1), ('spewing', 1)]
```

## 面接官が聞きそうなこと
- DP テーブル `dp[i][j]` は何を表す? 漸化式を説明して
- 計算量は? メモリを O(min(|a|, |b|)) にするには?
- 語彙が100万語あるとき `suggest` を速くするには? (長さの差による枝刈り, BK-tree, SymSpell, n-gram インデックス)
- 隣接文字の入れ替え (`teh` → `the`) も1操作と数えたい場合は? (Damerau-Levenshtein)
- 日本語の表記ゆれ (`サーバー` / `サーバ`) に使うときの注意点は?
