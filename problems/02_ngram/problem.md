# 02 n-gram の生成と頻度  (目標 10分 / ★)

n-gram は言語モデルや特徴量抽出の基本です。次の3つの関数を実装してください。

## 仕様
### `word_ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]`
- トークン列から単語 n-gram を **出現順に** 生成する。各 n-gram は **タプル** で返す
- `len(tokens) < n` なら空リスト

### `char_ngrams(text: str, n: int, pad: bool = False) -> list[str]`
- 文字列から文字 n-gram を出現順に生成する
- `pad=True` のときは、先頭に `"^"` を **n-1 個**、末尾に `"$"` を **n-1 個** 付けてから切り出す
- 長さが足りなければ空リスト

### `ngram_counts(sentences: list[list[str]], n: int, min_count: int = 1) -> dict[tuple[str, ...], int]`
- 文 (トークン列) のリストから、単語 n-gram の出現回数を数える
- **n-gram は文をまたがない** (文ごとに n-gram を作る)
- 出現回数が `min_count` 以上のものだけを含む辞書を返す (`collections.Counter` を返してもよい)

### 共通
- `n < 1` のときは3関数とも `ValueError` を送出する

## 例
```python
>>> word_ngrams(["I", "love", "NLP"], 2)
[('I', 'love'), ('love', 'NLP')]
>>> char_ngrams("自然言語", 2)
['自然', '然言', '言語']
>>> char_ngrams("abc", 2, pad=True)
['^a', 'ab', 'bc', 'c$']
>>> ngram_counts([["a", "b", "a", "b"], ["b", "a"]], 2)
{('a', 'b'): 2, ('b', 'a'): 2}
```
(最後の例で、文をまたいだ `('b', 'b')` は数えません)

## 面接官が聞きそうなこと
- n-gram の種類数が膨大になったらどう扱う? (語彙の打ち切り, feature hashing など)
- 分かち書きされない日本語で文字 n-gram が有用なのはなぜ?
- パディング記号を付ける意味は? (語頭・語末の情報)
