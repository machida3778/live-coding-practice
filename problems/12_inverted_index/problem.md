# 12 転置インデックスとブール検索・フレーズ検索  (目標 20分 / ★★)

検索エンジンの基本データ構造である **転置インデックス** (単語 → その単語を含む文書) を実装し、
AND / OR / NOT 検索と **フレーズ検索** ができるクラスを作ってください。

```python
class InvertedIndex:
    def add(self, doc_id: int, text: str) -> None: ...
    def search_and(self, terms: list[str]) -> list[int]: ...
    def search_or(self, terms: list[str]) -> list[int]: ...
    def search_and_not(self, must: list[str], must_not: list[str]) -> list[int]: ...
    def search_phrase(self, phrase: str) -> list[int]: ...
```

## 仕様
- トークン化: テキストを **小文字化** してから、正規表現 `[a-z0-9]+` にマッチする部分をトークンとする
  - 例: `"Hello, World! 2024"` → `["hello", "world", "2024"]`
- クエリの語 (`terms` / `must` / `must_not` の各要素) も小文字化して扱う (各要素は1単語と仮定してよい)
- `add(doc_id, text)`: 文書を追加する。`doc_id` は重複しない整数 (追加順は昇順とは限らない)
- `search_and(terms)`: すべての語を含む文書
- `search_or(terms)`: いずれかの語を含む文書
- `search_and_not(must, must_not)`: `must` の語をすべて含み、かつ `must_not` の語を1つも含まない文書
- `search_phrase(phrase)`: `phrase` を同じ方法でトークン化した語の並びが、**連続して、その順序で** 現れる文書
  - 例: `"new york"` は `"I love New York!"` や `"new, york"` にマッチするが、`"york new"` や `"new big york"` にはマッチしない
- 戻り値はすべて **doc_id の昇順リスト (重複なし)**
- `terms` / `must` が空リストの場合、`phrase` のトークンが0個の場合は空リストを返す
- 文書数が多くても高速に動くこと (**クエリのたびに全文書を走査しない**)。非公開テストには 2万文書規模のケースがある

## 例
```python
>>> idx = InvertedIndex()
>>> idx.add(1, "New York is a big city.")
>>> idx.add(2, "I love new books. York is old.")
>>> idx.add(3, "The city of York")
>>> idx.search_and(["york", "new"])
[1, 2]
>>> idx.search_or(["books", "CITY"])
[1, 2, 3]
>>> idx.search_and_not(["york"], ["new"])
[3]
>>> idx.search_phrase("new york")      # 文書2は new と york が離れている
[1]
>>> idx.search_and(["tokyo"])
[]
```

## 面接官が聞きそうなこと
- AND 検索で複数のポスティングリストを交差するとき、どの順で交差すると速い? ソート済みリスト同士ならどう交差する?
- フレーズ検索のために何を保存した? メモリ使用量はどれくらい増える?
- 文書の削除・更新にはどう対応する?
- 日本語のように空白で区切られない言語ではトークン化をどうする? (形態素解析 / 文字 n-gram)
- 検索結果をランキングしたい (TF-IDF / BM25) なら追加で何を保存する?
