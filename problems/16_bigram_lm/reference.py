import math
from collections import Counter, defaultdict

BOS, EOS, UNK = "<s>", "</s>", "<unk>"


class BigramLM:
    def __init__(self, k: float = 1.0):
        self.k = k

    def fit(self, sentences: list[list[str]]) -> "BigramLM":
        self.bigram: dict[str, Counter] = defaultdict(Counter)
        self.context: Counter = Counter()
        vocab = {EOS, UNK}
        for s in sentences:
            vocab.update(s)
            toks = [BOS] + list(s) + [EOS]
            for a, b in zip(toks, toks[1:]):
                self.bigram[a][b] += 1
                self.context[a] += 1
        self.vocab = vocab
        return self

    def _map(self, w: str) -> str:
        if w == BOS or w in self.vocab:
            return w
        return UNK

    def prob(self, word: str, prev: str) -> float:
        word, prev = self._map(word), self._map(prev)
        c = self.bigram[prev][word] if prev in self.bigram else 0
        return (c + self.k) / (self.context[prev] + self.k * len(self.vocab))

    def sentence_logprob(self, sentence: list[str]) -> float:
        toks = [BOS] + list(sentence) + [EOS]
        return sum(math.log(self.prob(b, a)) for a, b in zip(toks, toks[1:]))

    def perplexity(self, sentences: list[list[str]]) -> float:
        total, m = 0.0, 0
        for s in sentences:
            total += self.sentence_logprob(s)
            m += len(s) + 1
        return math.exp(-total / m)
