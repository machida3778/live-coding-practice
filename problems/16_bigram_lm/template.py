import math
from collections import Counter, defaultdict

BOS, EOS, UNK = "<s>", "</s>", "<unk>"


class BigramLM:
    def __init__(self, k: float = 1.0):
        self.k = k

    def fit(self, sentences: list[list[str]]) -> "BigramLM":
        raise NotImplementedError

    def prob(self, word: str, prev: str) -> float:
        raise NotImplementedError

    def sentence_logprob(self, sentence: list[str]) -> float:
        raise NotImplementedError

    def perplexity(self, sentences: list[list[str]]) -> float:
        raise NotImplementedError
