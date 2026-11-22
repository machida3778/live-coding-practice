import math
from collections import Counter


class NaiveBayes:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def fit(self, docs: list[list[str]], labels: list[str]) -> "NaiveBayes":
        raise NotImplementedError

    def predict_log_proba(self, doc: list[str]) -> dict[str, float]:
        raise NotImplementedError

    def predict(self, doc: list[str]) -> str:
        raise NotImplementedError
