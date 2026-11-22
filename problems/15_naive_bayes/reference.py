import math
from collections import Counter


class NaiveBayes:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def fit(self, docs: list[list[str]], labels: list[str]) -> "NaiveBayes":
        self.classes_ = sorted(set(labels))
        n = len(docs)
        label_counts = Counter(labels)
        self.log_prior_ = {c: math.log(label_counts[c] / n) for c in self.classes_}
        counts = {c: Counter() for c in self.classes_}
        for d, y in zip(docs, labels):
            counts[y].update(d)
        self.vocab_ = set()
        for c in self.classes_:
            self.vocab_.update(counts[c])
        v = len(self.vocab_)
        self.log_likelihood_ = {}
        for c in self.classes_:
            denom = sum(counts[c].values()) + self.alpha * v
            self.log_likelihood_[c] = {
                w: math.log((counts[c][w] + self.alpha) / denom) for w in self.vocab_
            }
        return self

    def _joint_log(self, doc: list[str]) -> dict[str, float]:
        known = [w for w in doc if w in self.vocab_]
        return {
            c: self.log_prior_[c] + sum(self.log_likelihood_[c][w] for w in known)
            for c in self.classes_
        }

    def predict_log_proba(self, doc: list[str]) -> dict[str, float]:
        joint = self._joint_log(doc)
        m = max(joint.values())
        lse = m + math.log(sum(math.exp(v - m) for v in joint.values()))
        return {c: v - lse for c, v in joint.items()}

    def predict(self, doc: list[str]) -> str:
        joint = self._joint_log(doc)
        return min(self.classes_, key=lambda c: (-joint[c], c))
