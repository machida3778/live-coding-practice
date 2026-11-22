import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline


def load_data(path: str) -> tuple[list[str], list[str], list[str], list[str]]:
    # keep_default_na=False: 空文字を NaN にせず "" のまま読む
    df = pd.read_csv(path, sep="\t", keep_default_na=False, dtype=str)
    df["text"] = df["text"].str.strip()
    df = df[df["text"] != ""]
    train = df[df["split"] == "train"]
    test = df[df["split"] == "test"]
    return train["text"].tolist(), train["label"].tolist(), test["text"].tolist(), test["label"].tolist()


def build_pipeline() -> Pipeline:
    # 日本語は空白区切りでないため、文字 n-gram を特徴量にする
    return Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(2, 3), sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=1000)),
    ])


def train_and_evaluate(train_texts: list[str], train_labels: list[str],
                       test_texts: list[str], test_labels: list[str]) -> tuple[Pipeline, float]:
    pipe = build_pipeline()
    pipe.fit(train_texts, train_labels)
    pred = pipe.predict(test_texts)
    return pipe, float(f1_score(test_labels, pred, average="macro"))


def top_features(pipeline: Pipeline, label: str, n: int) -> list[str]:
    clf = pipeline.named_steps["clf"]
    classes = list(clf.classes_)
    if label not in classes:
        raise ValueError(f"unknown label: {label}")
    if n <= 0:
        return []
    names = pipeline.named_steps["tfidf"].get_feature_names_out()
    coef = clf.coef_[classes.index(label)]
    # 係数の降順、同値なら特徴量名の昇順 (lexsort は最後のキーが第1キー)
    order = np.lexsort((names, -coef))
    return [str(names[i]) for i in order[:n]]
