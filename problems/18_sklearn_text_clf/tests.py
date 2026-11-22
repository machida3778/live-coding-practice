import csv
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
DATA = HERE / "data.tsv"
HIDDEN = HERE / "hidden.tsv"


def _read(path):
    """テスト側の独立したローダ (解答の load_data に依存しない)"""
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))
    return [(r.get("split", "hidden"), r["label"], r["text"].strip()) for r in rows if r["text"].strip()]


def _split():
    rows = _read(DATA)
    tr = [r for r in rows if r[0] == "train"]
    te = [r for r in rows if r[0] == "test"]
    return [r[2] for r in tr], [r[1] for r in tr], [r[2] for r in te], [r[1] for r in te]


_CACHE = {}


def _trained(sol):
    if "pipe" not in _CACHE:
        _CACHE["pipe"], _CACHE["f1"] = sol.train_and_evaluate(*_split())
    return _CACHE["pipe"], _CACHE["f1"]


def test_public_load_data(sol):
    tr_x, tr_y, te_x, te_y = sol.load_data(str(DATA))
    assert (len(tr_x), len(tr_y), len(te_x), len(te_y)) == (450, 450, 150, 150), \
        (len(tr_x), len(tr_y), len(te_x), len(te_y))
    exp = _split()
    assert tr_x[0] == exp[0][0] and tr_y[0] == exp[1][0], (tr_x[0], tr_y[0])


def test_public_build_pipeline(sol):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline
    pipe = sol.build_pipeline()
    assert isinstance(pipe, Pipeline), type(pipe)
    names = [name for name, _ in pipe.steps]
    assert names == ["tfidf", "clf"], names
    assert isinstance(pipe.named_steps["tfidf"], TfidfVectorizer), type(pipe.named_steps["tfidf"])


def test_public_macro_f1_threshold(sol):
    pipe, f1 = _trained(sol)
    assert isinstance(f1, float), f"float を返してください: {type(f1)}"
    assert f1 >= 0.9, f"macro-F1 = {f1:.3f} (合格ライン 0.9)"


def test_public_top_features_format(sol):
    pipe, _ = _trained(sol)
    feats = sol.top_features(pipe, "sports", 10)
    vocab = set(pipe.named_steps["tfidf"].get_feature_names_out())
    assert isinstance(feats, list) and len(feats) == 10, feats
    assert all(isinstance(f, str) and f in vocab for f in feats), feats


def test_hidden_load_data_cleaning(sol):
    tr_x, tr_y, te_x, te_y = sol.load_data(str(DATA))
    exp = _split()
    assert tr_x == exp[0] and tr_y == exp[1], "train 部分が期待と一致しません (前後空白の除去・空行の除外・順序)"
    assert te_x == exp[2] and te_y == exp[3], "test 部分が期待と一致しません"


def test_hidden_f1_is_macro_f1_on_test(sol):
    from sklearn.metrics import f1_score
    tr_x, tr_y, te_x, te_y = _split()
    pipe, f1 = _trained(sol)
    exp = f1_score(te_y, pipe.predict(te_x), average="macro")
    assert abs(f1 - exp) < 1e-9, f"返り値 {f1} と test の macro-F1 {exp} が一致しません"


def test_hidden_generalization(sol):
    from sklearn.metrics import f1_score
    pipe, _ = _trained(sol)
    rows = _read(HIDDEN)
    x = [r[2] for r in rows]
    y = [r[1] for r in rows]
    f1 = f1_score(y, pipe.predict(x), average="macro")
    assert f1 >= 0.9, f"非公開データでの macro-F1 = {f1:.3f} (合格ライン 0.9)"


def test_hidden_top_features_correct_order(sol):
    pipe, _ = _trained(sol)
    clf = pipe.named_steps["clf"]
    names = list(pipe.named_steps["tfidf"].get_feature_names_out())
    pos = {n: i for i, n in enumerate(names)}
    for label in clf.classes_:
        coef = clf.coef_[list(clf.classes_).index(label)]
        feats = sol.top_features(pipe, label, 15)
        assert len(feats) == 15 and len(set(feats)) == 15, feats
        vals = [coef[pos[f]] for f in feats]
        assert all(a >= b - 1e-12 for a, b in zip(vals, vals[1:])), f"{label}: 係数の降順になっていません"
        rest = np.delete(coef, [pos[f] for f in feats])
        assert rest.max() <= vals[-1] + 1e-12, f"{label}: 上位 15 個に入るべき特徴量が漏れています"


def test_hidden_top_features_semantic(sol):
    pipe, _ = _trained(sol)
    keywords = ["選手", "勝利", "連勝", "監督", "試合", "出場", "優勝", "ゴール", "ホームラン", "決勝", "勝", "選", "戦"]
    feats = [f.strip() for f in sol.top_features(pipe, "sports", 20)]
    hit = [f for f in feats if f and any(f in k or k in f for k in keywords)]
    assert len(hit) >= 3, f"sports の上位特徴量にスポーツらしい語がほとんどありません: {feats}"


def test_hidden_edge_cases(sol):
    pipe, _ = _trained(sol)
    assert sol.top_features(pipe, "economy", 0) == []
    try:
        sol.top_features(pipe, "politics", 3)
    except ValueError:
        pass
    else:
        raise AssertionError("存在しないラベルで ValueError が送出されていません")
