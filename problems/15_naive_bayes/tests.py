import itertools
import math
import random

TOL = 1e-9
DOCS = [["Chinese", "Beijing", "Chinese"],
        ["Chinese", "Chinese", "Shanghai"],
        ["Chinese", "Macao"],
        ["Tokyo", "Japan", "Chinese"]]
LABELS = ["china", "china", "china", "japan"]
TEST = ["Chinese", "Chinese", "Chinese", "Tokyo", "Japan"]


def _close(a, b):
    return math.isclose(a, b, rel_tol=TOL, abs_tol=TOL)


def _normalize(joint):
    m = max(joint.values())
    lse = m + math.log(sum(math.exp(v - m) for v in joint.values()))
    return {c: v - lse for c, v in joint.items()}


def _ref_joint(docs, labels, doc, alpha):
    classes = sorted(set(labels))
    vocab = {w for d in docs for w in d}
    out = {}
    for c in classes:
        cdocs = [d for d, y in zip(docs, labels) if y == c]
        total = sum(len(d) for d in cdocs)
        cnt = {}
        for d in cdocs:
            for w in d:
                cnt[w] = cnt.get(w, 0) + 1
        j = math.log(len(cdocs) / len(docs))
        for w in doc:
            if w in vocab:
                j += math.log((cnt.get(w, 0) + alpha) / (total + alpha * len(vocab)))
        out[c] = j
    return out


def _check_lp(got, exp, ctx=""):
    assert set(got) == set(exp), f"{ctx} キーが違います: {sorted(got)} vs {sorted(exp)}"
    for c in exp:
        assert _close(got[c], exp[c]), f"{ctx} class={c}: got={got[c]}, expected={exp[c]}"


def test_public_textbook_predict(sol):
    nb = sol.NaiveBayes(alpha=1.0).fit(DOCS, LABELS)
    assert nb.predict(TEST) == "china"
    assert nb.predict(["Tokyo", "Japan"]) == "japan"


def test_public_textbook_log_proba(sol):
    nb = sol.NaiveBayes(alpha=1.0).fit(DOCS, LABELS)
    joint = {"china": math.log(3 / 4) + 3 * math.log(6 / 14) + 2 * math.log(1 / 14),
             "japan": math.log(1 / 4) + 3 * math.log(2 / 9) + 2 * math.log(2 / 9)}
    _check_lp(nb.predict_log_proba(TEST), _normalize(joint))


def test_public_unknown_words_ignored(sol):
    nb = sol.NaiveBayes().fit(DOCS, LABELS)
    assert nb.predict(["Tokyo", "Japan", "Osaka"]) == "japan"
    _check_lp(nb.predict_log_proba(TEST + ["Osaka", "Kyoto"]), nb.predict_log_proba(TEST), "未知語:")


def test_hidden_fit_returns_self(sol):
    nb = sol.NaiveBayes()
    ret = nb.fit(DOCS, LABELS)
    assert ret is nb, "fit は self を返す"


def test_hidden_empty_doc_gives_prior(sol):
    nb = sol.NaiveBayes().fit(DOCS, LABELS)
    _check_lp(nb.predict_log_proba([]), {"china": math.log(3 / 4), "japan": math.log(1 / 4)}, "空文書:")
    _check_lp(nb.predict_log_proba(["Osaka"]), {"china": math.log(3 / 4), "japan": math.log(1 / 4)}, "未知語のみ:")
    assert nb.predict([]) == "china"


def test_hidden_alpha(sol):
    for alpha in (0.5, 2.0):
        nb = sol.NaiveBayes(alpha=alpha).fit(DOCS, LABELS)
        for doc in (TEST, ["Macao", "Japan"], ["Beijing"]):
            exp = _normalize(_ref_joint(DOCS, LABELS, doc, alpha))
            _check_lp(nb.predict_log_proba(doc), exp, f"alpha={alpha}, doc={doc}:")


def test_hidden_tie_break(sol):
    docs = [["the", "good"], ["the", "bad"], ["the", "ok"]]
    labels = ["pos", "neg", "mid"]
    nb = sol.NaiveBayes().fit(docs, labels)
    assert nb.predict([]) == "mid", "同点ならラベル辞書順で最小"
    assert nb.predict(["the", "the"]) == "mid"
    assert nb.predict(["unknown"]) == "mid"
    assert nb.predict(["the", "good"]) == "pos"
    assert nb.predict(["bad"]) == "neg"


def test_hidden_three_classes_unbalanced(sol):
    rng = random.Random(3)
    words = {"sports": ["goal", "match", "team", "win", "ball"],
             "tech": ["code", "python", "gpu", "model", "data"],
             "food": ["ramen", "sushi", "tasty", "eat", "rice"]}
    common = ["the", "a", "is", "today"]
    docs, labels = [], []
    for c, n in (("sports", 30), ("tech", 12), ("food", 5)):
        for _ in range(n):
            docs.append(rng.choices(words[c], k=rng.randint(2, 6)) + rng.choices(common, k=rng.randint(0, 3)))
            labels.append(c)
    nb = sol.NaiveBayes(alpha=0.7).fit(docs, labels)
    for doc in (["python", "data", "the"], ["sushi"], ["today", "is"], ["goal", "gpu", "ramen"], []):
        exp = _normalize(_ref_joint(docs, labels, doc, 0.7))
        _check_lp(nb.predict_log_proba(doc), exp, f"doc={doc}:")
        assert nb.predict(doc) == min(exp, key=lambda c: (-exp[c], c)), f"doc={doc}"


def test_hidden_long_document_stability(sol):
    nb = sol.NaiveBayes().fit(DOCS, LABELS)
    doc = ["Chinese", "Tokyo", "Japan", "Macao"] * 1000  # exp(J) はアンダーフローする長さ
    lp = nb.predict_log_proba(doc)
    assert all(math.isfinite(v) for v in lp.values()), f"-inf / nan が出ています: {lp}"
    assert _close(math.fsum(math.exp(v) for v in lp.values()), 1.0), "事後確率の和が1になりません"
    exp = _normalize(_ref_joint(DOCS, LABELS, doc, 1.0))
    _check_lp(lp, exp, "長文:")
    assert nb.predict(doc) == "japan"


def test_hidden_large(sol):
    rng = random.Random(11)
    vocab = [f"w{i}" for i in range(3000)]
    classes = ["a", "b", "c", "d"]
    # クラスごとに少しずつ異なる単語分布
    cum = {c: list(itertools.accumulate(1.0 + 5.0 * ((i + 7 * k) % 50 == 0) for i in range(len(vocab))))
           for k, c in enumerate(classes)}
    docs, labels = [], []
    for _ in range(4000):
        c = rng.choice(classes)
        docs.append(rng.choices(vocab, cum_weights=cum[c], k=30))
        labels.append(c)
    nb = sol.NaiveBayes().fit(docs, labels)
    tests = [rng.choices(vocab, k=30) + ["never_seen"] for _ in range(2000)]
    preds = [nb.predict(d) for d in tests]
    assert all(p in classes for p in preds)
    for d, p in list(zip(tests, preds))[:20]:
        exp = _normalize(_ref_joint(docs, labels, d, 1.0))
        assert p == min(exp, key=lambda c: (-exp[c], c))
        _check_lp(nb.predict_log_proba(d), exp)
