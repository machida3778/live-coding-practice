import math
import random

TOL = 1e-9


def _close(a, b):
    return math.isclose(a, b, rel_tol=TOL, abs_tol=TOL)


def _check_metrics(got, exp, ctx=""):
    for key in ("precision", "recall", "f1"):
        assert _close(got[key], exp[key]), f"{ctx} {key}: got={got[key]}, expected={exp[key]}"
    assert got["support"] == exp["support"], f"{ctx} support: got={got['support']}, expected={exp['support']}"


def _m(p, r, s):
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f, "support": s}


def test_public_extract_spans(sol):
    got = sol.extract_spans(["B-PER", "I-PER", "O", "B-LOC"])
    assert got == [("PER", 0, 2), ("LOC", 3, 4)], got
    got = sol.extract_spans(["B-PER", "I-PER", "O", "I-LOC", "I-LOC", "B-LOC", "I-PER"])
    assert got == [("PER", 0, 2), ("LOC", 3, 5), ("LOC", 5, 6), ("PER", 6, 7)], got


def test_public_evaluate(sol):
    gold = [["B-PER", "I-PER", "O", "B-LOC"], ["B-ORG", "O"]]
    pred = [["B-PER", "I-PER", "O", "B-ORG"], ["B-ORG", "O"]]
    r = sol.evaluate(gold, pred)
    assert set(r) == {"micro", "PER", "LOC", "ORG"}, f"keys = {sorted(r)}"
    _check_metrics(r["micro"], _m(2 / 3, 2 / 3, 3), "micro")
    _check_metrics(r["PER"], _m(1.0, 1.0, 1), "PER")
    _check_metrics(r["LOC"], _m(0.0, 0.0, 1), "LOC")
    _check_metrics(r["ORG"], _m(0.5, 1.0, 1), "ORG")


def test_hidden_edge_spans(sol):
    assert sol.extract_spans([]) == []
    assert sol.extract_spans(["O", "O"]) == []
    assert sol.extract_spans(["I-PER"]) == [("PER", 0, 1)], "文頭の I- は新しいエンティティ"
    assert sol.extract_spans(["B-PER", "B-PER"]) == [("PER", 0, 1), ("PER", 1, 2)], "連続する B- は別エンティティ"
    assert sol.extract_spans(["B-PER", "I-LOC", "I-LOC"]) == [("PER", 0, 1), ("LOC", 1, 3)], "別タイプの I- は新しいエンティティ"
    assert sol.extract_spans(["O", "B-LOC", "I-LOC", "I-LOC"]) == [("LOC", 1, 4)], "文末で閉じ忘れない"
    assert sol.extract_spans(["I-ORG", "I-ORG", "O", "I-ORG"]) == [("ORG", 0, 2), ("ORG", 3, 4)]


def test_hidden_type_with_hyphen(sol):
    got = sol.extract_spans(["B-WORK-OF-ART", "I-WORK-OF-ART", "O", "I-WORK-OF-ART", "B-E-MAIL"])
    assert got == [("WORK-OF-ART", 0, 2), ("WORK-OF-ART", 3, 4), ("E-MAIL", 4, 5)], got


def test_hidden_boundary_mismatch(sol):
    gold = [["B-PER", "I-PER", "I-PER", "O"]]
    pred = [["B-PER", "I-PER", "O", "O"]]
    r = sol.evaluate(gold, pred)
    _check_metrics(r["micro"], _m(0.0, 0.0, 1), "境界のずれは不正解")
    _check_metrics(r["PER"], _m(0.0, 0.0, 1))


def test_hidden_same_span_different_sentence(sol):
    # 文が違えば同じ (type, start, end) でも一致しない
    gold = [["B-PER", "O"], ["O", "O"]]
    pred = [["O", "O"], ["B-PER", "O"]]
    r = sol.evaluate(gold, pred)
    _check_metrics(r["micro"], _m(0.0, 0.0, 1), "文ごとに照合すること")


def test_hidden_no_entities(sol):
    r = sol.evaluate([["O", "O"], []], [["O", "O"], []])
    assert set(r) == {"micro"}, f"エンティティが無ければ micro のみ: {sorted(r)}"
    _check_metrics(r["micro"], _m(0.0, 0.0, 0), "分母0は0.0")


def test_hidden_pred_only_type_and_perfect(sol):
    gold = [["B-LOC", "O", "B-PER"], ["B-DATE", "I-DATE", "O"]]
    pred = [["B-LOC", "B-MISC", "B-PER"], ["B-DATE", "I-DATE", "O"]]
    r = sol.evaluate(gold, pred)
    assert set(r) == {"micro", "LOC", "PER", "DATE", "MISC"}, sorted(r)
    _check_metrics(r["MISC"], _m(0.0, 0.0, 0), "MISC")
    _check_metrics(r["DATE"], _m(1.0, 1.0, 1), "DATE")
    _check_metrics(r["micro"], _m(3 / 4, 1.0, 3), "micro")


def test_hidden_length_mismatch(sol):
    for gold, pred in (([["O"]], [["O"], ["O"]]), ([["O", "B-PER"]], [["O"]])):
        try:
            sol.evaluate(gold, pred)
        except ValueError:
            continue
        raise AssertionError("長さが一致しない場合は ValueError を送出すること")


def _ref_spans(tags):
    spans, cur, start = [], None, 0
    for i, t in enumerate(tags + ["O"]):
        if t == "O":
            p, ty = "O", None
        else:
            p, ty = t.split("-", 1)
        if cur is not None and (p != "I" or ty != cur):
            spans.append((cur, start, i))
            cur = None
        if p == "B" or (p == "I" and cur is None):
            cur, start = ty, i
    return spans


def test_hidden_large(sol):
    rng = random.Random(17)
    types = ["PER", "LOC", "ORG", "MISC"]
    tagset = ["O"] * 6 + [f"{p}-{t}" for p in "BI" for t in types]
    gold, pred = [], []
    for _ in range(20000):
        n = rng.randint(0, 40)
        g = [rng.choice(tagset) for _ in range(n)]
        p = [t if rng.random() < 0.85 else rng.choice(tagset) for t in g]
        gold.append(g)
        pred.append(p)
    r = sol.evaluate(gold, pred)
    tp, npred, ngold = {}, {}, {}
    for g, p in zip(gold, pred):
        gs, ps = set(_ref_spans(g)), set(_ref_spans(p))
        for t, _, _ in gs:
            ngold[t] = ngold.get(t, 0) + 1
        for t, _, _ in ps:
            npred[t] = npred.get(t, 0) + 1
        for t, _, _ in gs & ps:
            tp[t] = tp.get(t, 0) + 1
    T, P, G = sum(tp.values()), sum(npred.values()), sum(ngold.values())
    _check_metrics(r["micro"], _m(T / P, T / G, G), "micro")
    for t in types:
        _check_metrics(r[t], _m(tp.get(t, 0) / npred[t], tp.get(t, 0) / ngold[t], ngold[t]), t)
