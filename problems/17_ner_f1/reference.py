from collections import Counter


def extract_spans(tags: list[str]) -> list[tuple[str, int, int]]:
    spans = []
    cur_type, start = None, 0
    for i, tag in enumerate(tags):
        if tag == "O":
            if cur_type is not None:
                spans.append((cur_type, start, i))
            cur_type = None
            continue
        prefix, typ = tag.split("-", 1)
        if prefix == "B" or typ != cur_type:  # B-X, または続きになれない I-X
            if cur_type is not None:
                spans.append((cur_type, start, i))
            cur_type, start = typ, i
    if cur_type is not None:
        spans.append((cur_type, start, len(tags)))
    return spans


def _prf(tp: int, n_pred: int, n_gold: int) -> dict:
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gold if n_gold else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f, "support": n_gold}


def evaluate(gold: list[list[str]], pred: list[list[str]]) -> dict[str, dict]:
    if len(gold) != len(pred):
        raise ValueError("文の数が一致しません")
    tp, n_pred, n_gold = Counter(), Counter(), Counter()
    for g, p in zip(gold, pred):
        if len(g) != len(p):
            raise ValueError("タグ列の長さが一致しません")
        gs, ps = set(extract_spans(g)), set(extract_spans(p))
        n_gold.update(t for t, _, _ in gs)
        n_pred.update(t for t, _, _ in ps)
        tp.update(t for t, _, _ in gs & ps)
    result = {"micro": _prf(sum(tp.values()), sum(n_pred.values()), sum(n_gold.values()))}
    for t in sorted(set(n_gold) | set(n_pred)):
        result[t] = _prf(tp[t], n_pred[t], n_gold[t])
    return result
