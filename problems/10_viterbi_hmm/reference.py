import math

NEG_INF = float("-inf")
UNK_EMIT = 1e-6


def _log(p: float) -> float:
    return math.log(p) if p > 0 else NEG_INF


def viterbi(
    observations: list[str],
    states: list[str],
    start_p: dict[str, float],
    trans_p: dict[str, dict[str, float]],
    emit_p: dict[str, dict[str, float]],
) -> tuple[list[str], float]:
    if not observations:
        return [], 0.0

    def log_emit(s: str, w: str) -> float:
        return _log(emit_p[s].get(w, UNK_EMIT))

    # delta[s] = 時刻 t に状態 s にいる経路の最大対数確率
    delta = {s: _log(start_p.get(s, 0.0)) + log_emit(s, observations[0]) for s in states}
    backptrs: list[dict[str, str]] = []
    for w in observations[1:]:
        new_delta, bp = {}, {}
        for s in states:
            best_prev, best_score = states[0], NEG_INF
            for prev in states:  # states の順に見て、厳密に大きいときだけ更新 → 同点は先の状態
                score = delta[prev] + _log(trans_p[prev].get(s, 0.0))
                if score > best_score:
                    best_prev, best_score = prev, score
            new_delta[s] = best_score + log_emit(s, w)
            bp[s] = best_prev
        delta = new_delta
        backptrs.append(bp)

    last = max(states, key=lambda s: delta[s])  # max は最初の最大要素を返す
    if delta[last] == NEG_INF:
        return [], NEG_INF
    path = [last]
    for bp in reversed(backptrs):
        path.append(bp[path[-1]])
    path.reverse()
    return path, delta[last]
