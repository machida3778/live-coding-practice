def viterbi(
    observations: list[str],
    states: list[str],
    start_p: dict[str, float],
    trans_p: dict[str, dict[str, float]],
    emit_p: dict[str, dict[str, float]],
) -> tuple[list[str], float]:
    raise NotImplementedError
