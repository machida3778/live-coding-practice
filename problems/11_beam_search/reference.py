from typing import Callable


def beam_search(
    next_token_logprobs: Callable[[tuple[str, ...]], dict[str, float]],
    beam_width: int,
    max_len: int,
    eos: str = "</s>",
    alpha: float = 0.0,
) -> tuple[tuple[str, ...], float]:
    beam: list[tuple[tuple[str, ...], float]] = [((), 0.0)]
    finished: list[tuple[tuple[str, ...], float]] = []
    for _ in range(max_len):
        candidates = [
            (seq + (tok,), lp + tok_lp)
            for seq, lp in beam
            for tok, tok_lp in next_token_logprobs(seq).items()
        ]
        candidates.sort(key=lambda c: (-c[1], c[0]))
        beam = []
        for seq, lp in candidates[:beam_width]:
            (finished if seq[-1] == eos else beam).append((seq, lp))
        if not beam:
            break
    finished.extend(beam)  # max_len に達した未完了の仮説も完了扱い

    def score(c: tuple[tuple[str, ...], float]) -> float:
        return c[1] / (len(c[0]) ** alpha)

    best = min(finished, key=lambda c: (-score(c), c[0]))
    return best[0], score(best)
