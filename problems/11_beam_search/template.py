from typing import Callable


def beam_search(
    next_token_logprobs: Callable[[tuple[str, ...]], dict[str, float]],
    beam_width: int,
    max_len: int,
    eos: str = "</s>",
    alpha: float = 0.0,
) -> tuple[tuple[str, ...], float]:
    raise NotImplementedError
