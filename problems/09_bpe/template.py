def train_bpe(corpus: dict[str, int], num_merges: int) -> list[tuple[str, str]]:
    raise NotImplementedError


def encode(word: str, merges: list[tuple[str, str]]) -> list[str]:
    raise NotImplementedError
