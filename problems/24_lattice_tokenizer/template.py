UNK_POS = "未知語"


# ---- Part 1
def load_dictionary(path: str) -> dict[str, list[tuple[str, int]]]:
    raise NotImplementedError


def load_connection(path: str) -> dict[tuple[str, str], int]:
    raise NotImplementedError


class LatticeTokenizer:
    def __init__(self, dictionary: dict[str, list[tuple[str, int]]],
                 connection: dict[tuple[str, str], int], unk_cost: int = 10000):
        raise NotImplementedError

    # ---- Part 2
    def lookup(self, text: str, i: int) -> list[tuple[str, str, int]]:
        raise NotImplementedError

    # ---- Part 3
    def tokenize(self, text: str) -> list[tuple[str, str]]:
        raise NotImplementedError

    def best_cost(self, text: str) -> int:
        raise NotImplementedError
