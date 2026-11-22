from collections import Counter


def extract_spans(tags: list[str]) -> list[tuple[str, int, int]]:
    raise NotImplementedError


def evaluate(gold: list[list[str]], pred: list[list[str]]) -> dict[str, dict]:
    raise NotImplementedError
