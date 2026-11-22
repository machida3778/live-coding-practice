class InvertedIndex:
    def __init__(self) -> None:
        pass

    def add(self, doc_id: int, text: str) -> None:
        raise NotImplementedError

    def search_and(self, terms: list[str]) -> list[int]:
        raise NotImplementedError

    def search_or(self, terms: list[str]) -> list[int]:
        raise NotImplementedError

    def search_and_not(self, must: list[str], must_not: list[str]) -> list[int]:
        raise NotImplementedError

    def search_phrase(self, phrase: str) -> list[int]:
        raise NotImplementedError
