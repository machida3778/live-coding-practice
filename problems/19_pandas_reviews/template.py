import pandas as pd


def clean(df: pd.DataFrame) -> pd.DataFrame:
    raise NotImplementedError


def product_summary(df: pd.DataFrame) -> pd.DataFrame:
    raise NotImplementedError


def monthly_counts(df: pd.DataFrame) -> pd.Series:
    raise NotImplementedError


def sentiment_words(df: pd.DataFrame, n: int, stopwords: frozenset[str] = frozenset()) -> dict[str, list[tuple[str, int]]]:
    raise NotImplementedError
