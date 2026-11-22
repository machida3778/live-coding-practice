import pandas as pd


def clean(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["text"] = out["text"].str.strip()
    out = out.dropna(subset=["text", "rating"])
    out = out[out["text"] != ""]
    out = out.drop_duplicates(keep="first")
    out["date"] = pd.to_datetime(out["date"])
    out["rating"] = out["rating"].astype("int64")
    return out.reset_index(drop=True)


def product_summary(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.assign(length=df["text"].str.len())
        .groupby("product")
        .agg(n_reviews=("rating", "size"), mean_rating=("rating", "mean"), mean_length=("length", "mean"))
        .reset_index()
        .sort_values(["n_reviews", "product"], ascending=[False, True])
        .set_index("product")
    )
    return out.astype({"mean_rating": "float64", "mean_length": "float64"})


def monthly_counts(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series([], dtype="int64", name="count", index=pd.Index([], dtype=object, name="month"))
    months = df["date"].dt.to_period("M")
    full = pd.period_range(months.min(), months.max(), freq="M")
    s = months.value_counts().reindex(full, fill_value=0)
    s.index = s.index.strftime("%Y-%m")
    s.index.name = "month"
    s.name = "count"
    return s.astype("int64")


def sentiment_words(df: pd.DataFrame, n: int, stopwords: frozenset[str] = frozenset()) -> dict[str, list[tuple[str, int]]]:
    stop = set(stopwords)

    def top(sub: pd.DataFrame) -> list[tuple[str, int]]:
        if n <= 0 or sub.empty:
            return []
        word_sets = sub["text"].str.lower().str.findall(r"[a-z]+").apply(lambda ws: set(ws) - stop)
        counts = word_sets.explode().dropna().value_counts()
        items = sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:n]
        return [(w, int(c)) for w, c in items]

    return {"positive": top(df[df["rating"] >= 4]), "negative": top(df[df["rating"] <= 2])}
