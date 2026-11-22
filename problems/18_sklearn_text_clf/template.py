from sklearn.pipeline import Pipeline


def load_data(path: str) -> tuple[list[str], list[str], list[str], list[str]]:
    raise NotImplementedError


def build_pipeline() -> Pipeline:
    raise NotImplementedError


def train_and_evaluate(train_texts: list[str], train_labels: list[str],
                       test_texts: list[str], test_labels: list[str]) -> tuple[Pipeline, float]:
    raise NotImplementedError


def top_features(pipeline: Pipeline, label: str, n: int) -> list[str]:
    raise NotImplementedError
