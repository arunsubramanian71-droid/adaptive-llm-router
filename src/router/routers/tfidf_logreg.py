from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from router.routers.base import Router, validate_binary_labels


class TfidfLogisticRegressionRouter(Router):
    name = "tfidf_logreg"

    def __init__(self, **logreg_kwargs) -> None:
        logreg_kwargs.setdefault("max_iter", 1000)
        self._pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer(min_df=1, ngram_range=(1, 2))),
                ("clf", LogisticRegression(**logreg_kwargs)),
            ]
        )
        self._fitted = False

    def fit(self, prompts: list[str], labels: list[int]) -> None:
        validate_binary_labels(labels)
        self._pipeline.fit(prompts, labels)
        self._fitted = True

    def predict_proba(self, prompts: list[str]) -> list[float]:
        if not self._fitted:
            raise RuntimeError(f"{self.name} router has not been fit yet")
        proba = self._pipeline.predict_proba(prompts)
        idx = list(self._pipeline.named_steps["clf"].classes_).index(1)
        return proba[:, idx].tolist()
