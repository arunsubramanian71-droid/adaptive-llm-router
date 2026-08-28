from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from router.routers.base import Router, validate_binary_labels
from router.routers.features import featurize_batch


class HandcraftedFeatureRouter(Router):
    name = "handcrafted_logreg"

    def __init__(self, **logreg_kwargs) -> None:
        logreg_kwargs.setdefault("max_iter", 1000)
        self._scaler = StandardScaler()
        self._clf = LogisticRegression(**logreg_kwargs)
        self._fitted = False

    def fit(self, prompts: list[str], labels: list[int]) -> None:
        validate_binary_labels(labels)
        X = self._scaler.fit_transform(featurize_batch(prompts))
        self._clf.fit(X, labels)
        self._fitted = True

    def predict_proba(self, prompts: list[str]) -> list[float]:
        if not self._fitted:
            raise RuntimeError(f"{self.name} router has not been fit yet")
        X = self._scaler.transform(featurize_batch(prompts))
        proba = self._clf.predict_proba(X)
        idx = list(self._clf.classes_).index(1)
        return proba[:, idx].tolist()
