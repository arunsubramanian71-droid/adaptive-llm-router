from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingClassifier

from router.routers.base import Router, validate_binary_labels
from router.routers.features import featurize_batch


class GradientBoostingRouter(Router):
    """Gradient-boosted trees over the same handcrafted features as
    `HandcraftedFeatureRouter` — `sklearn`'s histogram-based implementation,
    which handles small/tabular data well without an extra dependency
    (xgboost/lightgbm)."""

    name = "gradient_boosting"

    def __init__(self, **gb_kwargs) -> None:
        # sklearn's default min_samples_leaf=20 is tuned for large datasets;
        # this project's pilots run in the hundreds of prompts, where that
        # default over-smooths leaves and can wash out real signal.
        gb_kwargs.setdefault("min_samples_leaf", 5)
        self._clf = HistGradientBoostingClassifier(**gb_kwargs)
        self._fitted = False

    def fit(self, prompts: list[str], labels: list[int]) -> None:
        validate_binary_labels(labels)
        X = featurize_batch(prompts)
        self._clf.fit(X, labels)
        self._fitted = True

    def predict_proba(self, prompts: list[str]) -> list[float]:
        if not self._fitted:
            raise RuntimeError(f"{self.name} router has not been fit yet")
        X = featurize_batch(prompts)
        proba = self._clf.predict_proba(X)
        idx = list(self._clf.classes_).index(1)
        return proba[:, idx].tolist()
