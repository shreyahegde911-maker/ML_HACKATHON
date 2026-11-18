"""Two reproducible candidate classifiers; reference columns are never features."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np

from pathsnap.features.engineering import FEATURES


def make_models(seed=42, jobs=2):
    return {
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.0, class_weight="balanced", max_iter=500, random_state=seed
            ),
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=80,
            max_depth=14,
            min_samples_leaf=20,
            max_features="sqrt",
            bootstrap=True,
            max_samples=0.75,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=jobs,
        ),
    }


def predict_candidates(model, frame):
    """Accept only the same four geometry/movement features used for fitting."""
    if frame.empty:
        return np.empty(0, dtype=float)
    return model.predict_proba(frame[FEATURES])[:, 1]
