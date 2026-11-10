"""
Why did the Random Forest say TAKE or DROP?

For a tree, a prediction is the root's probability plus the change at every split on the
path to the leaf; each change is credited to the feature that split. Averaging over the
forest gives  bias + sum(contributions) == predict_proba  exactly.
"""

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def _node_probabilities(tree) -> np.ndarray:
    values = tree.tree_.value[:, 0, :]
    return values[:, 1] / values.sum(axis=1)


def explain_forest_prediction(model: RandomForestClassifier, row: pd.DataFrame) -> Tuple[float, Dict[str, float]]:
    """Returns (bias, {feature: contribution}) for the probability of class 1 on a single-row frame."""
    features = list(row.columns)
    x = row.to_numpy(dtype=np.float32)
    contributions = np.zeros(len(features))
    bias = 0.0
    for estimator in model.estimators_:
        probs = _node_probabilities(estimator)
        path = estimator.decision_path(x).indices
        bias += probs[path[0]]
        for parent, child in zip(path[:-1], path[1:]):
            contributions[estimator.tree_.feature[parent]] += probs[child] - probs[parent]
    n = len(model.estimators_)
    return bias / n, {f: float(c / n) for f, c in zip(features, contributions)}
