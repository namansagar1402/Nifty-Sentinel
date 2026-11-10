"""
How good is the take/drop model, measured honestly.

- Trains the production Random Forest configuration, a gradient-boosting model and a plain
  logistic-regression baseline on training trades.
- Scores all three on trades they have never seen.
- Adds cross-validated ROC-AUC on the training trades, and the accuracy of the "always say
  no" baseline, so a high accuracy can't hide a lazy model.

Two data sources: simulated trades (independent random draws), and real Nifty history,
where the test set is the most recent months and cross-validation respects time order.
"""

from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ml_engine.classifier import TRAINING_SAMPLES, TRAINING_SEED, OptionOutcomePredictor
from ml_engine.real_data import time_split
from ml_engine.synthetic_data import generate_synthetic_options_data

EVALUATION_SEED = 99  # differs from TRAINING_SEED (42) and the backtest's seed (7)
ROC_POINTS = 60       # downsample the ROC curve so the chart stays light
MODEL_LABELS = {"random_forest": "Random Forest", "gradient_boosting": "Gradient boosting",
                "logistic_regression": "Logistic regression (baseline)"}
FEATURES = OptionOutcomePredictor().feature_names


def candidate_models() -> Dict[str, Any]:
    """The three models compared: production Random Forest, gradient boosting, logistic baseline."""
    return {
        "random_forest": OptionOutcomePredictor().model,
        "gradient_boosting": HistGradientBoostingClassifier(max_depth=4, learning_rate=0.1, max_iter=150,
                                                            random_state=TRAINING_SEED),
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    }


def _thin_curve(fpr: np.ndarray, tpr: np.ndarray, points: int = ROC_POINTS) -> Dict[str, List[float]]:
    idx = np.unique(np.linspace(0, len(fpr) - 1, min(points, len(fpr))).astype(int))
    return {"fpr": [round(float(x), 4) for x in fpr[idx]], "tpr": [round(float(y), 4) for y in tpr[idx]]}


def _score(model, x_train, y_train, x_test, y_test, cv) -> Dict[str, Any]:
    cv_auc = cross_val_score(clone(model), x_train, y_train, cv=cv, scoring="roc_auc")
    fitted = clone(model).fit(x_train, y_train)
    pred = fitted.predict(x_test)
    prob = fitted.predict_proba(x_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, prob)
    return {
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "precision": round(float(precision_score(y_test, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, prob)), 4),
        "cv_auc_mean": round(float(cv_auc.mean()), 4),
        "cv_auc_std": round(float(cv_auc.std()), 4),
        # rows = actual (drop, take), columns = predicted (drop, take)
        "confusion_matrix": confusion_matrix(y_test, pred, labels=[0, 1]).tolist(),
        "roc_curve": _thin_curve(fpr, tpr),
        "test_probabilities": prob,
        "fitted_model": fitted,
    }


def evaluate_on(train: pd.DataFrame, test: pd.DataFrame, cv) -> Dict[str, Any]:
    """Scores every candidate model trained on `train` and tested on `test`."""
    x_train, y_train = train[FEATURES], train["is_profitable"]
    x_test, y_test = test[FEATURES], test["is_profitable"]
    profitable_share = float(y_test.mean())
    return {
        "train_size": len(train),
        "test_size": len(test),
        "profitable_share": round(profitable_share, 4),
        "majority_baseline_accuracy": round(max(profitable_share, 1 - profitable_share), 4),
        "models": {name: _score(m, x_train, y_train, x_test, y_test, cv) for name, m in candidate_models().items()},
    }


def evaluate_models(test_samples: int = 2000, cv_folds: int = 5) -> Dict[str, Any]:
    """Simulated trades: train on the seed-42 draw, test on a separate seed-99 draw."""
    train = generate_synthetic_options_data(TRAINING_SAMPLES, seed=TRAINING_SEED)
    test = generate_synthetic_options_data(test_samples, seed=EVALUATION_SEED)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=TRAINING_SEED)
    report = evaluate_on(train, test, cv)
    report.update(cv_folds=cv_folds, source="simulated", test_frame=test)
    return report


def evaluate_real(trades: pd.DataFrame, test_months: int = 6, cv_folds: int = 5) -> Dict[str, Any]:
    """Real Nifty trades: test on the last `test_months`; time-ordered CV on the older, purged trades."""
    train, test, cutoff = time_split(trades, test_months)
    report = evaluate_on(train, test, TimeSeriesSplit(n_splits=cv_folds))
    report.update(cv_folds=cv_folds, source="real", cutoff=cutoff, test_frame=test,
                  train_start=train["entry_date"].min(), test_end=test["entry_date"].max())
    return report
