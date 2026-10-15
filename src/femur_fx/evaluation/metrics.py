"""Scoring and metric helpers."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    roc_auc_score,
)

_CLASS_F1_RE = re.compile(r"f1_class_(\d+)$")


def _f1_of_class(y_true, y_pred, label: int) -> float:
    return f1_score(y_true, y_pred, labels=[label], average="macro", zero_division=0)


def get_scorer(name: str):
    """Scorer by name.

    ``f1_class_<k>`` scores the F1 of class ``k`` alone (e.g. ``f1_class_1``
    for the trochanteric class). ``f1_weighted`` and
    ``f1_macro`` use ``zero_division=0``. Anything else is passed through to
    sklearn as a built-in scorer name.
    """
    match = _CLASS_F1_RE.match(name)
    if match:
        return make_scorer(_f1_of_class, label=int(match.group(1)))
    if name in ("f1_weighted", "f1_macro"):
        return make_scorer(f1_score, average=name.split("_")[1], zero_division=0)
    return name


def predict_labels(proba: np.ndarray, threshold: float | None = None) -> np.ndarray:
    """Arg-max labels, or ``P(class 1) >= threshold`` for binary problems."""
    if threshold is None:
        return proba.argmax(axis=1)
    if proba.shape[1] != 2:
        raise ValueError("decision_threshold is only supported for binary classification")
    return (proba[:, 1] >= threshold).astype(int)


def compute_metrics(y_true, y_pred, proba, class_names: list[str]) -> dict:
    labels = list(range(len(class_names)))
    report = classification_report(
        y_true, y_pred, labels=labels, target_names=class_names, zero_division=0, output_dict=True
    )
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_weighted": report["weighted avg"]["f1-score"],
        "f1_macro": report["macro avg"]["f1-score"],
    }
    for name in class_names:
        for key in ("precision", "recall", "f1-score"):
            metrics[f"{key.replace('-score', '')}_{name}"] = report[name][key]
    if proba is not None and len(class_names) == 2 and len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = roc_auc_score(y_true, proba[:, 1])
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    return metrics


def text_report(y_true, y_pred, class_names: list[str]) -> str:
    labels = list(range(len(class_names)))
    return classification_report(y_true, y_pred, labels=labels, target_names=class_names, zero_division=0)


def summarise_splits(split_metrics: list[dict]) -> pd.DataFrame:
    """One row per split plus ``mean`` and ``std`` rows (scalar metrics only)."""
    rows = [{k: v for k, v in m.items() if not isinstance(v, list)} for m in split_metrics]
    df = pd.DataFrame(rows).set_index("split")
    numeric = df.select_dtypes("number")
    summary = pd.DataFrame({"mean": numeric.mean(), "std": numeric.std(ddof=0)}).T
    return pd.concat([df, summary])
