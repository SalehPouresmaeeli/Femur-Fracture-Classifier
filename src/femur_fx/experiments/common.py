"""Shared bookkeeping for experiment runners: per-split results and reports."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from femur_fx.data.dataset import FemurDataset
from femur_fx.evaluation import compute_metrics, summarise_splits, text_report
from femur_fx.evaluation import plots

log = logging.getLogger(__name__)


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(data), indent=2), encoding="utf-8")


class SplitResults:
    """Collects test-set predictions split by split and writes the reports."""

    def __init__(self, dataset: FemurDataset, out_dir: Path):
        self.dataset = dataset
        self.out_dir = out_dir
        self.metrics: list[dict] = []
        self.predictions: list[pd.DataFrame] = []

    def add(self, split_name: str, test_idx: np.ndarray, y_pred: np.ndarray, proba: np.ndarray) -> dict:
        ds = self.dataset
        y_true = ds.y[test_idx]
        metrics = compute_metrics(y_true, y_pred, proba, ds.class_names)
        metrics = {"split": split_name, "n_test": len(test_idx), **metrics}
        self.metrics.append(metrics)

        report = text_report(y_true, y_pred, ds.class_names)
        log.info("[%s] F1 weighted %.4f\n%s", split_name, metrics["f1_weighted"], report)
        split_dir = self.out_dir / "splits"
        split_dir.mkdir(parents=True, exist_ok=True)
        (split_dir / f"{split_name}_report.txt").write_text(report, encoding="utf-8")
        plots.plot_confusion_matrix(
            metrics["confusion_matrix"], ds.class_names, split_dir / f"{split_name}_confusion.png", split_name
        )

        pred = ds.metadata.iloc[test_idx][["Patient_id", "Rotation around x-axis", "Rotation around y-axis",
                                           "Rotation around z-axis"]].copy()
        pred.insert(0, "split", split_name)
        pred["y_true"] = y_true
        pred["y_pred"] = y_pred
        for i, name in enumerate(ds.class_names):
            pred[f"proba_{name}"] = proba[:, i]
        self.predictions.append(pred)
        return metrics

    def finalise(self) -> dict:
        ds = self.dataset
        per_split = summarise_splits(self.metrics)
        per_split.to_csv(self.out_dir / "metrics_per_split.csv")
        predictions = pd.concat(self.predictions, ignore_index=True)
        predictions.to_csv(self.out_dir / "predictions.csv", index=False)

        total_cm = np.sum([m["confusion_matrix"] for m in self.metrics], axis=0)
        plots.plot_confusion_matrix(total_cm, ds.class_names, self.out_dir / "confusion_matrix.png",
                                    "Confusion matrix (all test splits)")
        proba = predictions[[f"proba_{n}" for n in ds.class_names]].to_numpy()
        plots.plot_probability_histograms(predictions["y_true"], proba, ds.class_names,
                                          self.out_dir / "probability_histograms.png")
        if len(ds.class_names) == 2:
            plots.plot_roc_curve(predictions["y_true"], proba[:, 1], self.out_dir / "roc_curve.png",
                                 "ROC (pooled test predictions)")
        if len(self.metrics) > 1:
            f1_cols = [c for c in per_split.columns if c.startswith("f1_")]
            plots.plot_metric_boxplot(per_split.drop(index=["mean", "std"]), f1_cols,
                                      self.out_dir / "f1_boxplot.png", "F1 across splits")

        summary = {
            "mean": per_split.loc["mean"].to_dict(),
            "std": per_split.loc["std"].to_dict(),
            "confusion_matrix_total": total_cm,
            "splits": self.metrics,
        }
        write_json(self.out_dir / "metrics.json", summary)
        log.info("Mean over %d split(s):\n%s", len(self.metrics), per_split.loc[["mean", "std"]].T.to_string())
        return summary
