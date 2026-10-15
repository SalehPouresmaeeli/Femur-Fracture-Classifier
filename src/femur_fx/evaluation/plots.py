"""Figures saved to disk (non-interactive backend, safe on headless machines)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay, auc, roc_curve  # noqa: E402


def save_figure(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_confusion_matrix(cm, class_names, path: Path, title: str = "Confusion matrix"):
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(np.asarray(cm), display_labels=class_names).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title)
    save_figure(fig, path)


def plot_roc_curve(y_true, proba_pos, path: Path, title: str = "ROC curve"):
    fpr, tpr, _ = roc_curve(y_true, proba_pos)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc(fpr, tpr):.2f}")
    ax.plot([0, 1], [0, 1], lw=1, linestyle="--", color="grey")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title=title)
    ax.legend(loc="lower right")
    save_figure(fig, path)


def plot_probability_histograms(y_true, proba, class_names, path: Path):
    """Per true class, the distribution of predicted probabilities for each class."""
    n = len(class_names)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
    for true_cls, ax in enumerate(axes[0]):
        mask = np.asarray(y_true) == true_cls
        for pred_cls, name in enumerate(class_names):
            ax.hist(proba[mask, pred_cls], bins=30, range=(0, 1), alpha=0.5, label=f"P({name})")
        ax.set(title=f"True class: {class_names[true_cls]}", xlabel="Predicted probability", ylabel="Count")
        ax.legend()
    save_figure(fig, path)


def plot_training_history(history: dict, path: Path):
    fig, ax = plt.subplots(figsize=(6, 4))
    for key in ("loss", "val_loss"):
        if key in history:
            ax.plot(history[key], label=key)
    ax.set(xlabel="Epoch", ylabel="Loss", yscale="log")
    ax.legend()
    save_figure(fig, path)


def plot_metric_boxplot(per_split: pd.DataFrame, columns: list[str], path: Path, title: str = ""):
    """Box plot of metrics across splits (e.g. the 20-seed robustness check)."""
    columns = [c for c in columns if c in per_split.columns]
    fig, ax = plt.subplots(figsize=(1.6 * len(columns) + 2, 4))
    ax.boxplot([per_split[c].dropna() for c in columns])
    ax.set_xticks(range(1, len(columns) + 1), columns)
    ax.set(ylabel="Score", title=title)
    ax.tick_params(axis="x", rotation=30)
    save_figure(fig, path)


def plot_image_grid(images: dict[str, np.ndarray], path: Path, cols: int = 3, flip: bool = True):
    """Show named images side by side (flipped vertically with ``np.flipud`` when ``flip`` is set)."""
    n = len(images)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, min(cols, n), figsize=(4 * min(cols, n), 4 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (title, img) in zip(axes.ravel(), images.items()):
        ax.imshow(np.flipud(img) if flip else img, cmap="gray")
        ax.set_title(title)
    save_figure(fig, path)
