"""Patient-grouped train/test splits.

Each patient contributes many CT images (one per load case), so every split keeps all
rows of a patient on the same side to avoid leaking patients across folds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from femur_fx.config import EvaluationConfig


@dataclass
class Split:
    name: str
    train: np.ndarray
    test: np.ndarray
    seed: int


def group_train_test_split(y, groups, test_size: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    return next(splitter.split(np.zeros(len(y)), y, groups))


def iter_splits(cfg: EvaluationConfig, y: np.ndarray, groups: np.ndarray, default_seed: int) -> Iterator[Split]:
    if cfg.strategy == "group_kfold":
        splitter = GroupKFold(n_splits=cfg.n_splits)
        for i, (train, test) in enumerate(splitter.split(np.zeros(len(y)), y, groups), start=1):
            yield Split(f"fold_{i}", train, test, default_seed)
    elif cfg.strategy == "group_shuffle_split":
        for i, seed in enumerate(cfg.seeds, start=1):
            train, test = group_train_test_split(y, groups, cfg.test_size, seed)
            yield Split(f"run_{i:02d}_seed_{seed}", train, test, seed)
    else:
        raise ValueError(f"Unknown evaluation strategy {cfg.strategy!r}; use 'group_kfold' or 'group_shuffle_split'")
