"""Assemble the modelling dataset: clinical table + images + labels."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from femur_fx.config import DataConfig
from femur_fx.data.labels import canonical_fracture_type, encode_labels
from femur_fx.data.vti import ROTATION_COLUMNS, load_image_directory

log = logging.getLogger(__name__)

PATIENT_COLUMN = "Patient_id"
TARGET_COLUMN = "Fracture type"
KEY_COLUMNS = [PATIENT_COLUMN, *ROTATION_COLUMNS]


@dataclass
class FemurDataset:
    """Images, tabular features and labels, row-aligned."""

    images: np.ndarray  # (n_samples, rows, cols), float32
    tabular: pd.DataFrame  # (n_samples, n_tabular_features)
    y: np.ndarray  # (n_samples,) integer class labels
    groups: np.ndarray  # (n_samples,) patient ids, used for group-wise splits
    class_names: list[str]
    metadata: pd.DataFrame  # clinical table rows matching each sample

    def __len__(self) -> int:
        return len(self.y)

    @property
    def image_shape(self) -> tuple[int, int]:
        return tuple(self.images.shape[1:])

    def to_matrix(self) -> np.ndarray:
        """Flattened pixels followed by tabular features, one row per sample.

        This is the input layout expected by the sklearn transformers in
        :mod:`femur_fx.preprocessing`.
        """
        pixels = self.images.reshape(len(self), -1)
        if self.tabular.shape[1] == 0:
            return pixels
        return np.hstack([pixels, self.tabular.to_numpy(dtype=pixels.dtype)])

    def summary(self) -> dict:
        counts = np.bincount(self.y, minlength=len(self.class_names))
        patients_per_class = {
            name: sorted(map(int, np.unique(self.groups[self.y == i])))
            for i, name in enumerate(self.class_names)
        }
        sets = [set(p) for p in patients_per_class.values()]
        return {
            "n_samples": len(self),
            "n_patients": int(len(np.unique(self.groups))),
            "image_shape": list(self.image_shape),
            "tabular_features": list(self.tabular.columns),
            "class_counts": {name: int(c) for name, c in zip(self.class_names, counts)},
            "patients_per_class": {k: len(v) for k, v in patients_per_class.items()},
            "patients_in_several_classes": sorted(set.intersection(*sets)) if len(sets) > 1 else [],
        }


def add_loading_direction(df: pd.DataFrame) -> pd.DataFrame:
    """Unit loading vector from the y/z rotations (degrees) of each load case."""
    df = df.copy()
    ry = np.deg2rad(df["Rotation around y-axis"])
    rz = np.deg2rad(df["Rotation around z-axis"])
    df["Loading_x"] = -np.cos(ry) * np.cos(rz)
    df["Loading_y"] = np.sin(rz)
    df["Loading_z"] = -np.sin(ry) * np.cos(rz)
    return df


def load_clinical_table(cfg: DataConfig) -> pd.DataFrame:
    """Read the Excel table, drop excluded patients and clean the target."""
    df = pd.read_excel(cfg.excel_path)
    df = add_loading_direction(df)
    for col in ROTATION_COLUMNS:
        df[col] = df[col].astype(float)

    df = df[~df[PATIENT_COLUMN].isin(cfg.excluded_patients)].copy()
    df[TARGET_COLUMN] = df[TARGET_COLUMN].map(canonical_fracture_type)
    dropped = df[TARGET_COLUMN].isin(cfg.drop_fracture_types)
    if dropped.any():
        log.info("Dropping %d rows with fracture type in %s", dropped.sum(), cfg.drop_fracture_types)
    return df[~dropped].reset_index(drop=True)


def _use_fixed_images(merged: pd.DataFrame, meta: pd.DataFrame, images: np.ndarray) -> np.ndarray:
    """For each row, pick the patient's un-rotated CT image."""
    neutral = meta[(meta["Rotation around y-axis"] == 0) & (meta["Rotation around z-axis"] == 0)]
    duplicated = neutral[PATIENT_COLUMN].duplicated()
    if duplicated.any():
        raise ValueError(f"Several neutral images for patients {neutral.loc[duplicated, PATIENT_COLUMN].tolist()}")
    index = dict(zip(neutral[PATIENT_COLUMN], neutral.index))
    missing = set(merged[PATIENT_COLUMN]) - set(index)
    if missing:
        raise ValueError(f"No un-rotated image for patients {sorted(missing)}")
    return images[merged[PATIENT_COLUMN].map(index).to_numpy()]


def check_inputs_exist(cfg: DataConfig) -> None:
    hint = "Copy the data into the project's data/ folder (see data/README.md) or set the path with --set"
    if not Path(cfg.excel_path).is_file():
        raise FileNotFoundError(f"Excel table not found: {cfg.excel_path}\n{hint} data.excel_path=...")
    if not Path(cfg.image_dir).is_dir():
        raise FileNotFoundError(f"Image folder not found: {cfg.image_dir}\n{hint} data.image_dir=...")


def build_dataset(cfg: DataConfig) -> FemurDataset:
    """Load, clean and align everything described by ``cfg``."""
    check_inputs_exist(cfg)
    table = load_clinical_table(cfg)
    cache_dir = Path(cfg.cache_dir) if cfg.cache_dir else None
    meta, images = load_image_directory(cfg.image_dir, cache_dir)

    meta = meta.reset_index().rename(columns={"index": "_image_index"})
    merged = table.merge(meta, on=KEY_COLUMNS, how="inner", validate="one_to_one")
    n_unmatched_rows = len(table) - len(merged)
    if n_unmatched_rows:
        log.warning("%d table rows have no matching image and were dropped", n_unmatched_rows)
    n_unused_images = len(meta) - len(merged)
    if n_unused_images:
        log.info("%d of %d images are not used (excluded patients, dropped fracture types or no table row)",
                 n_unused_images, len(meta))
    merged = merged.sort_values(KEY_COLUMNS).reset_index(drop=True)

    if cfg.image_mode == "rotated":
        sample_images = images[merged["_image_index"].to_numpy()]
    elif cfg.image_mode == "fixed":
        sample_images = _use_fixed_images(merged, meta, images)
    else:
        raise ValueError(f"Unknown image_mode {cfg.image_mode!r}; use 'rotated' or 'fixed'")

    missing = [c for c in cfg.tabular_features if c not in merged.columns]
    if missing:
        raise KeyError(f"Tabular features not found in the table: {missing}")

    y, class_names = encode_labels(merged[TARGET_COLUMN], cfg.label_scheme)
    return FemurDataset(
        images=sample_images,
        tabular=merged[list(cfg.tabular_features)].astype(float),
        y=y,
        groups=merged[PATIENT_COLUMN].to_numpy(),
        class_names=class_names,
        metadata=merged.drop(columns=["_image_index"]),
    )
