"""Experiment configuration: YAML files -> typed dataclasses.

A config file may name a ``base`` file (path relative to itself) whose
contents are deep-merged underneath it, so shared settings such as data
paths live in one place::

    base: ../data.yaml
    name: my_experiment
    model: {type: random_forest}

Relative file paths (``data.excel_path``, ``data.image_dir``,
``data.cache_dir``, ``output_dir``) are resolved against the folder of the
config file that sets them, so runs work from any working directory.
Paths given with ``--set`` overrides are relative to the working directory.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    excel_path: str = ""
    image_dir: str = ""
    # Patients without usable CT images.
    excluded_patients: list[int] = field(default_factory=list)
    # "binary": Neck (SC, TC) vs Trochanteric (IT, PT); "detailed": IT/PT/SC/TC.
    label_scheme: str = "binary"
    drop_fracture_types: list[str] = field(default_factory=lambda: ["ST"])
    # "rotated": the image matching each load case; "fixed": every row
    # uses the patient's un-rotated (0, 0, 0) image.
    image_mode: str = "rotated"
    tabular_features: list[str] = field(default_factory=list)
    # Loaded VTI images are cached here as .npz; null disables caching.
    cache_dir: str | None = ".cache"


@dataclass
class ModelConfig:
    # "knn", "random_forest" or "cnn".
    type: str = "random_forest"
    # Classifier keyword arguments (sklearn models only).
    params: dict[str, Any] = field(default_factory=dict)
    # CNN layers (cnn only): conv_layers, dense_units, padding, dropout,
    # tabular_units, head_units... see femur_fx.models.cnn.CNNArchitecture.
    architecture: dict[str, Any] | None = None


@dataclass
class PreprocessingConfig:
    # e.g. {method: median, disk_radius: 1}; every setting of the method is required
    # (see femur_fx.preprocessing.denoise.REQUIRED_SETTINGS).
    denoise: dict[str, Any] | None = None
    # e.g. {P: 8, R: 1, method: uniform}
    lbp: dict[str, Any] | None = None
    # e.g. {sampling_strategy: auto, random_state: 0}; null disables SMOTE.
    smote: dict[str, Any] | None = None
    # CNN only: "none", "max" or "standard" (statistics from the training split).
    image_scaling: str = "none"


@dataclass
class SearchConfig:
    # "grid" or "random".
    kind: str = "grid"
    # Dict, or list of dicts, of pipeline parameters (step__param). For random
    # search a value may be {distribution: uniform|randint|loguniform, ...}.
    param_grid: Any = field(default_factory=dict)
    n_iter: int = 10
    scoring: str = "f1_weighted"
    cv_splits: int = 5
    # The search runs on a group-wise train split; the rest is a held-out test.
    test_size: float = 0.2
    random_state: int = 0
    n_jobs: int = -1
    # Use the best parameters for the evaluation stage that follows.
    apply_best: bool = True


@dataclass
class EvaluationConfig:
    # "group_kfold" or "group_shuffle_split" (one split per seed).
    strategy: str = "group_kfold"
    n_splits: int = 5
    test_size: float = 0.2
    seeds: list[int] = field(default_factory=lambda: [0])
    # Binary only: predict class 1 when P(class 1) >= threshold.
    decision_threshold: float | None = None
    # Fit the final pipeline on all data and save it (sklearn models).
    refit_full: bool = True


@dataclass
class TrainingConfig:
    """CNN training settings."""

    epochs: int = 50
    batch_size: int = 32
    patience: int = 5
    # Fraction of training patients held out for early stopping. 0 reuses the
    # test split for validation.
    validation_size: float = 0.0
    learning_rate: float = 1e-3
    tensorboard: bool = False
    save_models: bool = True


@dataclass
class ExperimentConfig:
    name: str = "experiment"
    description: str = ""
    seed: int = 42
    output_dir: str = "outputs"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    search: SearchConfig | None = None
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_SECTIONS = {
    "data": DataConfig,
    "model": ModelConfig,
    "preprocessing": PreprocessingConfig,
    "search": SearchConfig,
    "evaluation": EvaluationConfig,
    "training": TrainingConfig,
}


def _build(cls, values: dict[str, Any] | None, where: str):
    values = values or {}
    known = {f.name for f in fields(cls)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"Unknown key(s) in '{where}': {sorted(unknown)}")
    return cls(**values)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


_DATA_PATH_KEYS = ("excel_path", "image_dir", "cache_dir")


def _resolve(value: str, base_dir: Path) -> str:
    path = Path(value)
    return str(path if path.is_absolute() else (base_dir / path).resolve())


def resolve_paths(raw: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    """Make the relative file paths in one config file absolute, relative to ``base_dir``."""
    if isinstance(raw.get("output_dir"), str) and raw["output_dir"]:
        raw["output_dir"] = _resolve(raw["output_dir"], base_dir)
    data = raw.get("data")
    if isinstance(data, dict):
        for key in _DATA_PATH_KEYS:
            if isinstance(data.get(key), str) and data[key]:
                data[key] = _resolve(data[key], base_dir)
    return raw


def load_raw_config(path: str | Path) -> dict[str, Any]:
    """Read a YAML config, resolving its ``base`` chain and relative paths."""
    path = Path(path)
    if not path.is_file():
        template = path.with_name(f"{path.stem}.example{path.suffix}")
        hint = f"; create it from the template: copy {template} {path}" if template.is_file() else ""
        raise FileNotFoundError(f"Config file not found: {path}{hint}")
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    raw = resolve_paths(raw, path.resolve().parent)
    base = raw.pop("base", None)
    if base:
        raw = deep_merge(load_raw_config(path.parent / base), raw)
    return raw


def apply_overrides(raw: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply ``dotted.key=value`` overrides; values are parsed as YAML."""
    raw = copy.deepcopy(raw)
    for item in overrides:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"Override must look like key=value, got {item!r}")
        node = raw
        *parents, leaf = key.strip().split(".")
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = yaml.safe_load(value)
    return raw


def config_from_dict(raw: dict[str, Any]) -> ExperimentConfig:
    raw = dict(raw)
    sections = {}
    for name, cls in _SECTIONS.items():
        if name in raw:
            value = raw.pop(name)
            sections[name] = None if value is None and name == "search" else _build(cls, value, name)
    cfg = _build(ExperimentConfig, raw, "top level")
    for name, value in sections.items():
        setattr(cfg, name, value)
    return cfg


def load_config(path: str | Path, overrides: list[str] | None = None) -> ExperimentConfig:
    raw = load_raw_config(path)
    if overrides:
        raw = apply_overrides(raw, overrides)
    return config_from_dict(raw)
