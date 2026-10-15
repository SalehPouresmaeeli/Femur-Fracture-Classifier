"""Run one experiment end to end: data -> model -> reports in a run folder."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import yaml

from femur_fx.config import DataConfig, ExperimentConfig
from femur_fx.data import FemurDataset, build_dataset
from femur_fx.experiments.common import write_json

log = logging.getLogger(__name__)

_dataset_cache: dict[str, FemurDataset] = {}


def get_dataset(cfg: DataConfig) -> FemurDataset:
    """Build the dataset once per distinct data config within this process."""
    key = json.dumps(asdict(cfg), sort_keys=True)
    if key not in _dataset_cache:
        _dataset_cache[key] = build_dataset(cfg)
    return _dataset_cache[key]


def make_run_dir(cfg: ExperimentConfig) -> Path:
    base = Path(cfg.output_dir) / cfg.name / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir, n = base, 1
    while run_dir.exists():  # same experiment started twice within one second
        n += 1
        run_dir = base.with_name(f"{base.name}_{n}")
    run_dir.mkdir(parents=True)
    return run_dir


def run_experiment(cfg: ExperimentConfig, dataset: FemurDataset | None = None) -> Path:
    """Run ``cfg`` and return its output folder.

    Pass ``dataset`` to skip loading (e.g. in tests or interactive sessions).
    """
    run_dir = make_run_dir(cfg)
    handler = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.addHandler(handler)
    try:
        log.info("=== Experiment %s -> %s", cfg.name, run_dir)
        if cfg.description:
            log.info("%s", cfg.description)
        with open(run_dir / "config.yaml", "w", encoding="utf-8") as fh:
            yaml.safe_dump(cfg.to_dict(), fh, sort_keys=False)

        if dataset is None:
            dataset = get_dataset(cfg.data)
        summary = dataset.summary()
        log.info(
            "Dataset: %d samples, %d patients, classes %s",
            summary["n_samples"], summary["n_patients"], summary["class_counts"],
        )
        write_json(run_dir / "dataset_summary.json", summary)

        if cfg.model.type == "cnn":
            from femur_fx.experiments.cnn import run_cnn_experiment

            run_cnn_experiment(cfg, dataset, run_dir)
        else:
            from femur_fx.experiments.classical import run_classical_experiment

            run_classical_experiment(cfg, dataset, run_dir)
        log.info("=== Finished %s", cfg.name)
    finally:
        root.removeHandler(handler)
        handler.close()
    return run_dir
