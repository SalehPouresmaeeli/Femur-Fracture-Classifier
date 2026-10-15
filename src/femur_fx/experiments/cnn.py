"""Training and grouped evaluation of the Keras CNNs."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from femur_fx.config import ExperimentConfig
from femur_fx.data.dataset import FemurDataset
from femur_fx.evaluation import group_train_test_split, iter_splits, predict_labels
from femur_fx.evaluation import plots
from femur_fx.experiments.common import SplitResults
from femur_fx.models.cnn import IMAGE_INPUT, TABULAR_INPUT, CNNArchitecture, build_cnn
from femur_fx.preprocessing import ImageDenoiser, smote_images_and_tabular

log = logging.getLogger(__name__)


def fit_image_scaler(images: np.ndarray, mode: str):
    """Scaling fitted on training images only: ``none``, ``max`` or ``standard``."""
    if mode == "none":
        return lambda x: x
    if mode == "max":
        scale = float(np.abs(images).max()) or 1.0
        return lambda x: x / scale
    if mode == "standard":
        mean, std = float(images.mean()), float(images.std()) or 1.0
        return lambda x: (x - mean) / std
    raise ValueError(f"Unknown image_scaling {mode!r}; use 'none', 'max' or 'standard'")


def _inputs(images: np.ndarray, tabular: np.ndarray, with_tabular: bool):
    images = images[..., np.newaxis].astype(np.float32)
    if not with_tabular:
        return images
    return {IMAGE_INPUT: images, TABULAR_INPUT: tabular.astype(np.float32)}


def run_cnn_experiment(cfg: ExperimentConfig, dataset: FemurDataset, out_dir: Path) -> dict:
    import tensorflow as tf

    if not cfg.model.architecture:
        raise ValueError("model.architecture is required for CNN experiments")
    arch = CNNArchitecture.from_config(cfg.model.architecture)
    if cfg.preprocessing.lbp is not None:
        raise ValueError("LBP features are not supported for CNN experiments")
    train_cfg = cfg.training
    with_tabular = arch.uses_tabular
    if not with_tabular and dataset.tabular.shape[1]:
        log.warning("The architecture is image-only; ignoring tabular features %s", list(dataset.tabular))

    images = dataset.images
    if cfg.preprocessing.denoise is not None:
        log.info("Denoising %d images (%s)", len(images), cfg.preprocessing.denoise)
        images = ImageDenoiser(image_shape=dataset.image_shape, **cfg.preprocessing.denoise).denoise_images(images)
    tabular = dataset.tabular.to_numpy(dtype=np.float32)
    y, groups = dataset.y, dataset.groups
    n_classes = len(dataset.class_names)
    reseed = cfg.evaluation.strategy == "group_shuffle_split"

    results = SplitResults(dataset, out_dir)
    for split in iter_splits(cfg.evaluation, y, groups, cfg.seed):
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(split.seed)

        if train_cfg.validation_size > 0:
            fit_rel, val_rel = group_train_test_split(
                y[split.train], groups[split.train], train_cfg.validation_size, split.seed
            )
            fit_idx, val_idx = split.train[fit_rel], split.train[val_rel]
        else:
            fit_idx, val_idx = split.train, split.test

        scale = fit_image_scaler(images[fit_idx], cfg.preprocessing.image_scaling)
        fit_images, fit_tab, fit_y = scale(images[fit_idx]), tabular[fit_idx], y[fit_idx]
        if cfg.preprocessing.smote is not None:
            smote_kwargs = dict(cfg.preprocessing.smote)
            if reseed:
                smote_kwargs["random_state"] = split.seed
            fit_images, fit_tab, fit_y = smote_images_and_tabular(fit_images, fit_tab, fit_y, **smote_kwargs)
            log.info("[%s] class counts after SMOTE: %s", split.name, np.bincount(fit_y).tolist())

        model = build_cnn(arch, dataset.image_shape, n_classes, tabular.shape[1], train_cfg.learning_rate,
                          name=cfg.name)
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=train_cfg.patience, restore_best_weights=True)
        ]
        if train_cfg.tensorboard:
            callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=str(out_dir / "tensorboard" / split.name)))

        history = model.fit(
            _inputs(fit_images, fit_tab, with_tabular),
            tf.keras.utils.to_categorical(fit_y, n_classes),
            validation_data=(
                _inputs(scale(images[val_idx]), tabular[val_idx], with_tabular),
                tf.keras.utils.to_categorical(y[val_idx], n_classes),
            ),
            epochs=train_cfg.epochs,
            batch_size=train_cfg.batch_size,
            callbacks=callbacks,
            verbose=2,
        )
        plots.plot_training_history(history.history, out_dir / "splits" / f"{split.name}_loss.png")

        proba = model.predict(_inputs(scale(images[split.test]), tabular[split.test], with_tabular), verbose=0)
        results.add(split.name, split.test, predict_labels(proba, cfg.evaluation.decision_threshold), proba)
        if train_cfg.save_models:
            model_dir = out_dir / "models"
            model_dir.mkdir(parents=True, exist_ok=True)
            model.save(model_dir / f"{split.name}.keras")

    return results.finalise()
