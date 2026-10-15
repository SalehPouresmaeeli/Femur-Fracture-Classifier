"""Class-imbalance handling."""

from __future__ import annotations

import numpy as np


def make_smote(sampling_strategy="auto", random_state=0, **kwargs):
    """SMOTE oversampler. ``"auto"`` grows every minority class to the majority size."""
    from imblearn.over_sampling import SMOTE

    return SMOTE(sampling_strategy=sampling_strategy, random_state=random_state, **kwargs)


def smote_images_and_tabular(images: np.ndarray, tabular: np.ndarray, y: np.ndarray, **smote_kwargs):
    """Oversample images and tabular features together.

    Resampling the two inputs in a single call keeps each synthetic image
    paired with the synthetic tabular row generated from the same neighbours.
    """
    n, *image_shape = images.shape
    joint = np.hstack([images.reshape(n, -1), tabular])
    joint_res, y_res = make_smote(**smote_kwargs).fit_resample(joint, y)
    n_pixels = int(np.prod(image_shape))
    images_res = joint_res[:, :n_pixels].reshape(-1, *image_shape).astype(images.dtype, copy=False)
    tabular_res = joint_res[:, n_pixels:].astype(tabular.dtype, copy=False)
    return images_res, tabular_res, y_res
