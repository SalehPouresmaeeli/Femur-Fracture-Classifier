"""Helpers for the flat ``[pixels..., tabular...]`` feature-matrix layout."""

from __future__ import annotations

import numpy as np


def split_matrix(X: np.ndarray, image_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Split rows of ``X`` into an image stack and the trailing tabular columns."""
    if image_shape is None:
        raise ValueError("image_shape must be set to the (rows, cols) of the images")
    image_shape = tuple(image_shape)
    X = np.asarray(X)
    n_pixels = int(np.prod(image_shape))
    if X.ndim != 2 or X.shape[1] < n_pixels:
        raise ValueError(f"Expected at least {n_pixels} columns for images of shape {image_shape}, got {X.shape}")
    images = X[:, :n_pixels].reshape(len(X), *image_shape)
    return images, X[:, n_pixels:]


def join_matrix(image_features: np.ndarray, tabular: np.ndarray) -> np.ndarray:
    """Concatenate per-sample image features (any shape) with tabular columns."""
    flat = image_features.reshape(len(image_features), -1)
    if tabular.shape[1] == 0:
        return flat
    return np.hstack([flat, tabular.astype(flat.dtype, copy=False)])
