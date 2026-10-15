"""Local Binary Pattern (LBP) texture features."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from femur_fx.preprocessing.layout import join_matrix, split_matrix


def lbp_n_bins(P: int, method: str) -> int:
    """Number of distinct LBP codes, so every histogram has the same length."""
    if method == "uniform":
        return P + 2
    if method == "nri_uniform":
        return P * (P - 1) + 3
    if method in ("default", "ror"):
        return 2**P
    raise ValueError(f"Unsupported LBP method {method!r}; use uniform, nri_uniform, default or ror")


def lbp_image(image: np.ndarray, P: int = 8, R: float = 1, method: str = "uniform") -> np.ndarray:
    """LBP code map of an image (min-max normalised to uint8 first)."""
    from skimage import img_as_ubyte
    from skimage.feature import local_binary_pattern

    image = np.asarray(image, dtype=np.float64)
    span = image.max() - image.min()
    normalised = (image - image.min()) / span if span > 0 else np.zeros_like(image)
    return local_binary_pattern(img_as_ubyte(normalised), P, R, method)


def lbp_histogram(image: np.ndarray, P: int = 8, R: float = 1, method: str = "uniform") -> np.ndarray:
    """Normalised histogram of LBP codes (fixed length ``lbp_n_bins(P, method)``)."""
    n_bins = lbp_n_bins(P, method)
    hist, _ = np.histogram(lbp_image(image, P, R, method).ravel(), bins=n_bins, range=(0, n_bins), density=True)
    return hist


class LBPTransformer(BaseEstimator, TransformerMixin):
    """Replace the pixel block of a ``[pixels..., tabular...]`` matrix by its LBP histogram."""

    def __init__(self, image_shape=None, P=8, R=1, method="uniform"):
        self.image_shape = image_shape
        self.P = P
        self.R = R
        self.method = method

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        images, tabular = split_matrix(X, self.image_shape)
        hists = np.array([lbp_histogram(img, self.P, self.R, self.method) for img in images])
        return join_matrix(hists, tabular)
