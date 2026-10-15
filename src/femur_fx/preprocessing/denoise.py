"""Speckle-noise reduction for the CT images.

Three filters are available: median, Wiener and non-local means. Run
``python -m femur_fx noise`` to check the images for noise and compare them.

There are no default filter settings: every setting of the chosen method
must be given, e.g. in the ``preprocessing.denoise`` section of a config::

    denoise: {method: nl_means, h_coef: ..., patch_size: ..., patch_distance: ...}
"""

from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin

from femur_fx.preprocessing.layout import join_matrix, split_matrix


def median_denoise(image: np.ndarray, disk_radius: int) -> np.ndarray:
    from skimage.filters import median
    from skimage.morphology import disk

    return median(image, disk(disk_radius))


def wiener_denoise(image: np.ndarray, window: int) -> np.ndarray:
    from scipy.signal import wiener

    # Flat background regions have zero local variance -> 0/0 inside wiener.
    with np.errstate(divide="ignore", invalid="ignore"):
        filtered = wiener(image, mysize=(window, window))
    return np.nan_to_num(filtered, nan=0.0, posinf=0.0, neginf=0.0)


def nl_means_denoise(image: np.ndarray, h_coef: float, patch_size: int, patch_distance: int) -> np.ndarray:
    """Non-local means with filter strength ``h = h_coef * estimated noise sigma``."""
    from skimage.restoration import denoise_nl_means, estimate_sigma

    sigma = estimate_sigma(image, average_sigmas=True)
    return denoise_nl_means(
        image, h=h_coef * sigma, fast_mode=True, patch_size=patch_size, patch_distance=patch_distance
    )


DENOISERS = {
    "median": median_denoise,
    "wiener": wiener_denoise,
    "nl_means": nl_means_denoise,
}

# Settings each method needs.
REQUIRED_SETTINGS = {
    "median": ("disk_radius",),
    "wiener": ("window",),
    "nl_means": ("h_coef", "patch_size", "patch_distance"),
}


def check_settings(method: str, settings: dict) -> dict:
    """Return only the settings ``method`` uses; raise if one is missing or the method is unknown."""
    if method not in DENOISERS:
        raise ValueError(f"Unknown denoising method {method!r}; choose from {sorted(DENOISERS)}")
    missing = [name for name in REQUIRED_SETTINGS[method] if settings.get(name) is None]
    if missing:
        raise ValueError(f"Denoising method {method!r} needs setting(s) {missing} (e.g. in preprocessing.denoise)")
    return {name: settings[name] for name in REQUIRED_SETTINGS[method]}


def denoise_image(image: np.ndarray, method: str, **settings) -> np.ndarray:
    kwargs = check_settings(method, settings)
    return DENOISERS[method](np.asarray(image, dtype=np.float64), **kwargs).astype(np.float32)


class ImageDenoiser(BaseEstimator, TransformerMixin):
    """Denoise each image independently in a ``[pixels..., tabular...]`` matrix.

    Tabular columns pass through unchanged. All method settings are exposed
    so they can be tuned with ``GridSearchCV`` (``denoise__h_coef`` etc.);
    only those of ``method`` are used, and all of them must be set.
    """

    def __init__(
        self,
        image_shape=None,
        method=None,
        h_coef=None,
        patch_size=None,
        patch_distance=None,
        disk_radius=None,
        window=None,
        n_jobs=None,
    ):
        self.image_shape = image_shape
        self.method = method
        self.h_coef = h_coef
        self.patch_size = patch_size
        self.patch_distance = patch_distance
        self.disk_radius = disk_radius
        self.window = window
        self.n_jobs = n_jobs

    def _method_kwargs(self) -> dict:
        if self.method is None:
            raise ValueError(f"Denoising needs a method; choose from {sorted(DENOISERS)}")
        settings = {name: getattr(self, name) for names in REQUIRED_SETTINGS.values() for name in names}
        return check_settings(self.method, settings)

    def fit(self, X, y=None):
        self._method_kwargs()  # fail early on missing settings
        return self

    def denoise_images(self, images: np.ndarray) -> np.ndarray:
        kwargs = self._method_kwargs()
        if self.n_jobs in (None, 1):
            out = [denoise_image(img, self.method, **kwargs) for img in images]
        else:
            out = Parallel(n_jobs=self.n_jobs)(delayed(denoise_image)(img, self.method, **kwargs) for img in images)
        return np.stack(out)

    def transform(self, X):
        images, tabular = split_matrix(X, self.image_shape)
        return join_matrix(self.denoise_images(images), tabular)
