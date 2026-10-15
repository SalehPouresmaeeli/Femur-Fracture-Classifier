"""Heuristic noise characterisation of the CT images.

Statistics are computed over non-zero (foreground) pixels only. The rules are
simple heuristics, not statistical tests; they help decide whether the
denoising filters in :mod:`femur_fx.preprocessing.denoise` are needed.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import fft, stats

from femur_fx.data.vti import read_vti
from femur_fx.evaluation import plots
from femur_fx.preprocessing.denoise import check_settings, denoise_image

log = logging.getLogger(__name__)


def _log_spectrum(image: np.ndarray) -> np.ndarray:
    masked = np.where(image > 0, image, 0.0)
    return np.log1p(np.abs(fft.fftshift(fft.fft2(masked))))


def noise_statistics(image: np.ndarray) -> dict:
    from skimage import img_as_float
    from skimage.restoration import estimate_sigma

    image = img_as_float(image)
    fg = image[image > 0]
    if fg.size == 0:
        raise ValueError("Image has no non-zero pixels")
    spectrum = _log_spectrum(image)
    return {
        "mean": float(fg.mean()),
        "std": float(fg.std()),
        "variance": float(fg.var()),
        "skewness": float(stats.skew(fg)),
        "kurtosis": float(stats.kurtosis(fg)),
        "sigma_est": float(estimate_sigma(fg, average_sigmas=True)),
        "mean_square_diff": float(np.mean((fg - fg.mean()) ** 2)),
        "spectrum_max": float(spectrum.max()),
        "spectrum_median": float(np.median(spectrum)),
    }


def classify_noise(s: dict) -> list[str]:
    """Apply simple rules of thumb to :func:`noise_statistics` output."""
    found = []
    if s["skewness"] > 1 or s["kurtosis"] > 3:
        found.append("salt_and_pepper_or_outlier")
    elif s["std"] > 1.5 * s["sigma_est"]:
        found.append("gaussian")
    if s["mean_square_diff"] > s["std"]:
        found.append("speckle")
    if s["variance"] / s["mean"] < 1:
        found.append("poisson")
    if s["spectrum_max"] > 5 * s["spectrum_median"]:
        found.append("periodic")
    return found


def noise_report(image_dir: str | Path) -> pd.DataFrame:
    """Statistics and detected noise types for every .vti file in a folder."""
    rows = []
    for path in sorted(Path(image_dir).glob("*.vti")):
        s = noise_statistics(read_vti(path))
        rows.append({"filename": path.name, **s, "noise_types": ";".join(classify_noise(s))})
    return pd.DataFrame(rows)


def plot_noise_diagnostics(image: np.ndarray, path: Path) -> None:
    """Foreground intensity histogram and log magnitude spectrum."""
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.hist(image[image > 0].ravel(), bins=50, color="grey")
    ax1.set(title="Foreground intensity histogram", xlabel="Intensity", ylabel="Count")
    ax2.imshow(_log_spectrum(image), cmap="gray")
    ax2.set(title="Log magnitude spectrum (periodic noise check)")
    ax2.axis("off")
    plots.save_figure(fig, path)


# Generic filter settings for plot_denoising_comparison: each filter at a mild and a
# strong setting. Pass your own with `python -m femur_fx noise --variants <file.yaml>`.
DEFAULT_VARIANTS = {
    "median r=1": {"method": "median", "disk_radius": 1},
    "median r=4": {"method": "median", "disk_radius": 4},
    "wiener 2x2": {"method": "wiener", "window": 2},
    "wiener 7x7": {"method": "wiener", "window": 7},
    "NLM h=1.0 sigma": {"method": "nl_means", "h_coef": 1.0, "patch_size": 9, "patch_distance": 13},
    "NLM h=2.0 sigma": {"method": "nl_means", "h_coef": 2.0, "patch_size": 9, "patch_distance": 13},
}


def load_variants(path: str | Path) -> dict[str, dict]:
    """Read filter settings to compare from YAML: ``label: {method: ..., <settings>}`` per entry."""
    with open(path, encoding="utf-8") as fh:
        variants = yaml.safe_load(fh) or {}
    if not isinstance(variants, dict) or not variants:
        raise ValueError(f"{path} must map labels to filter settings, e.g. 'median r=1: {{method: median, disk_radius: 1}}'")
    for label, spec in variants.items():
        if not isinstance(spec, dict) or "method" not in spec:
            raise ValueError(f"Variant {label!r} in {path} needs a 'method' and its settings")
        settings = {k: v for k, v in spec.items() if k != "method"}
        check_settings(spec["method"], settings)
    return variants


def plot_denoising_comparison(image: np.ndarray, path: Path, variants: dict[str, dict] | None = None) -> None:
    variants = variants or DEFAULT_VARIANTS
    panels = {"original": image}
    for title, spec in variants.items():
        settings = {k: v for k, v in spec.items() if k != "method"}
        panels[title] = denoise_image(image, spec["method"], **settings)
    plots.plot_image_grid(panels, path, cols=4)


def run_noise_analysis(
    image_dir: str | Path, out_dir: str | Path, example: str | None = None, variants: dict[str, dict] | None = None
) -> Path:
    """Write ``noise_report.csv`` and diagnostic figures for one example image."""
    image_dir, out_dir = Path(image_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = noise_report(image_dir)
    if report.empty:
        raise FileNotFoundError(f"No .vti files found in {image_dir}")
    report.to_csv(out_dir / "noise_report.csv", index=False)
    counts = report["noise_types"].str.split(";").explode().value_counts()
    log.info("Noise types over %d images:\n%s", len(report), counts.to_string())

    example_path = image_dir / example if example else image_dir / report["filename"].iloc[0]
    image = read_vti(example_path)
    plot_noise_diagnostics(image, out_dir / "noise_diagnostics.png")
    plot_denoising_comparison(image, out_dir / "denoising_comparison.png", variants)
    log.info("Noise analysis written to %s (example image: %s)", out_dir, example_path.name)
    return out_dir
