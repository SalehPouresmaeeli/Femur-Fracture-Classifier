import numpy as np
import pandas as pd
import pytest

from femur_fx.analysis import classify_noise, noise_statistics, run_noise_analysis
from femur_fx.analysis.noise import load_variants
from femur_fx.cli import main


def test_noise_statistics_ignore_background():
    rng = np.random.default_rng(0)
    image = rng.gamma(4.0, 50.0, (16, 20)) + 100
    padded = np.pad(image, 5)  # zero background must not change the statistics
    s1, s2 = noise_statistics(image), noise_statistics(padded)
    for key in ("mean", "std", "skewness", "kurtosis"):
        assert s1[key] == s2[key]


def test_classify_noise_rules():
    base = {"skewness": 0, "kurtosis": 0, "std": 1.0, "sigma_est": 1.0, "mean_square_diff": 0.5,
            "variance": 10.0, "mean": 1.0, "spectrum_max": 1.0, "spectrum_median": 1.0}
    assert classify_noise(base) == []
    assert classify_noise({**base, "skewness": 2}) == ["salt_and_pepper_or_outlier"]
    assert classify_noise({**base, "mean_square_diff": 5}) == ["speckle"]
    assert classify_noise({**base, "variance": 0.5}) == ["poisson"]
    assert classify_noise({**base, "spectrum_max": 6}) == ["periodic"]


def test_run_noise_analysis(synthetic_data, tmp_path):
    out = run_noise_analysis(synthetic_data["image_dir"], tmp_path / "noise")
    report = pd.read_csv(out / "noise_report.csv")
    assert len(report) == 66
    assert (out / "noise_diagnostics.png").exists()
    assert (out / "denoising_comparison.png").exists()


def test_load_variants(tmp_path):
    path = tmp_path / "variants.yaml"
    path.write_text("soft: {method: median, disk_radius: 1}\n"
                    "strong: {method: nl_means, h_coef: 3.0, patch_size: 3, patch_distance: 4}\n")
    assert load_variants(path) == {
        "soft": {"method": "median", "disk_radius": 1},
        "strong": {"method": "nl_means", "h_coef": 3.0, "patch_size": 3, "patch_distance": 4},
    }


@pytest.mark.parametrize("content, message", [
    ("soft: {method: nl_means, h_coef: 1.0}\n", "patch_size"),
    ("soft: {disk_radius: 1}\n", "method"),
    ("soft: {method: blur}\n", "Unknown denoising method"),
    ("", "must map labels"),
])
def test_load_variants_rejects_incomplete_settings(tmp_path, content, message):
    path = tmp_path / "variants.yaml"
    path.write_text(content)
    with pytest.raises(ValueError, match=message):
        load_variants(path)


def test_noise_command_with_variants(synthetic_data, tmp_path):
    variants = tmp_path / "variants.yaml"
    variants.write_text("soft: {method: median, disk_radius: 1}\n")
    out = tmp_path / "noise"
    code = main(["noise", "--image-dir", synthetic_data["image_dir"], "--out", str(out),
                 "--variants", str(variants)])
    assert code == 0
    assert (out / "denoising_comparison.png").exists()
