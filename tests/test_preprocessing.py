import numpy as np
import pytest

from femur_fx.preprocessing import (
    ImageDenoiser,
    LBPTransformer,
    denoise_image,
    lbp_histogram,
    lbp_n_bins,
    smote_images_and_tabular,
)

SHAPE = (16, 20)

# Arbitrary test settings for each denoising method.
SETTINGS = {
    "median": {"disk_radius": 1},
    "wiener": {"window": 3},
    "nl_means": {"h_coef": 1.0, "patch_size": 3, "patch_distance": 4},
}


@pytest.fixture
def images():
    rng = np.random.default_rng(0)
    return rng.gamma(4.0, 50.0, (6, *SHAPE)).astype(np.float32)


@pytest.mark.parametrize("method, P", [("uniform", 8), ("nri_uniform", 8), ("uniform", 16), ("default", 8)])
def test_lbp_histogram_has_fixed_length(images, method, P):
    hists = [lbp_histogram(img, P, 1, method) for img in images]
    assert {len(h) for h in hists} == {lbp_n_bins(P, method)}
    for h in hists:
        assert h.sum() == pytest.approx(1.0)


def test_lbp_transformer_keeps_tabular(images):
    tab = np.arange(12, dtype=float).reshape(6, 2)
    X = np.hstack([images.reshape(6, -1), tab])
    out = LBPTransformer(image_shape=SHAPE, P=8, R=1).fit_transform(X)
    assert out.shape == (6, lbp_n_bins(8, "uniform") + 2)
    np.testing.assert_array_equal(out[:, -2:], tab)


def test_lbp_constant_image_does_not_crash():
    assert np.isfinite(lbp_histogram(np.ones(SHAPE))).all()


@pytest.mark.parametrize("method", ["median", "wiener", "nl_means"])
def test_denoiser_treats_images_independently(images, method):
    """Each image is filtered on its own; the result must not depend on the other images in the batch."""
    X = images.reshape(len(images), -1)
    stacked = ImageDenoiser(image_shape=SHAPE, method=method, **SETTINGS[method]).fit_transform(X)
    single = ImageDenoiser(image_shape=SHAPE, method=method, **SETTINGS[method]).fit_transform(X[:1])
    np.testing.assert_allclose(stacked[:1], single, rtol=1e-5)
    np.testing.assert_allclose(stacked[0], denoise_image(images[0], method, **SETTINGS[method]).ravel(), rtol=1e-5)
    assert np.isfinite(stacked).all()


def test_denoiser_passes_tabular_through(images):
    tab = np.array([[1.0, -1.0]] * 6)
    X = np.hstack([images.reshape(6, -1), tab])
    out = ImageDenoiser(image_shape=SHAPE, method="median", **SETTINGS["median"]).fit_transform(X)
    np.testing.assert_array_equal(out[:, -2:], tab)


def test_denoiser_parallel_matches_serial(images):
    X = images.reshape(len(images), -1)
    serial = ImageDenoiser(image_shape=SHAPE, method="nl_means", **SETTINGS["nl_means"]).fit_transform(X)
    parallel = ImageDenoiser(image_shape=SHAPE, method="nl_means", n_jobs=2, **SETTINGS["nl_means"]).fit_transform(X)
    np.testing.assert_allclose(serial, parallel)


def test_unknown_denoiser():
    with pytest.raises(ValueError, match="Unknown denoising method"):
        denoise_image(np.ones(SHAPE), "gaussian_blur")


@pytest.mark.parametrize("method, missing", [("median", "disk_radius"), ("wiener", "window"),
                                             ("nl_means", "patch_distance")])
def test_denoiser_settings_are_required(images, method, missing):
    """There are no default filter settings; a missing one must be named in the error."""
    settings = {k: v for k, v in SETTINGS[method].items() if k != missing}
    with pytest.raises(ValueError, match=missing):
        denoise_image(images[0], method, **settings)
    with pytest.raises(ValueError, match=missing):
        ImageDenoiser(image_shape=SHAPE, method=method, **settings).fit(images.reshape(len(images), -1))


def test_denoiser_needs_method_and_image_shape(images):
    X = images.reshape(len(images), -1)
    with pytest.raises(ValueError, match="method"):
        ImageDenoiser(image_shape=SHAPE).fit(X)
    with pytest.raises(ValueError, match="image_shape"):
        ImageDenoiser(method="median", **SETTINGS["median"]).fit_transform(X)
    with pytest.raises(ValueError, match="image_shape"):
        LBPTransformer().fit_transform(X)


def test_smote_keeps_images_and_tabular_paired():
    """Each synthetic image must be paired with the tabular row created from the same neighbours."""
    rng = np.random.default_rng(0)
    n = 40
    images = rng.normal(size=(n, 4, 5)).astype(np.float32)
    y = np.array([0] * 30 + [1] * 10)
    # tabular = linear function of the image; SMOTE interpolation must preserve it
    tabular = images.reshape(n, -1).sum(axis=1, keepdims=True)
    img_res, tab_res, y_res = smote_images_and_tabular(images, tabular, y, random_state=0)
    assert np.bincount(y_res).tolist() == [30, 30]
    assert img_res.shape == (60, 4, 5)
    np.testing.assert_allclose(tab_res[:, 0], img_res.reshape(60, -1).sum(axis=1), rtol=1e-4, atol=1e-4)
