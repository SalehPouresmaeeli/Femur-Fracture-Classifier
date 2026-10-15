"""CNN tests with small made-up architectures; skipped when TensorFlow is not installed."""

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from femur_fx.config import EvaluationConfig, ExperimentConfig, ModelConfig, PreprocessingConfig, TrainingConfig  # noqa: E402
from femur_fx.experiments import run_experiment  # noqa: E402
from femur_fx.models.cnn import CNNArchitecture, build_cnn, weighted_f1_score  # noqa: E402

IMAGE_ONLY = {"conv_layers": [[8, True], [16, False]], "padding": "same", "dense_units": 16, "dropout": 0.25}
WITH_TABULAR = {**IMAGE_ONLY, "tabular_units": [8], "head_units": 16}


def test_builder_follows_the_architecture():
    """Parameter counts worked out by hand for a (16, 20) image, so every setting must be honoured."""
    # conv 1x3x3x8 + 8 = 80 | pool -> 8x10 | conv 8x3x3x16 + 16 = 1168 | flatten 8*10*16 = 1280
    # dense 1280*16 + 16 = 20496 | output 16*2 + 2 = 34
    image_only = build_cnn(CNNArchitecture.from_config(IMAGE_ONLY), (16, 20), num_classes=2)
    assert image_only.count_params() == 80 + 1168 + 20496 + 34

    # + tabular branch 3*8 + 8 = 32 | head (16 + 8)*16 + 16 = 400
    with_tabular = build_cnn(CNNArchitecture.from_config(WITH_TABULAR), (16, 20), num_classes=2, num_tabular=3)
    assert with_tabular.count_params() == 80 + 1168 + 20496 + 32 + 400 + 34
    assert with_tabular.output_shape == (None, 2)


def test_valid_padding_shrinks_feature_maps():
    same = build_cnn(CNNArchitecture.from_config(IMAGE_ONLY), (16, 20), num_classes=2)
    valid = build_cnn(CNNArchitecture.from_config({**IMAGE_ONLY, "padding": "valid"}), (16, 20), num_classes=2)
    assert valid.count_params() < same.count_params()


@pytest.mark.parametrize("spec, message", [
    ("cnn_1", "mapping"),
    ({"dense_units": 16}, "conv_layers"),
    ({**IMAGE_ONLY, "filters": 3}, "Unknown key"),
    ({**IMAGE_ONLY, "padding": "full"}, "padding"),
    ({**IMAGE_ONLY, "conv_layers": []}, "at least one"),
    ({**IMAGE_ONLY, "tabular_units": [8]}, "head_units"),
])
def test_invalid_architectures_are_rejected(spec, message):
    with pytest.raises(ValueError, match=message):
        CNNArchitecture.from_config(spec)


def test_tabular_architecture_requires_features():
    with pytest.raises(ValueError, match="tabular_features"):
        build_cnn(CNNArchitecture.from_config(WITH_TABULAR), (16, 20), num_classes=2, num_tabular=0)


def test_weighted_f1_metric():
    y_true = tf.constant([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=tf.float32)
    perfect = float(weighted_f1_score(y_true, y_true))
    assert perfect == pytest.approx(1.0, abs=1e-5)


def test_cnn_experiment(tmp_path, data_config, dataset):
    cfg = ExperimentConfig(
        name="cnn_test", output_dir=str(tmp_path / "out"), data=data_config,
        model=ModelConfig(type="cnn", architecture=WITH_TABULAR),
        preprocessing=PreprocessingConfig(smote={"k_neighbors": 3}, image_scaling="max",
                                          denoise={"method": "nl_means", "h_coef": 1.0, "patch_size": 3,
                                                   "patch_distance": 4}),
        evaluation=EvaluationConfig(strategy="group_shuffle_split", seeds=[0], test_size=0.3),
        training=TrainingConfig(epochs=2, batch_size=8, validation_size=0.25),
    )
    run_dir = run_experiment(cfg, dataset)
    for name in ("metrics.json", "predictions.csv", "splits/run_01_seed_0_loss.png"):
        assert (run_dir / name).exists(), name
    model = tf.keras.models.load_model(run_dir / "models" / "run_01_seed_0.keras", compile=False)
    assert model.output_shape == (None, 2)


@pytest.mark.parametrize("name, spec", [("image_only", IMAGE_ONLY), ("with_tabular", WITH_TABULAR)])
def test_both_input_paths_train(tmp_path, data_config, dataset, name, spec):
    """One epoch with k-fold for the single-input and the two-input model."""
    cfg = ExperimentConfig(
        name=f"cnn_{name}", output_dir=str(tmp_path / "out"), data=data_config,
        model=ModelConfig(type="cnn", architecture=spec),
        preprocessing=PreprocessingConfig(image_scaling="standard"),
        evaluation=EvaluationConfig(strategy="group_kfold", n_splits=2),
        training=TrainingConfig(epochs=1, batch_size=16, save_models=False),
    )
    run_dir = run_experiment(cfg, dataset)
    predictions = np.loadtxt(run_dir / "predictions.csv", delimiter=",", skiprows=1, usecols=(-2, -1))
    assert len(predictions) == len(dataset)
    np.testing.assert_allclose(predictions.sum(axis=1), 1.0, atol=1e-5)  # softmax probabilities
