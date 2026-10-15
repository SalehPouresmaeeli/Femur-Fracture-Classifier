"""End-to-end runs of the experiment runner on the synthetic dataset."""

import json

import joblib
import numpy as np
import pandas as pd
import pytest

from femur_fx.cli import main
from femur_fx.config import (
    DataConfig,
    EvaluationConfig,
    ExperimentConfig,
    ModelConfig,
    PreprocessingConfig,
    SearchConfig,
)
from femur_fx.evaluation import get_scorer, iter_splits, predict_labels
from femur_fx.experiments import run_experiment


def _cfg(tmp_path, data_config: DataConfig, **kwargs) -> ExperimentConfig:
    return ExperimentConfig(name="test", output_dir=str(tmp_path / "out"), data=data_config, **kwargs)


def test_group_splits_never_share_patients(dataset):
    for strategy in ("group_kfold", "group_shuffle_split"):
        cfg = EvaluationConfig(strategy=strategy, n_splits=5, seeds=[1, 2, 2])
        splits = list(iter_splits(cfg, dataset.y, dataset.groups, 0))
        assert len({s.name for s in splits}) == len(splits)  # duplicate seeds get distinct names
        for s in splits:
            assert not set(dataset.groups[s.train]) & set(dataset.groups[s.test])


def test_class_f1_scorer():
    from sklearn.dummy import DummyClassifier

    X, y = np.zeros((4, 1)), np.array([0, 0, 1, 1])
    always_one = DummyClassifier(strategy="constant", constant=1).fit(X, y)
    assert get_scorer("f1_class_1")(always_one, X, y) == pytest.approx(2 / 3)
    assert get_scorer("f1_class_0")(always_one, X, y) == 0.0


def test_predict_labels_threshold():
    proba = np.array([[0.9, 0.1], [0.8, 0.2], [0.4, 0.6]])
    assert predict_labels(proba).tolist() == [0, 0, 1]
    assert predict_labels(proba, threshold=0.15).tolist() == [0, 1, 1]


def test_knn_search_and_kfold(tmp_path, data_config, dataset):
    cfg = _cfg(
        tmp_path, data_config,
        model=ModelConfig(type="knn"),
        preprocessing=PreprocessingConfig(
            smote={"sampling_strategy": "auto", "random_state": 0, "k_neighbors": 3},
            denoise={"method": "nl_means", "h_coef": 1.0, "patch_size": 3, "patch_distance": 4},
            lbp={"P": 8, "R": 1, "method": "uniform"},
        ),
        search=SearchConfig(kind="grid", scoring="f1_class_1", cv_splits=3, n_jobs=1,
                            param_grid={"clf__n_neighbors": [1, 3]}),
        evaluation=EvaluationConfig(strategy="group_kfold", n_splits=3),
    )
    run_dir = run_experiment(cfg, dataset)

    for name in ("config.yaml", "run.log", "dataset_summary.json", "metrics.json", "metrics_per_split.csv",
                 "predictions.csv", "confusion_matrix.png", "roc_curve.png", "probability_histograms.png",
                 "f1_boxplot.png", "model.joblib", "search/cv_results.csv", "search/best.json"):
        assert (run_dir / name).exists(), name

    predictions = pd.read_csv(run_dir / "predictions.csv")
    assert len(predictions) == len(dataset)  # k-fold: every sample tested exactly once
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert len(metrics["splits"]) == 3
    assert np.sum(metrics["confusion_matrix_total"]) == len(dataset)

    model = joblib.load(run_dir / "model.joblib")
    assert [name for name, _ in model.steps] == ["smote", "denoise", "lbp", "clf"]
    assert model.predict(dataset.to_matrix()[:5]).shape == (5,)


def test_random_forest_seeds_and_threshold(tmp_path, data_config, dataset):
    cfg = _cfg(
        tmp_path, data_config,
        model=ModelConfig(type="random_forest", params={"n_estimators": 20}),
        preprocessing=PreprocessingConfig(lbp={"P": 8, "R": 1, "method": "uniform"}),
        search=SearchConfig(kind="random", n_iter=2, scoring="f1_weighted", cv_splits=3, n_jobs=1,
                            param_grid={"clf__max_depth": {"distribution": "randint", "low": 2, "high": 6}}),
        evaluation=EvaluationConfig(strategy="group_shuffle_split", seeds=[3, 4], test_size=0.3,
                                    decision_threshold=0.3, refit_full=False),
    )
    run_dir = run_experiment(cfg, dataset)
    per_split = pd.read_csv(run_dir / "metrics_per_split.csv", index_col=0)
    assert list(per_split.index) == ["run_01_seed_3", "run_02_seed_4", "mean", "std"]
    assert not (run_dir / "model.joblib").exists()
    predictions = pd.read_csv(run_dir / "predictions.csv")
    expected = (predictions["proba_Trochanteric"] >= 0.3).astype(int)
    assert (predictions["y_pred"] == expected).all()


def test_cli_run_with_overrides(tmp_path, synthetic_data):
    config = tmp_path / "exp.yaml"
    config.write_text(
        "name: cli_test\n"
        "model: {type: random_forest, params: {n_estimators: 10}}\n"
        "data: {excluded_patients: [11], tabular_features: [Loading_x]}\n"
        "evaluation: {strategy: group_kfold, n_splits: 2}\n"
    )
    code = main(["run", str(config),
                 "--set", f"data.excel_path={synthetic_data['excel_path']}",
                 "--set", f"data.image_dir={synthetic_data['image_dir']}",
                 "--set", f"data.cache_dir={tmp_path / 'cache'}",
                 "--set", f"output_dir={tmp_path / 'out'}"])
    assert code == 0
    assert len(list((tmp_path / "out" / "cli_test").glob("*/metrics.json"))) == 1


def test_cli_rejects_unknown_only_name(tmp_path):
    config = tmp_path / "exp.yaml"
    config.write_text("name: real_name\n")
    assert main(["run", str(config), "--only", "real_name", "typo_name"]) == 1


def test_run_dirs_never_collide(tmp_path):
    from femur_fx.experiments.runner import make_run_dir

    cfg = ExperimentConfig(name="same", output_dir=str(tmp_path))
    dirs = {make_run_dir(cfg) for _ in range(3)}
    assert len(dirs) == 3


def test_cli_reports_failure(tmp_path):
    config = tmp_path / "bad.yaml"
    config.write_text(f"name: bad\noutput_dir: {tmp_path.as_posix()}/out\ndata: {{image_dir: {tmp_path.as_posix()}/missing}}\n")
    assert main(["run", str(config)]) == 1
