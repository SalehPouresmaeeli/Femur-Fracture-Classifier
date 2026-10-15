import shutil
from pathlib import Path

import pytest

from femur_fx.config import SearchConfig, apply_overrides, config_from_dict, load_config, load_raw_config

PROJECT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT / "configs"
# Private configs (git-ignored) are validated too when they exist locally.
PRIVATE_EXPERIMENTS = sorted((CONFIG_DIR / "experiments").glob("*.yaml"))


@pytest.fixture
def fresh_configs(tmp_path):
    """The public configs as a new user gets them, before creating configs/data.yaml."""
    configs = tmp_path / "configs"
    configs.mkdir()
    for name in ("example.yaml", "data.example.yaml"):
        shutil.copy(CONFIG_DIR / name, configs / name)
    return configs


def test_example_runs_after_copying_the_data_template(fresh_configs):
    shutil.copy(fresh_configs / "data.example.yaml", fresh_configs / "data.yaml")
    cfg = load_config(fresh_configs / "example.yaml")
    assert cfg.name == "example"
    assert cfg.model.type == "random_forest"
    assert Path(cfg.data.excel_path) == fresh_configs.parent / "data" / "table.xlsx"


def test_missing_data_config_points_to_the_template(fresh_configs):
    with pytest.raises(FileNotFoundError, match="data.example.yaml"):
        load_config(fresh_configs / "example.yaml")


@pytest.mark.parametrize("path", PRIVATE_EXPERIMENTS, ids=lambda p: p.stem)
def test_private_experiments_load(path):
    cfg = load_config(path)
    assert cfg.name == path.stem
    assert cfg.data.excel_path and cfg.data.image_dir
    assert cfg.model.type in {"knn", "random_forest", "cnn"}


def test_base_chain_merges_nested_values(tmp_path):
    (tmp_path / "grandparent.yaml").write_text(
        "data: {excluded_patients: [1, 2]}\n"
        "model: {type: random_forest, params: {n_estimators: 10, max_depth: 3}}\n")
    (tmp_path / "parent.yaml").write_text(
        "base: grandparent.yaml\npreprocessing: {lbp: {P: 4, R: 1}}\nmodel: {params: {n_estimators: 20}}\n")
    (tmp_path / "child.yaml").write_text(
        "base: parent.yaml\nname: child\nevaluation: {strategy: group_shuffle_split, seeds: [1, 2]}\n")

    cfg = load_config(tmp_path / "child.yaml")
    assert cfg.model.params == {"n_estimators": 20, "max_depth": 3}  # nested dicts merge, parent wins
    assert cfg.model.type == "random_forest"  # from grandparent
    assert cfg.preprocessing.lbp == {"P": 4, "R": 1}  # from parent
    assert cfg.evaluation.seeds == [1, 2]
    assert cfg.data.excluded_patients == [1, 2]


def test_overrides():
    raw = apply_overrides({"data": {"image_dir": "a"}}, ["data.image_dir=/x/y", "evaluation.n_splits=3",
                                                         "preprocessing.denoise={method: median}"])
    cfg = config_from_dict(raw)
    assert cfg.data.image_dir == "/x/y"
    assert cfg.evaluation.n_splits == 3
    assert cfg.preprocessing.denoise == {"method": "median"}


def test_unknown_key_is_rejected():
    with pytest.raises(ValueError, match="evaluation"):
        config_from_dict({"evaluation": {"n_split": 3}})


def test_search_section_is_optional():
    assert config_from_dict({}).search is None
    assert isinstance(config_from_dict({"search": {"kind": "random"}}).search, SearchConfig)


def test_data_template_uses_project_data_folder():
    cfg = config_from_dict(load_raw_config(CONFIG_DIR / "data.example.yaml"))
    assert Path(cfg.data.excel_path) == PROJECT / "data" / "table.xlsx"
    assert Path(cfg.data.image_dir) == PROJECT / "data" / "images"
    assert Path(cfg.data.cache_dir) == PROJECT / ".cache"
    assert Path(cfg.output_dir) == PROJECT / "outputs"
    assert cfg.data.excluded_patients == []


def test_relative_paths_resolve_against_defining_file(tmp_path, monkeypatch):
    shared = tmp_path / "configs"
    (shared / "exp").mkdir(parents=True)
    (shared / "data.yaml").write_text("output_dir: ../out\ndata: {image_dir: ../imgs, excel_path: D:/abs/t.xlsx}\n")
    (shared / "exp" / "a.yaml").write_text("base: ../data.yaml\nname: a\ndata: {cache_dir: cache}\n")
    monkeypatch.chdir(tmp_path / "configs" / "exp")  # working directory must not matter

    cfg = load_config(shared / "exp" / "a.yaml", ["data.excel_path=rel.xlsx"])
    assert Path(cfg.output_dir) == tmp_path / "out"
    assert Path(cfg.data.image_dir) == tmp_path / "imgs"  # inherited: relative to data.yaml
    assert Path(cfg.data.cache_dir) == shared / "exp" / "cache"  # relative to a.yaml
    assert cfg.data.excel_path == "rel.xlsx"  # --set values stay relative to the working directory


def test_base_key_is_removed(tmp_path):
    (tmp_path / "a.yaml").write_text("name: a\nseed: 1\n")
    (tmp_path / "b.yaml").write_text("base: a.yaml\nname: b\n")
    assert load_raw_config(tmp_path / "b.yaml") == {"name": "b", "seed": 1}
