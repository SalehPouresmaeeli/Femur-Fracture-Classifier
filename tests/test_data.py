import numpy as np
import pandas as pd
import pytest

from conftest import IMAGE_SHAPE, PATIENTS
from femur_fx.data import build_dataset, canonical_fracture_type, encode_labels, parse_vti_filename, read_vti
from femur_fx.data.vti import load_image_directory


@pytest.mark.parametrize("raw, expected", [
    ("TC-SN", "TC"), ("TC_IN", "TC"), (" IT-POST-less L ", "IT"), ("IT--L", "IT"), ("IT -ANT", "IT"),
    ("TS-SN", "TC"), ("TS", "TC"), ("SC-SN", "SC"), ("PT", "PT"), ("ST", "ST"), ("IT-less L", "IT"),
])
def test_canonical_fracture_type(raw, expected):
    assert canonical_fracture_type(raw) == expected


def test_canonical_fracture_type_rejects_unknown():
    with pytest.raises(ValueError):
        canonical_fracture_type("XX-1")


def test_encode_labels_binary():
    y, names = encode_labels(pd.Series(["TC", "IT", "SC", "PT"]), "binary")
    assert names == ["Neck", "Trochanteric"]
    assert y.tolist() == [0, 1, 0, 1]


def test_encode_labels_binary_rejects_st():
    with pytest.raises(ValueError, match="ST"):
        encode_labels(pd.Series(["TC", "ST"]), "binary")


def test_parse_vti_filename():
    assert parse_vti_filename("Hip016_[0.0, -10.0, 20.0].vti") == (16, (0.0, -10.0, 20.0))
    assert parse_vti_filename("Hip016_scan_v2_[0.0, -10.0, 20.0].vti") == (16, (0.0, -10.0, 20.0))


def test_read_vti_keeps_orientation(tmp_path):
    from conftest import write_vti

    image = np.arange(np.prod(IMAGE_SHAPE), dtype=np.float32).reshape(IMAGE_SHAPE)
    write_vti(tmp_path / "Hip001_x_[0.0, 0.0, 0.0].vti", image)
    np.testing.assert_array_equal(read_vti(tmp_path / "Hip001_x_[0.0, 0.0, 0.0].vti"), image)


def test_image_cache_roundtrip(synthetic_data, tmp_path):
    meta1, imgs1 = load_image_directory(synthetic_data["image_dir"], tmp_path)
    assert len(list(tmp_path.glob("*.npz"))) == 1
    meta2, imgs2 = load_image_directory(synthetic_data["image_dir"], tmp_path)
    pd.testing.assert_frame_equal(meta1, meta2)
    np.testing.assert_array_equal(imgs1, imgs2)


def test_build_dataset(dataset):
    # 10 kept patients x 6 rotations, minus the ST row
    assert len(dataset) == 10 * 6 - 1
    assert dataset.image_shape == IMAGE_SHAPE
    assert 11 not in dataset.groups
    assert dataset.class_names == ["Neck", "Trochanteric"]
    md = dataset.metadata
    # each image belongs to its row: patient id and rotation are encoded in the pixels
    np.testing.assert_array_equal(dataset.images[:, 0, 0], md["Patient_id"])
    np.testing.assert_allclose(
        dataset.images[:, 0, 1], 1000 + md["Rotation around y-axis"] * 10 + md["Rotation around z-axis"])
    expected = [int(PATIENTS[p].startswith(("IT", "PT"))) for p in md["Patient_id"]]
    assert dataset.y.tolist() == expected


def test_loading_direction(dataset):
    md = dataset.metadata
    neutral = md[(md["Rotation around y-axis"] == 0) & (md["Rotation around z-axis"] == 0)]
    np.testing.assert_allclose(neutral[["Loading_x", "Loading_y", "Loading_z"]], [[-1, 0, 0]] * len(neutral), atol=1e-12)
    norms = np.linalg.norm(dataset.tabular.to_numpy(), axis=1)
    np.testing.assert_allclose(norms, 1.0)


def test_to_matrix_layout(dataset):
    X = dataset.to_matrix()
    n_pixels = int(np.prod(IMAGE_SHAPE))
    assert X.shape == (len(dataset), n_pixels + 3)
    np.testing.assert_allclose(X[:, n_pixels:], dataset.tabular.to_numpy())


def test_fixed_image_mode(data_config):
    data_config.image_mode = "fixed"
    ds = build_dataset(data_config)
    # every row shows the neutral (ry = rz = 0) image of its own patient
    np.testing.assert_array_equal(ds.images[:, 0, 0], ds.metadata["Patient_id"])
    np.testing.assert_array_equal(ds.images[:, 0, 1], 1000)


def test_missing_inputs_give_helpful_error(data_config, tmp_path):
    data_config.excel_path = str(tmp_path / "missing.xlsx")
    with pytest.raises(FileNotFoundError, match="data/README.md"):
        build_dataset(data_config)


def test_unknown_tabular_feature(data_config):
    data_config.tabular_features = ["Nope"]
    with pytest.raises(KeyError):
        build_dataset(data_config)
