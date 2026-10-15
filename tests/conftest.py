"""Synthetic stand-in for the real data: small .vti images + an Excel table.

Each image carries its patient id in pixel (0, 0) and a rotation-dependent
value in pixel (0, 1), so tests can check that rows and images stay aligned.
Trochanteric patients get a bright blob, so models have something to learn.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from femur_fx.config import DataConfig

IMAGE_SHAPE = (16, 20)  # (rows, cols)
ROT_Y = [-10.0, 0.0, 10.0]
ROT_Z = [0.0, 10.0]
# patient id -> raw fracture type (deliberately messy spellings)
PATIENTS = {
    1: "TC-SN", 2: "TC_IN", 3: "SC-SN", 4: "TS-SN", 5: "TC",
    6: "IT-POST-less L", 7: "IT--L", 8: "PT", 9: "IT-ANT", 10: "TC",
    11: "TC",  # excluded patient
}
EXCLUDED = [11]


def write_vti(path, image: np.ndarray) -> None:
    import vtk
    from vtk.util import numpy_support

    rows, cols = image.shape
    data = vtk.vtkImageData()
    data.SetDimensions(cols, rows, 1)  # VTK is x-fastest: x = columns
    data.GetPointData().SetScalars(numpy_support.numpy_to_vtk(image.ravel().astype(np.float32), deep=True))
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(data)
    writer.Write()


def make_image(rng, patient: int, ry: float, rz: float, trochanteric: bool) -> np.ndarray:
    img = rng.gamma(4.0, 50.0, IMAGE_SHAPE).astype(np.float32) + 100
    img[:, -3:] = 0  # background
    if trochanteric:
        img[4:10, 5:12] += 600
    img[0, 0] = patient
    img[0, 1] = 1000 + ry * 10 + rz
    return img


@pytest.fixture(scope="session")
def synthetic_data(tmp_path_factory):
    root = tmp_path_factory.mktemp("femur")
    image_dir = root / "images"
    image_dir.mkdir()
    rng = np.random.default_rng(0)
    rows = []
    for pid, ftype in PATIENTS.items():
        trochanteric = ftype.startswith(("IT", "PT"))
        for ry in ROT_Y:
            for rz in ROT_Z:
                # one ST row for patient 5, which must be dropped
                rtype = "ST" if (pid == 5 and ry == 10 and rz == 10) else ftype
                rows.append({
                    "Patient_id": pid, "Age": 60 + pid, "Weight": 70.0, "Height": 170.0, "BMI": 24.2,
                    "Fracture Status": pid % 2, "T-score": -2.0 + pid / 10,
                    "Rotation around x-axis": 0, "Rotation around y-axis": int(ry),
                    "Rotation around z-axis": int(rz), "Fracture type": f" {rtype} ",
                })
                name = f"Hip{pid:03d}_[0.0, {ry}, {rz}].vti"
                write_vti(image_dir / name, make_image(rng, pid, ry, rz, trochanteric))
    excel = root / "table.xlsx"
    # Shuffle so that alignment cannot rely on row order.
    pd.DataFrame(rows).sample(frac=1, random_state=1).to_excel(excel, index=False)
    return {"excel_path": str(excel), "image_dir": str(image_dir), "root": root}


@pytest.fixture
def data_config(synthetic_data, tmp_path) -> DataConfig:
    return DataConfig(
        excel_path=synthetic_data["excel_path"],
        image_dir=synthetic_data["image_dir"],
        excluded_patients=EXCLUDED,
        tabular_features=["Loading_x", "Loading_y", "Loading_z"],
        cache_dir=str(tmp_path / "cache"),
    )


@pytest.fixture
def dataset(data_config):
    from femur_fx.data import build_dataset

    return build_dataset(data_config)
