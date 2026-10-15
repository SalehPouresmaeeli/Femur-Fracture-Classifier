"""Reading CT images stored as VTK XML image data (.vti)."""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

ROTATION_COLUMNS = ["Rotation around x-axis", "Rotation around y-axis", "Rotation around z-axis"]

_PATIENT_RE = re.compile(r"Hip(\d+)", re.IGNORECASE)
_ROTATION_RE = re.compile(r"_\[([^\]]*)\]")


def parse_vti_filename(filename: str) -> tuple[int, tuple[float, float, float]]:
    """Extract the patient id and load-case rotation from a file name.

    ``Hip016_[0.0, -10.0, 20.0].vti`` -> ``(16, (0.0, -10.0, 20.0))``
    """
    patient = _PATIENT_RE.search(filename)
    rotation = _ROTATION_RE.search(filename)
    if not patient or not rotation:
        raise ValueError(f"Cannot parse patient id / rotation from file name: {filename}")
    values = tuple(float(v) for v in rotation.group(1).split(","))
    if len(values) != 3:
        raise ValueError(f"Expected three rotation angles in file name: {filename}")
    return int(patient.group(1)), values


def read_vti(path: str | Path) -> np.ndarray:
    """Read the scalar point data of a .vti file as a ``(rows, cols)`` array.

    VTK stores point data x-fastest, so a volume with dimensions
    ``(nx, ny, nz)`` maps to a C-ordered array of shape ``(nz, ny, nx)``.
    Singleton axes are dropped, leaving a ``(rows, cols)`` CT image.
    """
    import vtk
    from vtk.util import numpy_support

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    image_data = reader.GetOutput()
    if image_data is None:
        raise ValueError(f"Failed to read VTK image data from {path}")
    scalars = image_data.GetPointData().GetScalars()
    if scalars is None:
        raise ValueError(f"No scalar point data in {path}")

    nx, ny, nz = image_data.GetDimensions()
    array = numpy_support.vtk_to_numpy(scalars).reshape(nz, ny, nx)
    image = np.squeeze(array)
    if image.ndim != 2:
        raise ValueError(f"Expected a (rows, cols) CT image in {path}, got shape {array.shape}")
    return image.astype(np.float32)


def _directory_fingerprint(files: list[Path]) -> str:
    digest = hashlib.sha1()
    for f in files:
        stat = f.stat()
        digest.update(f"{f.name}|{stat.st_size}|{int(stat.st_mtime)}".encode())
    return digest.hexdigest()[:16]


def load_image_directory(
    image_dir: str | Path, cache_dir: str | Path | None = None
) -> tuple[pd.DataFrame, np.ndarray]:
    """Load every .vti file in ``image_dir``.

    Returns a metadata frame (``filename``, ``Patient_id`` and the three
    rotation columns), sorted by patient and rotation, and the matching
    ``(n_images, rows, cols)`` image stack.
    """
    image_dir = Path(image_dir)
    files = sorted(image_dir.glob("*.vti"))
    if not files:
        raise FileNotFoundError(f"No .vti files found in {image_dir}")

    cache_file = None
    if cache_dir is not None:
        cache_file = Path(cache_dir) / f"vti_{image_dir.name}_{_directory_fingerprint(files)}.npz"
        if cache_file.exists():
            log.info("Loading cached images from %s", cache_file)
            cached = np.load(cache_file, allow_pickle=False)
            meta = pd.DataFrame({"filename": cached["filename"], "Patient_id": cached["patient_id"]})
            for i, col in enumerate(ROTATION_COLUMNS):
                meta[col] = cached["rotation"][:, i]
            return meta, cached["images"]

    log.info("Reading %d .vti files from %s", len(files), image_dir)
    records, images = [], []
    for f in files:
        patient_id, rotation = parse_vti_filename(f.name)
        records.append((f.name, patient_id, *rotation))
        images.append(read_vti(f))

    shapes = {img.shape for img in images}
    if len(shapes) != 1:
        raise ValueError(f"Images in {image_dir} have different shapes: {sorted(shapes)}")

    meta = pd.DataFrame(records, columns=["filename", "Patient_id", *ROTATION_COLUMNS])
    order = meta.sort_values(["Patient_id", *ROTATION_COLUMNS]).index.to_numpy()
    meta = meta.loc[order].reset_index(drop=True)
    stack = np.stack(images)[order]

    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            cache_file,
            images=stack,
            filename=meta["filename"].to_numpy(dtype=str),
            patient_id=meta["Patient_id"].to_numpy(),
            rotation=meta[ROTATION_COLUMNS].to_numpy(),
        )
        log.info("Cached images to %s", cache_file)
    return meta, stack
