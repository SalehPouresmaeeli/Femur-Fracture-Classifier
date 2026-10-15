"""Cleaning and encoding of the ``Fracture type`` column.

The raw spreadsheet contains many spellings of the same fracture type
(``IT-POST-less L``, ``TC_SN``, ``IT--L``...). Only the leading type code
matters, so each value is reduced to that code:

* IT, PT  -> trochanteric fractures
* SC, TC  -> neck fractures (``TS`` is a typo for ``TC``)
* ST      -> dropped by default (only a handful of rows)
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

TYPE_ALIASES = {"TS": "TC"}
KNOWN_TYPES = {"IT", "PT", "SC", "TC", "ST"}
BINARY_CLASSES = {"Neck": ("SC", "TC"), "Trochanteric": ("IT", "PT")}

_CODE_RE = re.compile(r"[A-Za-z]+")


def canonical_fracture_type(raw: str) -> str:
    """Reduce a raw fracture-type string to its type code (``'IT-POST' -> 'IT'``)."""
    match = _CODE_RE.search(str(raw).strip())
    if not match:
        raise ValueError(f"Unrecognised fracture type: {raw!r}")
    code = match.group(0).upper()
    code = TYPE_ALIASES.get(code, code)
    if code not in KNOWN_TYPES:
        raise ValueError(f"Unrecognised fracture type: {raw!r} (code {code!r})")
    return code


def encode_labels(fracture_types: pd.Series, scheme: str = "binary") -> tuple[np.ndarray, list[str]]:
    """Map canonical fracture types to integer labels.

    Class indices follow the alphabetical order of class names
    (binary: 0 = Neck, 1 = Trochanteric).
    """
    if scheme == "binary":
        lookup = {code: name for name, codes in BINARY_CLASSES.items() for code in codes}
        unmapped = set(fracture_types) - set(lookup)
        if unmapped:
            raise ValueError(f"Fracture types {sorted(unmapped)} have no binary class; drop them first")
        names = fracture_types.map(lookup)
    elif scheme == "detailed":
        names = fracture_types
    else:
        raise ValueError(f"Unknown label scheme {scheme!r}; use 'binary' or 'detailed'")

    class_names = sorted(names.unique())
    index = {name: i for i, name in enumerate(class_names)}
    return names.map(index).to_numpy(dtype=np.int64), class_names
