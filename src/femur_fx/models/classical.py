"""Classical (sklearn) model pipelines.

Every pipeline consumes the ``[pixels..., tabular...]`` matrix produced by
:meth:`FemurDataset.to_matrix` and is built from optional steps::

    smote -> denoise -> lbp -> clf

SMOTE is an imblearn pipeline step, so it only resamples data the pipeline
is *fitted* on. Cross-validation and hyper-parameter search therefore never
see synthetic samples in their validation folds.
"""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from femur_fx.config import ModelConfig, PreprocessingConfig
from femur_fx.preprocessing import ImageDenoiser, LBPTransformer, make_smote

CLASSIFIERS = {
    "knn": (KNeighborsClassifier, {"n_jobs": -1}),
    "random_forest": (RandomForestClassifier, {"class_weight": "balanced", "random_state": 0, "n_jobs": -1}),
}


def make_classifier(model_type: str, params: dict | None = None):
    try:
        cls, defaults = CLASSIFIERS[model_type]
    except KeyError:
        raise ValueError(f"Unknown classifier {model_type!r}; choose from {sorted(CLASSIFIERS)}") from None
    return cls(**{**defaults, **(params or {})})


def build_classical_pipeline(
    model: ModelConfig, preprocessing: PreprocessingConfig, image_shape: tuple[int, int]
):
    from imblearn.pipeline import Pipeline

    steps = []
    if preprocessing.smote is not None:
        steps.append(("smote", make_smote(**preprocessing.smote)))
    if preprocessing.denoise is not None:
        steps.append(("denoise", ImageDenoiser(image_shape=image_shape, **preprocessing.denoise)))
    if preprocessing.lbp is not None:
        steps.append(("lbp", LBPTransformer(image_shape=image_shape, **preprocessing.lbp)))
    steps.append(("clf", make_classifier(model.type, model.params)))
    return Pipeline(steps)


def set_random_state(estimator, seed: int):
    """Set every ``random_state`` parameter of a (nested) estimator to ``seed``."""
    params = {k: seed for k in estimator.get_params(deep=True) if k == "random_state" or k.endswith("__random_state")}
    return estimator.set_params(**params)
