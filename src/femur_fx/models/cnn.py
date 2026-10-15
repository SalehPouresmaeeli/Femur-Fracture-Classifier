"""Configurable Keras CNN.

No architecture is defined in code: the layers come from the ``model.architecture``
section of an experiment config, for example::

    model:
      type: cnn
      architecture:
        conv_layers:          # one entry per conv layer: [filters, max-pool after it?]
          - [16, true]
          - [32, true]
        padding: valid        # "valid" or "same"
        dense_units: 64       # dense layer after the image branch
        dropout: 0.5          # dropout after each dense block
        tabular_units: [16]   # optional dense branch for tabular features; omit for image-only
        head_units: 64        # dense layer after joining images and tabular features

TensorFlow is imported lazily so the rest of the package works without it.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

IMAGE_INPUT = "image_input"
TABULAR_INPUT = "additional_input"


@dataclass(frozen=True)
class CNNArchitecture:
    conv_layers: tuple[tuple[int, bool], ...]
    dense_units: int
    padding: str = "valid"
    kernel_size: int = 3
    pool_size: int = 2
    dropout: float = 0.5
    tabular_units: tuple[int, ...] | None = None
    head_units: int | None = None

    @property
    def uses_tabular(self) -> bool:
        return self.tabular_units is not None

    @classmethod
    def from_config(cls, spec: dict) -> CNNArchitecture:
        """Validate a ``model.architecture`` dict from a config file."""
        if not isinstance(spec, dict):
            raise ValueError("model.architecture must be a mapping of CNN settings (see femur_fx.models.cnn)")
        known = {f.name for f in fields(cls)}
        unknown = set(spec) - known
        if unknown:
            raise ValueError(f"Unknown key(s) in model.architecture: {sorted(unknown)}; allowed: {sorted(known)}")
        missing = {"conv_layers", "dense_units"} - set(spec)
        if missing:
            raise ValueError(f"model.architecture is missing {sorted(missing)}")

        conv_layers = tuple((int(filters), bool(pool)) for filters, pool in spec["conv_layers"])
        if not conv_layers:
            raise ValueError("model.architecture.conv_layers needs at least one layer")
        if spec.get("padding", "valid") not in ("valid", "same"):
            raise ValueError("model.architecture.padding must be 'valid' or 'same'")
        tabular = spec.get("tabular_units")
        if tabular is not None and spec.get("head_units") is None:
            raise ValueError("model.architecture.head_units is required when tabular_units is set")
        return cls(**{
            **spec,
            "conv_layers": conv_layers,
            "tabular_units": None if tabular is None else tuple(int(u) for u in tabular),
        })


def weighted_f1_score(y_true, y_pred):
    """Support-weighted F1 of one-hot targets vs. rounded softmax outputs (Keras metric)."""
    import tensorflow as tf

    eps = tf.keras.backend.epsilon()
    y_true = tf.cast(y_true, y_pred.dtype)
    y_pred = tf.round(tf.clip_by_value(y_pred, 0, 1))
    tp = tf.reduce_sum(y_true * y_pred, axis=0)
    precision = tp / (tf.reduce_sum(y_pred, axis=0) + eps)
    recall = tp / (tf.reduce_sum(y_true, axis=0) + eps)
    f1_per_class = 2 * precision * recall / (precision + recall + eps)
    weights = tf.reduce_sum(y_true, axis=0) / tf.reduce_sum(y_true)
    return tf.reduce_sum(f1_per_class * weights)


def build_cnn(
    architecture: CNNArchitecture,
    image_shape: tuple[int, int],
    num_classes: int,
    num_tabular: int = 0,
    learning_rate: float = 1e-3,
    name: str = "cnn",
):
    """Build and compile a CNN from ``architecture``."""
    import tensorflow as tf
    from tensorflow.keras import layers

    arch = architecture
    image_input = layers.Input(shape=(*image_shape, 1), name=IMAGE_INPUT)
    x = image_input
    for filters, pool in arch.conv_layers:
        x = layers.Conv2D(filters, (arch.kernel_size, arch.kernel_size), activation="relu", padding=arch.padding)(x)
        if pool:
            x = layers.MaxPooling2D(pool_size=(arch.pool_size, arch.pool_size))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(arch.dense_units, activation="relu")(x)
    head = layers.Dropout(arch.dropout)(x)

    if arch.uses_tabular:
        if num_tabular < 1:
            raise ValueError("This architecture has tabular_units, so it needs data.tabular_features")
        tabular_input = layers.Input(shape=(num_tabular,), name=TABULAR_INPUT)
        t = tabular_input
        for units in arch.tabular_units:
            t = layers.Dense(units, activation="relu")(t)
        head = layers.concatenate([head, t])
        head = layers.Dense(arch.head_units, activation="relu")(head)
        head = layers.Dropout(arch.dropout)(head)
        inputs = {IMAGE_INPUT: image_input, TABULAR_INPUT: tabular_input}
    else:
        inputs = image_input

    output = layers.Dense(num_classes, activation="softmax")(head)
    model = tf.keras.Model(inputs=inputs, outputs=output, name=name)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=[weighted_f1_score],
    )
    return model
