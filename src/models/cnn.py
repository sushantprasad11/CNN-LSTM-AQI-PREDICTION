"""
1D CNN Model for AQI Classification
=====================================
Convolutional Neural Network that extracts local/sequential patterns
from the time series input using 1D convolutions, followed by dense
layers for multi-class AQI category classification.
"""

import os
workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
os.environ.setdefault('KERAS_HOME', os.path.join(workspace_dir, '.keras'))
os.environ.setdefault('MPLCONFIGDIR', os.path.join(workspace_dir, '.matplotlib'))
os.makedirs(os.environ['KERAS_HOME'], exist_ok=True)
os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)

import tensorflow as tf
from tensorflow.keras import layers, models


def build_cnn_model(input_shape, num_classes=6, config=None):
    """
    Build a 1D CNN model for AQI category classification.

    Architecture (per synopsis):
    - Conv1D → BatchNorm → ReLU → Conv1D → BatchNorm → ReLU → MaxPool
    - Flatten → Dense → Dropout → Dense (softmax)

    Parameters
    ----------
    input_shape : tuple
        Shape of input sequences: (window_size, n_features).
    num_classes : int
        Number of AQI categories (default: 6).
    config : dict, optional
        Model hyperparameters from config.yaml.

    Returns
    -------
    tf.keras.Model
        Compiled CNN model.
    """
    if config is None:
        config = {
            "filters": [64, 128],
            "kernel_size": 3,
            "pool_size": 2,
            "dense_units": 128,
            "dropout_rate": 0.3,
        }

    filters = config.get("filters", [64, 128])
    kernel_size = config.get("kernel_size", 3)
    pool_size = config.get("pool_size", 2)
    dense_units = config.get("dense_units", 128)
    dropout_rate = config.get("dropout_rate", 0.3)

    model = models.Sequential(name="CNN_AQI")

    # First convolutional block
    model.add(layers.Conv1D(
        filters=filters[0], kernel_size=kernel_size,
        padding="same", input_shape=input_shape,
        name="conv1d_1"
    ))
    model.add(layers.BatchNormalization(name="bn_1"))
    model.add(layers.Activation("relu", name="relu_1"))

    # Second convolutional block
    model.add(layers.Conv1D(
        filters=filters[1], kernel_size=kernel_size,
        padding="same", name="conv1d_2"
    ))
    model.add(layers.BatchNormalization(name="bn_2"))
    model.add(layers.Activation("relu", name="relu_2"))

    # Max pooling
    model.add(layers.MaxPooling1D(pool_size=pool_size, name="maxpool"))

    # Dropout after convolutional layers (per synopsis)
    model.add(layers.Dropout(dropout_rate, name="conv_dropout"))

    # Flatten and dense head
    model.add(layers.Flatten(name="flatten"))
    model.add(layers.Dense(dense_units, activation="relu", name="dense_1"))
    model.add(layers.Dropout(dropout_rate, name="dense_dropout"))

    # Output layer — softmax for multi-class classification
    model.add(layers.Dense(num_classes, activation="softmax", name="output"))

    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    print(f"\n[CNN Model] Built successfully")
    print(f"  Input shape: {input_shape}")
    print(f"  Num classes: {num_classes}")
    print(f"  Total params: {model.count_params():,}")

    return model
