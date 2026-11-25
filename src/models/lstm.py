"""
LSTM Model for AQI Classification
====================================
Long Short-Term Memory network that captures long-range temporal
dependencies in the pollutant time series for AQI category prediction.
"""

import os
workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
os.environ.setdefault('KERAS_HOME', os.path.join(workspace_dir, '.keras'))
os.environ.setdefault('MPLCONFIGDIR', os.path.join(workspace_dir, '.matplotlib'))
os.makedirs(os.environ['KERAS_HOME'], exist_ok=True)
os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)

import tensorflow as tf
from tensorflow.keras import layers, models


def build_lstm_model(input_shape, num_classes=6, config=None):
    """
    Build an LSTM model for AQI category classification.

    Architecture (per synopsis):
    - LSTM → Dropout → LSTM → Dropout → Dense → Dropout → Dense (softmax)

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
        Compiled LSTM model.
    """
    if config is None:
        config = {
            "units": [128, 64],
            "dropout_rate": 0.3,
            "recurrent_dropout": 0.2,
            "dense_units": 128,
        }

    units = config.get("units", [128, 64])
    dropout_rate = config.get("dropout_rate", 0.3)
    recurrent_dropout = config.get("recurrent_dropout", 0.2)
    dense_units = config.get("dense_units", 128)

    model = models.Sequential(name="LSTM_AQI")

    # First LSTM layer — return sequences for stacking
    model.add(layers.LSTM(
        units=units[0],
        return_sequences=True,
        recurrent_dropout=recurrent_dropout,
        input_shape=input_shape,
        name="lstm_1"
    ))
    model.add(layers.Dropout(dropout_rate, name="dropout_1"))

    # Second LSTM layer — return final output only
    model.add(layers.LSTM(
        units=units[1],
        return_sequences=False,
        recurrent_dropout=recurrent_dropout,
        name="lstm_2"
    ))
    model.add(layers.Dropout(dropout_rate, name="dropout_2"))

    # Dense head
    model.add(layers.Dense(dense_units, activation="relu", name="dense_1"))
    model.add(layers.Dropout(dropout_rate, name="dense_dropout"))

    # Output layer
    model.add(layers.Dense(num_classes, activation="softmax", name="output"))

    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    print(f"\n[LSTM Model] Built successfully")
    print(f"  Input shape: {input_shape}")
    print(f"  Num classes: {num_classes}")
    print(f"  Total params: {model.count_params():,}")

    return model
