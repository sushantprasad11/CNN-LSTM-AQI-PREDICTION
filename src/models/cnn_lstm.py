"""
Hybrid CNN-LSTM Model for AQI Classification
===============================================
Combines Convolutional Neural Network (for local feature extraction)
with Long Short-Term Memory (for temporal dependency modelling) into
a single hybrid architecture — the core model of this project.

The CNN component extracts salient local features from the input sequence,
while the LSTM component models how these features evolve over time.
"""

import os
workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
os.environ.setdefault('KERAS_HOME', os.path.join(workspace_dir, '.keras'))
os.environ.setdefault('MPLCONFIGDIR', os.path.join(workspace_dir, '.matplotlib'))
os.makedirs(os.environ['KERAS_HOME'], exist_ok=True)
os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)

import tensorflow as tf
from tensorflow.keras import layers, models


def build_cnn_lstm_model(input_shape, num_classes=6, config=None):
    """
    Build a Hybrid CNN-LSTM model for AQI category classification.

    Architecture (per synopsis):
    - Conv1D → BatchNorm → ReLU → Conv1D → BatchNorm → ReLU → MaxPool → Dropout
    - LSTM → Dropout → LSTM → Dropout
    - Dense → Dropout → Dense (softmax)

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
        Compiled Hybrid CNN-LSTM model.
    """
    if config is None:
        config = {
            "cnn_filters": [64, 128],
            "cnn_kernel_size": 3,
            "cnn_pool_size": 2,
            "lstm_units": [128, 64],
            "lstm_dropout": 0.3,
            "recurrent_dropout": 0.2,
            "dense_units": 128,
            "dropout_rate": 0.3,
        }

    cnn_filters = config.get("cnn_filters", [64, 128])
    kernel_size = config.get("cnn_kernel_size", 3)
    pool_size = config.get("cnn_pool_size", 2)
    lstm_units = config.get("lstm_units", [128, 64])
    lstm_dropout = config.get("lstm_dropout", 0.3)
    recurrent_dropout = config.get("recurrent_dropout", 0.2)
    dense_units = config.get("dense_units", 128)
    dropout_rate = config.get("dropout_rate", 0.3)

    model = models.Sequential(name="Hybrid_CNN_LSTM_AQI")

    # ── CNN Component ──────────────────────────────────────────
    # First convolutional block
    model.add(layers.Conv1D(
        filters=cnn_filters[0], kernel_size=kernel_size,
        padding="same", input_shape=input_shape,
        name="conv1d_1"
    ))
    model.add(layers.BatchNormalization(name="bn_1"))
    model.add(layers.Activation("relu", name="relu_1"))

    # Second convolutional block
    model.add(layers.Conv1D(
        filters=cnn_filters[1], kernel_size=kernel_size,
        padding="same", name="conv1d_2"
    ))
    model.add(layers.BatchNormalization(name="bn_2"))
    model.add(layers.Activation("relu", name="relu_2"))

    # Max pooling to reduce temporal dimension
    model.add(layers.MaxPooling1D(pool_size=pool_size, name="maxpool"))

    # Dropout after CNN layers (per synopsis)
    model.add(layers.Dropout(dropout_rate, name="cnn_dropout"))

    # ── LSTM Component ─────────────────────────────────────────
    # The output of CNN is a 3D tensor — feeds directly into LSTM
    # First LSTM layer — return sequences for stacking
    model.add(layers.LSTM(
        units=lstm_units[0],
        return_sequences=True,
        recurrent_dropout=recurrent_dropout,
        name="lstm_1"
    ))
    model.add(layers.Dropout(lstm_dropout, name="lstm_dropout_1"))

    # Second LSTM layer — return final output only
    model.add(layers.LSTM(
        units=lstm_units[1],
        return_sequences=False,
        recurrent_dropout=recurrent_dropout,
        name="lstm_2"
    ))
    model.add(layers.Dropout(lstm_dropout, name="lstm_dropout_2"))

    # ── Dense Classification Head ──────────────────────────────
    model.add(layers.Dense(dense_units, activation="relu", name="dense_1"))
    model.add(layers.Dropout(dropout_rate, name="dense_dropout"))

    # Output layer — softmax for 6-class classification
    model.add(layers.Dense(num_classes, activation="softmax", name="output"))

    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    print(f"\n[Hybrid CNN-LSTM Model] Built successfully")
    print(f"  Input shape: {input_shape}")
    print(f"  Num classes: {num_classes}")
    print(f"  Total params: {model.count_params():,}")

    return model
