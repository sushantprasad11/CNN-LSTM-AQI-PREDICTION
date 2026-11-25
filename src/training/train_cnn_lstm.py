"""
Hybrid CNN-LSTM Training Module
==================================
Trains the Hybrid CNN-LSTM model on AQI sequence data.
"""

import tensorflow as tf
from src.models.cnn_lstm import build_cnn_lstm_model
from src.training.callbacks import get_callbacks


def train_cnn_lstm(X_train, y_train, X_test, y_test, config=None,
                   model_config=None, checkpoint_dir="models"):
    """
    Build and train the Hybrid CNN-LSTM model.

    Parameters
    ----------
    X_train : np.ndarray
        Training sequences (samples, timesteps, features).
    y_train : np.ndarray
        Training labels.
    X_test : np.ndarray
        Test sequences (used as validation during training).
    y_test : np.ndarray
        Test labels.
    config : dict, optional
        Training hyperparameters.
    model_config : dict, optional
        Hybrid CNN-LSTM architecture hyperparameters.
    checkpoint_dir : str
        Directory to save model checkpoints.

    Returns
    -------
    model : tf.keras.Model
        Trained Hybrid CNN-LSTM model.
    history : tf.keras.callbacks.History
        Training history object.
    """
    if config is None:
        config = {
            "epochs": 100,
            "batch_size": 64,
            "learning_rate": 0.001,
            "early_stopping": {"monitor": "val_loss", "patience": 15, "restore_best_weights": True},
            "checkpoint": {"monitor": "val_loss", "save_best_only": True},
        }

    input_shape = (X_train.shape[1], X_train.shape[2])
    num_classes = len(set(y_train))

    # Build model
    model = build_cnn_lstm_model(input_shape, num_classes, model_config)
    model.summary()

    # Get callbacks
    callbacks = get_callbacks("cnn_lstm", config, checkpoint_dir)

    # Train
    print(f"\n[Hybrid CNN-LSTM Training] Starting...")
    print(f"  Epochs: {config.get('epochs', 100)}")
    print(f"  Batch size: {config.get('batch_size', 64)}")

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=config.get("epochs", 100),
        batch_size=config.get("batch_size", 64),
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluate
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n[Hybrid CNN-LSTM Training] Complete!")
    print(f"  Test Loss: {loss:.4f}")
    print(f"  Test Accuracy: {accuracy:.4f}")

    return model, history
