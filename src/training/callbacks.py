"""
Training Callbacks
====================
Custom Keras callbacks for early stopping and model checkpointing
as specified in the synopsis.
"""

import os
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


def get_callbacks(model_name, config=None, checkpoint_dir="models"):
    """
    Create a list of training callbacks.

    Parameters
    ----------
    model_name : str
        Name of the model (used for checkpoint filenames).
        e.g., 'cnn', 'lstm', 'cnn_lstm'
    config : dict, optional
        Training configuration with early_stopping/checkpoint params.
    checkpoint_dir : str
        Directory to save model checkpoints.

    Returns
    -------
    list
        List of Keras callbacks.
    """
    if config is None:
        config = {
            "early_stopping": {
                "monitor": "val_loss",
                "patience": 15,
                "restore_best_weights": True,
            },
            "checkpoint": {
                "monitor": "val_loss",
                "save_best_only": True,
            }
        }

    es_config = config.get("early_stopping", {})
    ckpt_config = config.get("checkpoint", {})

    callbacks = []

    # Early Stopping — stop training when validation loss stops improving
    early_stop = EarlyStopping(
        monitor=es_config.get("monitor", "val_loss"),
        patience=es_config.get("patience", 15),
        restore_best_weights=es_config.get("restore_best_weights", True),
        verbose=1,
    )
    callbacks.append(early_stop)

    # Model Checkpoint — save the best model weights
    checkpoint_path = os.path.join(checkpoint_dir, model_name, "best_model.keras")
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    checkpoint = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor=ckpt_config.get("monitor", "val_loss"),
        save_best_only=ckpt_config.get("save_best_only", True),
        verbose=1,
    )
    callbacks.append(checkpoint)

    # Learning Rate Reduction — reduce LR when loss plateaus
    lr_reducer = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=7,
        min_lr=1e-6,
        verbose=1,
    )
    callbacks.append(lr_reducer)

    print(f"[Callbacks] Created {len(callbacks)} callbacks for '{model_name}'")
    print(f"  Checkpoint path: {checkpoint_path}")

    return callbacks
