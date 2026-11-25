"""
Prediction Module
===================
Handles loading trained models and generating predictions
for new or test data.
"""

import os
workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
os.environ.setdefault('KERAS_HOME', os.path.join(workspace_dir, '.keras'))
os.environ.setdefault('MPLCONFIGDIR', os.path.join(workspace_dir, '.matplotlib'))
os.makedirs(os.environ['KERAS_HOME'], exist_ok=True)
os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)

import numpy as np
import tensorflow as tf


AQI_LABELS = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]


def load_trained_model(model_path):
    """
    Load a trained Keras model from disk.

    Parameters
    ----------
    model_path : str
        Path to the saved .keras model file.

    Returns
    -------
    tf.keras.Model
        Loaded model.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = tf.keras.models.load_model(model_path)
    print(f"[Predict] Loaded model from: {model_path}")
    return model


def predict_aqi(model, X, return_probabilities=False):
    """
    Generate AQI category predictions.

    Parameters
    ----------
    model : tf.keras.Model
        Trained model.
    X : np.ndarray
        Input sequences (samples, timesteps, features).
    return_probabilities : bool
        If True, also return class probabilities.

    Returns
    -------
    np.ndarray
        Predicted class labels (integers 0-5).
    np.ndarray (optional)
        Class probabilities if return_probabilities=True.
    """
    probabilities = model.predict(X, verbose=0)
    predictions = np.argmax(probabilities, axis=1)

    # Print distribution
    unique, counts = np.unique(predictions, return_counts=True)
    print(f"[Predict] Generated {len(predictions)} predictions")
    for u, c in zip(unique, counts):
        label = AQI_LABELS[u] if u < len(AQI_LABELS) else f"Class_{u}"
        print(f"  {label}: {c} ({100 * c / len(predictions):.1f}%)")

    if return_probabilities:
        return predictions, probabilities
    return predictions


def get_best_model_path(models_dir="models"):
    """
    Find the best model checkpoint across all model types.

    Parameters
    ----------
    models_dir : str
        Base directory containing model subdirectories.

    Returns
    -------
    dict
        Mapping from model_name to checkpoint path.
    """
    model_paths = {}
    for model_name in ["cnn", "lstm", "cnn_lstm"]:
        path = os.path.join(models_dir, model_name, "best_model.keras")
        if os.path.exists(path):
            model_paths[model_name] = path
            print(f"[Predict] Found: {path}")
        else:
            print(f"[Predict] Not found: {path}")

    return model_paths
