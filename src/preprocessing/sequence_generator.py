"""
Sequence Generator — Sliding Window
=====================================
Creates sliding window sequences from time series data for feeding
into CNN, LSTM, and Hybrid CNN-LSTM models.

Each sequence is a window of `window_size` consecutive timesteps,
and the target is the AQI category code at the next timestep.
"""

import numpy as np
import os


def create_sequences(features, targets, window_size=24):
    """
    Create sliding window sequences for time series classification.

    Parameters
    ----------
    features : np.ndarray
        2D array of shape (n_samples, n_features).
    targets : np.ndarray
        1D array of shape (n_samples,) containing integer class labels.
    window_size : int
        Number of consecutive timesteps per input sequence (default: 24 hours).

    Returns
    -------
    X : np.ndarray
        3D array of shape (n_sequences, window_size, n_features).
    y : np.ndarray
        1D array of shape (n_sequences,) — the target for each sequence.
    """
    X, y = [], []

    for i in range(window_size, len(features)):
        X.append(features[i - window_size:i])
        y.append(targets[i])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    print(f"[Sequences] Created {len(X)} sequences")
    print(f"  X shape: {X.shape}  (samples, timesteps, features)")
    print(f"  y shape: {y.shape}")
    print(f"  Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    return X, y


def split_sequences(X, y, test_size=0.30):
    """
    Split sequences into train/test sets (chronological — no shuffle).

    For time series data, we MUST split chronologically to avoid look-ahead bias.

    Parameters
    ----------
    X : np.ndarray
        3D feature sequences.
    y : np.ndarray
        1D target labels.
    test_size : float
        Fraction for the test set (default: 0.30 per synopsis).

    Returns
    -------
    X_train, X_test, y_train, y_test : np.ndarray
        Split arrays.
    """
    split_idx = int(len(X) * (1 - test_size))

    X_train = X[:split_idx]
    X_test = X[split_idx:]
    y_train = y[:split_idx]
    y_test = y[split_idx:]

    print(f"[Split] Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")
    print(f"  Train class dist: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    print(f"  Test class dist:  {dict(zip(*np.unique(y_test, return_counts=True)))}")

    return X_train, X_test, y_train, y_test


def save_sequences(X_train, X_test, y_train, y_test, filepath):
    """
    Save train/test sequences to a compressed .npz file.

    Parameters
    ----------
    X_train, X_test : np.ndarray
        Feature sequences.
    y_train, y_test : np.ndarray
        Target labels.
    filepath : str
        Output path (e.g., 'data/processed/sequences.npz').
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    np.savez_compressed(
        filepath,
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test
    )
    print(f"[Sequences] Saved to: {filepath}")


def load_sequences(filepath):
    """
    Load train/test sequences from a .npz file.

    Parameters
    ----------
    filepath : str
        Path to the .npz file.

    Returns
    -------
    X_train, X_test, y_train, y_test : np.ndarray
    """
    data = np.load(filepath)
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]

    print(f"[Sequences] Loaded from: {filepath}")
    print(f"  X_train: {X_train.shape}, X_test: {X_test.shape}")

    return X_train, X_test, y_train, y_test
