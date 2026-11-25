"""
Confusion Matrix Visualization
================================
Generates and plots confusion matrices for AQI category classification,
including normalized and raw count views.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os


AQI_LABELS = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]


def plot_confusion_matrix(y_true, y_pred, model_name="Model",
                          normalize=True, save_path=None, figsize=(10, 8)):
    """
    Plot a confusion matrix heatmap.

    Parameters
    ----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.
    model_name : str
        Model name for the title.
    normalize : bool
        If True, normalize rows to show percentages.
    save_path : str, optional
        If provided, save the figure.
    figsize : tuple
        Figure size.

    Returns
    -------
    np.ndarray
        The confusion matrix.
    """
    # Get unique labels present in the data
    unique_labels = sorted(set(y_true) | set(y_pred))
    display_labels = [AQI_LABELS[i] if i < len(AQI_LABELS) else f"Class_{i}"
                      for i in unique_labels]

    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)

    if normalize:
        # Normalize by row (true labels)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_display = np.divide(cm.astype(float), row_sums,
                               where=row_sums != 0, out=np.zeros_like(cm, dtype=float))
        fmt = ".2%"
        title_suffix = "(Normalized)"
    else:
        cm_display = cm
        fmt = "d"
        title_suffix = "(Counts)"

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm_display,
        annot=True, fmt=fmt,
        cmap="Blues",
        xticklabels=display_labels,
        yticklabels=display_labels,
        square=True,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_xlabel("Predicted Category", fontsize=12)
    ax.set_ylabel("True Category", fontsize=12)
    ax.set_title(f"{model_name} — Confusion Matrix {title_suffix}", fontsize=14)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[CM] Saved to: {save_path}")

    plt.show()
    return cm


def plot_side_by_side_cm(y_true, y_pred, model_name="Model", save_path=None):
    """
    Plot both raw counts and normalized confusion matrices side by side.

    Parameters
    ----------
    y_true, y_pred : array-like
        True and predicted labels.
    model_name : str
        Model name for titles.
    save_path : str, optional
        Save path for the combined figure.

    Returns
    -------
    np.ndarray
        Raw confusion matrix.
    """
    unique_labels = sorted(set(y_true) | set(y_pred))
    display_labels = [AQI_LABELS[i] if i < len(AQI_LABELS) else f"Class_{i}"
                      for i in unique_labels]

    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm.astype(float), row_sums,
                        where=row_sums != 0, out=np.zeros_like(cm, dtype=float))

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Raw counts
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=display_labels, yticklabels=display_labels,
                square=True, linewidths=0.5, ax=axes[0])
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    axes[0].set_title(f"{model_name} — Counts")

    # Normalized
    sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Oranges",
                xticklabels=display_labels, yticklabels=display_labels,
                square=True, linewidths=0.5, ax=axes[1])
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")
    axes[1].set_title(f"{model_name} — Normalized")

    plt.suptitle(f"{model_name} — Confusion Matrix Analysis", fontsize=15, y=1.02)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[CM] Saved to: {save_path}")

    plt.show()
    return cm
