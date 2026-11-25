"""
Evaluation Metrics Module
===========================
Computes precision, recall, F1-score, accuracy, and classification reports
for multi-class AQI category classification.
"""

import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    accuracy_score, classification_report
)


# AQI category labels for readable reports
AQI_LABELS = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]


def compute_metrics(y_true, y_pred, average="weighted"):
    """
    Compute precision, recall, F1, and accuracy.

    Parameters
    ----------
    y_true : array-like
        True class labels.
    y_pred : array-like
        Predicted class labels.
    average : str
        Averaging strategy for multi-class ('weighted', 'macro', 'micro').

    Returns
    -------
    dict
        Dictionary with precision, recall, f1, accuracy.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average=average, zero_division=0),
    }
    return metrics


def get_classification_report(y_true, y_pred, labels=None, target_names=None):
    """
    Generate a detailed classification report.

    Parameters
    ----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.
    labels : list, optional
        Label indices to include.
    target_names : list of str, optional
        Display names for each class.

    Returns
    -------
    str
        Formatted classification report.
    dict
        Report as a dictionary.
    """
    if target_names is None:
        # Determine which labels are present
        unique_labels = sorted(set(y_true) | set(y_pred))
        target_names = [AQI_LABELS[i] if i < len(AQI_LABELS) else f"Class_{i}"
                        for i in unique_labels]
        labels = unique_labels

    report_str = classification_report(
        y_true, y_pred,
        labels=labels,
        target_names=target_names,
        zero_division=0,
    )

    report_dict = classification_report(
        y_true, y_pred,
        labels=labels,
        target_names=target_names,
        zero_division=0,
        output_dict=True,
    )

    return report_str, report_dict


def print_model_metrics(model_name, y_true, y_pred):
    """
    Print a formatted metrics summary for a model.

    Parameters
    ----------
    model_name : str
        Name of the model.
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.

    Returns
    -------
    dict
        Computed metrics.
    """
    metrics = compute_metrics(y_true, y_pred)
    report_str, report_dict = get_classification_report(y_true, y_pred)

    print(f"\n{'=' * 60}")
    print(f"  {model_name} — Evaluation Results")
    print(f"{'=' * 60}")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f} (weighted)")
    print(f"  Recall:    {metrics['recall']:.4f} (weighted)")
    print(f"  F1-Score:  {metrics['f1_score']:.4f} (weighted)")
    print(f"\nDetailed Classification Report:")
    print(report_str)
    print(f"{'=' * 60}\n")

    metrics["report_dict"] = report_dict
    return metrics
