"""
Early Warning Alert System (Extension 3)
==========================================
A threshold-based early-warning module that converts predicted AQI
categories into actionable health alerts.

Per the synopsis: an alert is generated whenever the predicted AQI
category falls into "Very Poor" (code 4) or "Severe" (code 5).

The module's reliability is itself evaluated using precision, recall,
and F1-score against actual categories.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report


# Alert threshold categories
ALERT_CATEGORIES = {4, 5}  # "Very Poor" and "Severe"
ALERT_LABELS = {4: "Very Poor", 5: "Severe"}

AQI_LABELS = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]


def generate_alerts(predicted_categories):
    """
    Generate binary alerts from predicted AQI category codes.

    Parameters
    ----------
    predicted_categories : array-like
        Predicted AQI category codes (0-5).

    Returns
    -------
    np.ndarray
        Binary alert array: 1 = alert (Very Poor/Severe), 0 = no alert.
    """
    predicted = np.asarray(predicted_categories)
    alerts = np.isin(predicted, list(ALERT_CATEGORIES)).astype(int)

    n_alerts = alerts.sum()
    print(f"[Early Warning] Generated {n_alerts} alerts "
          f"out of {len(alerts)} predictions "
          f"({100 * n_alerts / len(alerts):.1f}%)")

    return alerts


def evaluate_early_warning(y_true_categories, y_pred_categories):
    """
    Evaluate the early warning module by comparing predicted alerts
    against actual alerts (derived from true AQI categories).

    Parameters
    ----------
    y_true_categories : array-like
        True AQI category codes.
    y_pred_categories : array-like
        Predicted AQI category codes.

    Returns
    -------
    dict
        Evaluation metrics for the alert system.
    """
    # Generate binary alerts for both true and predicted
    true_alerts = np.isin(np.asarray(y_true_categories), list(ALERT_CATEGORIES)).astype(int)
    pred_alerts = generate_alerts(y_pred_categories)

    # Compute alert-level metrics
    metrics = {
        "alert_precision": precision_score(true_alerts, pred_alerts, zero_division=0),
        "alert_recall": recall_score(true_alerts, pred_alerts, zero_division=0),
        "alert_f1_score": f1_score(true_alerts, pred_alerts, zero_division=0),
        "total_true_alerts": int(true_alerts.sum()),
        "total_pred_alerts": int(pred_alerts.sum()),
        "total_samples": len(true_alerts),
    }

    # True positives, false positives, false negatives
    tp = int(np.sum((pred_alerts == 1) & (true_alerts == 1)))
    fp = int(np.sum((pred_alerts == 1) & (true_alerts == 0)))
    fn = int(np.sum((pred_alerts == 0) & (true_alerts == 1)))
    tn = int(np.sum((pred_alerts == 0) & (true_alerts == 0)))

    metrics["true_positives"] = tp
    metrics["false_positives"] = fp
    metrics["false_negatives"] = fn
    metrics["true_negatives"] = tn

    print(f"\n{'=' * 60}")
    print(f"  EARLY WARNING SYSTEM — Evaluation")
    print(f"{'=' * 60}")
    print(f"  Total Predictions:   {metrics['total_samples']}")
    print(f"  True High-Risk:      {metrics['total_true_alerts']}")
    print(f"  Predicted High-Risk: {metrics['total_pred_alerts']}")
    print(f"")
    print(f"  True Positives:  {tp}  (Correctly detected high-risk)")
    print(f"  False Positives: {fp}  (False alarms)")
    print(f"  False Negatives: {fn}  (Missed high-risk — DANGEROUS)")
    print(f"  True Negatives:  {tn}  (Correctly cleared)")
    print(f"")
    print(f"  Alert Precision: {metrics['alert_precision']:.4f}")
    print(f"  Alert Recall:    {metrics['alert_recall']:.4f}")
    print(f"  Alert F1-Score:  {metrics['alert_f1_score']:.4f}")
    print(f"{'=' * 60}")

    return metrics


def format_alert_message(aqi_category_code, aqi_value=None, timestamp=None):
    """
    Generate a human-readable alert message.

    Parameters
    ----------
    aqi_category_code : int
        Predicted AQI category code.
    aqi_value : float, optional
        Predicted AQI numeric value.
    timestamp : str, optional
        Timestamp for the prediction.

    Returns
    -------
    dict or None
        Alert message dict, or None if no alert.
    """
    if aqi_category_code not in ALERT_CATEGORIES:
        return None

    category_label = AQI_LABELS[aqi_category_code] if aqi_category_code < len(AQI_LABELS) else "Unknown"

    alert = {
        "level": "CRITICAL" if aqi_category_code == 5 else "WARNING",
        "category": category_label,
        "category_code": aqi_category_code,
        "aqi_value": aqi_value,
        "timestamp": timestamp,
        "message": "",
        "health_advisory": "",
    }

    if aqi_category_code == 5:  # Severe
        alert["message"] = (
            "🔴 SEVERE AIR QUALITY ALERT: AQI is in the SEVERE range. "
            "Immediate public health risk."
        )
        alert["health_advisory"] = (
            "AVOID all outdoor activities. Keep windows and doors closed. "
            "Use air purifiers indoors. Vulnerable populations (children, "
            "elderly, respiratory patients) should take extra precautions. "
            "Seek medical attention if experiencing breathing difficulty."
        )
    elif aqi_category_code == 4:  # Very Poor
        alert["message"] = (
            "🟠 VERY POOR AIR QUALITY ALERT: AQI is in the VERY POOR range. "
            "Significant health risk for sensitive groups."
        )
        alert["health_advisory"] = (
            "MINIMIZE outdoor activities, especially strenuous exercise. "
            "Wear N95 masks if going outside. Keep windows closed. "
            "Sensitive groups should avoid all outdoor exertion."
        )

    return alert
