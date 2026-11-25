"""
Latest Prediction Store & Syncer
==================================
Manages saving and loading the latest prediction result for real-time
display on the web dashboard.
"""

import os
import json
from datetime import datetime
import numpy as np


LATEST_PREDICTION_PATH = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
    "results",
    "latest_prediction.json"
)


def _make_serializable(obj):
    """Convert numpy types to native Python for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def save_latest_prediction(data, pred_type, command_str=None):
    """
    Save the latest prediction output to results/latest_prediction.json
    so that the dashboard can dynamically pick it up in real-time.

    Parameters
    ----------
    data : dict
        Prediction result payload (from predict_for_datetime).
    pred_type : str
        'single_datetime' or 'range_prediction'.
    command_str : str, optional
        Command line executed by user.
    """
    os.makedirs(os.path.dirname(LATEST_PREDICTION_PATH), exist_ok=True)

    payload = {
        "type": pred_type,
        "command": command_str or "python prediction script",
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "timestamp_iso": datetime.now().isoformat(),
        "data": _make_serializable(data),
    }

    # Extract quick summary fields for easy front-end display
    if pred_type == "single_datetime":
        cat = data.get("predicted_category", "Unknown")
        code = data.get("predicted_code", 0)
        conf = data.get("confidence", 0.0)
        dt = data.get("datetime", "")
        mode = data.get("mode", "PREDICTION")
        risk = data.get("risk_info", {})

        payload["summary"] = {
            "title": f"AQI for {dt} ({mode})",
            "dominant_category": cat,
            "dominant_code": code,
            "avg_confidence": round(conf, 1),
            "target_datetime": dt,
            "mode": mode,
            "risk_level": risk.get("risk_level", "Unknown"),
            "advisory": risk.get("advisory", ""),
        }

    elif pred_type == "range_prediction":
        predictions = data if isinstance(data, list) else data.get("predictions", [])
        payload["summary"] = {
            "title": f"Range Forecast ({len(predictions)} Timestamps)",
            "dominant_category": predictions[0].get("predicted_category", "Unknown") if predictions else "Unknown",
            "dominant_code": predictions[0].get("predicted_code", 0) if predictions else 0,
            "avg_confidence": round(np.mean([p.get("confidence", 0) for p in predictions]), 1) if predictions else 0.0,
            "total_points": len(predictions),
        }

    with open(LATEST_PREDICTION_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    return LATEST_PREDICTION_PATH


def load_latest_prediction():
    """
    Load the latest prediction payload from results/latest_prediction.json.

    Returns
    -------
    dict or None
    """
    if os.path.exists(LATEST_PREDICTION_PATH):
        try:
            with open(LATEST_PREDICTION_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                data["last_modified"] = os.path.getmtime(LATEST_PREDICTION_PATH)
                return data
        except Exception:
            pass

    return None
