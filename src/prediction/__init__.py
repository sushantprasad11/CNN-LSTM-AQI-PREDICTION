"""Prediction subpackage."""

from src.prediction.early_warning import (
    generate_alerts, evaluate_early_warning, format_alert_message
)
from src.prediction.health_risk import (
    map_health_risk, add_health_risk_columns, get_risk_summary
)

# predict.py and ensemble.py require TensorFlow — import conditionally
try:
    from src.prediction.predict import load_trained_model, predict_aqi
    from src.prediction.ensemble import EnsemblePredictor
except ImportError:
    pass  # TensorFlow not installed; predict functions unavailable

from src.prediction.latest_result import (
    save_latest_prediction, load_latest_prediction
)

__all__ = [
    "generate_alerts", "evaluate_early_warning", "format_alert_message",
    "map_health_risk", "add_health_risk_columns", "get_risk_summary",
    "load_trained_model", "predict_aqi",
    "EnsemblePredictor",
    "save_latest_prediction", "load_latest_prediction",
]
