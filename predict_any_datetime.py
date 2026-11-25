"""
Predict AQI Category for Any Date and Time — Past OR Future
=============================================================
Supports:
  - Past dates   → uses Open-Meteo historical archive APIs
  - Future dates → uses Open-Meteo forecast APIs (up to ~5-7 days ahead)
  - Mixed ranges → automatically combines both APIs

Usage:
    python predict_any_datetime.py "2025-06-15 14:00"       # Past date
    python predict_any_datetime.py "2026-10-28 10:00"       # Future date
    python predict_any_datetime.py                          # Default example

Requirements:
    - Trained model at: models/cnn_lstm/best_model.keras
    - Fitted scaler at: artifacts/scaler.joblib
    - Internet connection (for Open-Meteo API calls)
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import warnings
warnings.filterwarnings('ignore')

# UTF-8 for Windows console
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.data.forecast_fetch import fetch_pollution_smart, fetch_weather_smart
from src.data.data_merger import merge_pollution_weather
from src.data.data_cleaning import handle_missing_values
from src.preprocessing.aqi_calculator import calculate_aqi_dataframe
from src.preprocessing.feature_engineering import engineer_all_features
from src.preprocessing.scaler import AQIScaler
from src.prediction.predict import load_trained_model, AQI_LABELS
from src.prediction.health_risk import map_health_risk
from src.prediction.latest_result import save_latest_prediction
from src.data.data_loader import load_config


def predict_for_datetime(target_datetime_str, model_path="models/cnn_lstm/best_model.keras"):
    """
    Predict AQI category for a specific date and time (past or future).

    For past dates, fetches historical data from Open-Meteo archive APIs.
    For future dates, fetches forecast data from Open-Meteo forecast APIs.
    For near-present dates, seamlessly combines both.

    Parameters
    ----------
    target_datetime_str : str
        Target datetime in "YYYY-MM-DD HH:MM" format.
    model_path : str
        Path to the trained CNN-LSTM model.

    Returns
    -------
    dict
        Prediction results including category, confidence, probabilities,
        and health risk information.
    """
    # Normalize target datetime string (handle 'T', seconds, or date-only)
    clean_dt_str = str(target_datetime_str).strip().replace("T", " ")
    if len(clean_dt_str) == 10:  # YYYY-MM-DD
        clean_dt_str += " 12:00"
    elif len(clean_dt_str) > 16:  # YYYY-MM-DD HH:MM:SS
        clean_dt_str = clean_dt_str[:16]

    target_dt = datetime.strptime(clean_dt_str, "%Y-%m-%d %H:%M")
    config = load_config("config/config.yaml")
    now = datetime.utcnow()

    # Determine if this is a past or future prediction
    is_future = target_dt > now
    mode = "FORECAST" if is_future else "HISTORICAL"

    # We need ~96 hours before the target to compute all features
    # (24h lag features + 24h rolling windows + 24h model input window)
    fetch_start = (target_dt - timedelta(hours=96)).strftime("%Y-%m-%d")
    fetch_end = target_dt.strftime("%Y-%m-%d")

    lat = config["data"]["latitude"]   # 28.6139 (Delhi)
    lon = config["data"]["longitude"]  # 77.2090

    # Check forecast limit
    if is_future:
        days_ahead = (target_dt - now).days
        if days_ahead > 7:
            print(f"\n⚠  WARNING: Target is {days_ahead} days in the future.")
            print(f"   Open-Meteo provides air quality forecasts up to ~5 days")
            print(f"   and weather forecasts up to ~16 days.")
            print(f"   Predictions beyond this range may have limited/no data.\n")

    print(f"\n{'='*60}")
    print(f"  AQI PREDICTION [{mode}]")
    print(f"  Target:   {target_datetime_str}")
    print(f"  Location: Delhi ({lat}, {lon})")
    print(f"  Data:     {fetch_start} to {fetch_end}")
    print(f"{'='*60}\n")

    # ── Step 1: Fetch data using smart fetchers (with local fallback) ──
    try:
        print("[1/7] Fetching pollution data (smart: historical + forecast)...")
        pollution_df = fetch_pollution_smart(
            lat, lon, fetch_start, fetch_end,
            pollutants=config["data"]["pollutants"]
        )

        print("\n[2/7] Fetching weather data (smart: archive + forecast)...")
        weather_df = fetch_weather_smart(
            lat, lon, fetch_start, fetch_end,
            variables=config["data"]["weather_variables"]
        )

        # ── Step 2: Merge ──
        print("\n[3/7] Merging datasets...")
        merged_df = merge_pollution_weather(pollution_df, weather_df)

        if len(merged_df) < 48:
            raise ValueError(
                f"Insufficient data after merge: {len(merged_df)} rows. "
                f"Need at least 48 hourly observations."
            )

        # ── Step 3: Clean ──
        print("\n[4/7] Cleaning data...")
        merged_df = handle_missing_values(merged_df, method='interpolate', limit=6)
        merged_df = merged_df.ffill().bfill()

        # ── Step 4: Compute AQI ──
        print("\n[5/7] Computing AQI...")
        aqi_df = calculate_aqi_dataframe(merged_df)

        # ── Step 5: Feature Engineering ──
        print("\n[6/7] Engineering features...")
        featured_df, feature_groups = engineer_all_features(aqi_df, config)

    except Exception as fetch_err:
        local_path = "data/processed/featured_data.csv"
        if os.path.exists(local_path):
            print(f"\n  [Notice] Live API unavailable ({fetch_err}).")
            print(f"           Loading historical data from local dataset ({local_path})...")
            featured_df = pd.read_csv(local_path, parse_dates=['datetime'], index_col='datetime')
        else:
            raise fetch_err

    # Get feature columns (must match training order)
    exclude_cols = ['aqi', 'aqi_category', 'aqi_code']
    feature_cols = [c for c in featured_df.columns if c not in exclude_cols]

    # ── Step 6: Scale with saved scaler ──
    print("\n[7/7] Scaling and preparing input...")
    scaler = AQIScaler()
    scaler.load("artifacts/scaler.joblib")

    # Align feature columns with what the scaler expects
    scaler_cols = scaler.feature_columns
    missing_cols = [c for c in scaler_cols if c not in featured_df.columns]
    if missing_cols:
        print(f"  Info: Adding {len(missing_cols)} columns expected by scaler (set to 0)")
        for col in missing_cols:
            featured_df[col] = 0.0

    extra_cols = [c for c in feature_cols if c not in scaler_cols]
    if extra_cols:
        print(f"  Info: {len(extra_cols)} extra columns not in scaler (will be ignored)")

    # Use scaler's feature column order for consistency
    feature_cols = scaler_cols
    scaled_df = scaler.transform(featured_df)

    # ── Step 7: Extract 24-hour window and predict ──
    target_ts = pd.Timestamp(target_dt)

    if target_ts in scaled_df.index:
        target_idx = scaled_df.index.get_loc(target_ts)
        actual_ts = target_ts
    else:
        # Find nearest timestamp
        nearest_idx = scaled_df.index.get_indexer([target_ts], method='nearest')[0]
        if nearest_idx < 0:
            # Fallback: use the last available timestamp
            nearest_idx = len(scaled_df) - 1
            print(f"  Target timestamp not in data; using latest: {scaled_df.index[nearest_idx]}")
        target_idx = nearest_idx
        actual_ts = scaled_df.index[target_idx]
        print(f"  Using nearest available timestamp: {actual_ts}")

    if target_idx < 24:
        raise ValueError(
            f"Not enough preceding data for a 24-hour window. "
            f"Need at least 24 hourly datapoints before target, have {target_idx}."
        )

    # Extract the 24-hour window ending at the target
    window_data = scaled_df[feature_cols].iloc[target_idx - 24:target_idx].values
    X_input = np.array([window_data], dtype=np.float32)  # Shape: (1, 24, n_features)

    print(f"  Input shape: {X_input.shape}  (1 sample, 24 timesteps, {X_input.shape[2]} features)")

    # Load model and predict
    model = load_trained_model(model_path)
    probabilities = model.predict(X_input, verbose=0)
    predicted_code = int(np.argmax(probabilities, axis=1)[0])
    confidence = float(probabilities[0][predicted_code]) * 100

    # Get health risk information
    predicted_label = AQI_LABELS[predicted_code]
    risk_info = map_health_risk(predicted_code)

    # ── Display Results ──
    print(f"\n{'='*60}")
    print(f"  🎯  PREDICTION RESULT  [{mode}]")
    print(f"{'='*60}")
    print(f"  Date/Time:    {actual_ts}")
    print(f"  AQI Category: {predicted_label} (Code {predicted_code})")
    print(f"  Confidence:   {confidence:.1f}%")
    print(f"  Risk Level:   {risk_info['risk_level']}")
    print(f"  Advisory:     {risk_info['advisory']}")
    if is_future:
        print(f"  ⚠ Note:       Based on forecast data — accuracy may vary")
    print(f"{'='*60}")

    # Show all class probabilities
    print(f"\n  Class Probabilities:")
    for i, label in enumerate(AQI_LABELS):
        prob = probabilities[0][i] * 100
        bar = "█" * int(prob / 2.5)
        marker = " ◄" if i == predicted_code else ""
        print(f"    {label:15s} {prob:5.1f}% {bar}{marker}")

    res = {
        "datetime": str(actual_ts),
        "mode": mode,
        "predicted_category": predicted_label,
        "predicted_code": predicted_code,
        "confidence": confidence,
        "probabilities": {AQI_LABELS[i]: float(probabilities[0][i]) for i in range(6)},
        "probabilities_percent": {AQI_LABELS[i]: round(float(probabilities[0][i]) * 100, 2) for i in range(6)},
        "risk_info": risk_info,
    }

    try:
        save_latest_prediction(
            res,
            pred_type="single_datetime",
            command_str=f'python predict_any_datetime.py "{target_datetime_str}"'
        )
        print(f"\n  ⚡ Dynamic Dashboard Sync: Result saved to results/latest_prediction.json")
        print(f"     Open http://127.0.0.1:5000 to view it live on the dashboard!")
    except Exception as e:
        print(f"\n  Warning: Could not sync to dashboard: {e}")

    return res


def predict_range(start_datetime_str, end_datetime_str, interval_hours=1,
                  model_path="models/cnn_lstm/best_model.keras"):
    """
    Predict AQI for a range of dates/times (past or future).

    Parameters
    ----------
    start_datetime_str : str
        Start datetime "YYYY-MM-DD HH:MM".
    end_datetime_str : str
        End datetime "YYYY-MM-DD HH:MM".
    interval_hours : int
        Hours between predictions.
    model_path : str
        Path to the trained model.

    Returns
    -------
    list of dict
        List of prediction results for each timestamp.
    """
    start_dt = datetime.strptime(start_datetime_str, "%Y-%m-%d %H:%M")
    end_dt = datetime.strptime(end_datetime_str, "%Y-%m-%d %H:%M")
    config = load_config("config/config.yaml")
    now = datetime.utcnow()

    lat = config["data"]["latitude"]
    lon = config["data"]["longitude"]

    # Fetch all needed data in one go (with buffer)
    fetch_start = (start_dt - timedelta(hours=96)).strftime("%Y-%m-%d")
    fetch_end = end_dt.strftime("%Y-%m-%d")

    print(f"\n{'='*60}")
    print(f"  BATCH AQI PREDICTION")
    print(f"  Range: {start_datetime_str} to {end_datetime_str}")
    print(f"  Interval: every {interval_hours} hour(s)")
    print(f"{'='*60}\n")

    # Fetch all data once
    print("[1/5] Fetching pollution data...")
    pollution_df = fetch_pollution_smart(
        lat, lon, fetch_start, fetch_end,
        pollutants=config["data"]["pollutants"]
    )

    print("\n[2/5] Fetching weather data...")
    weather_df = fetch_weather_smart(
        lat, lon, fetch_start, fetch_end,
        variables=config["data"]["weather_variables"]
    )

    # Process all data once
    print("\n[3/5] Preprocessing...")
    merged_df = merge_pollution_weather(pollution_df, weather_df)
    merged_df = handle_missing_values(merged_df, method='interpolate', limit=6)
    merged_df = merged_df.ffill().bfill()
    aqi_df = calculate_aqi_dataframe(merged_df)

    print("\n[4/5] Feature engineering...")
    featured_df, feature_groups = engineer_all_features(aqi_df, config)

    scaler = AQIScaler()
    scaler.load("artifacts/scaler.joblib")
    feature_cols = scaler.feature_columns

    # Add any missing columns
    for col in feature_cols:
        if col not in featured_df.columns:
            featured_df[col] = 0.0

    scaled_df = scaler.transform(featured_df)

    # Load model
    print("\n[5/5] Running predictions...")
    model = load_trained_model(model_path)

    # Generate predictions for each timestamp
    results = []
    current = start_dt
    while current <= end_dt:
        target_ts = pd.Timestamp(current)

        if target_ts in scaled_df.index:
            target_idx = scaled_df.index.get_loc(target_ts)
        else:
            nearest_idx = scaled_df.index.get_indexer([target_ts], method='nearest')[0]
            if nearest_idx < 0:
                current += timedelta(hours=interval_hours)
                continue
            target_idx = nearest_idx

        if target_idx >= 24:
            window_data = scaled_df[feature_cols].iloc[target_idx - 24:target_idx].values
            X_input = np.array([window_data], dtype=np.float32)
            probabilities = model.predict(X_input, verbose=0)
            predicted_code = int(np.argmax(probabilities, axis=1)[0])
            confidence = float(probabilities[0][predicted_code]) * 100

            results.append({
                "datetime": str(scaled_df.index[target_idx]),
                "predicted_category": AQI_LABELS[predicted_code],
                "predicted_code": predicted_code,
                "confidence": confidence,
            })

        current += timedelta(hours=interval_hours)

    # Print summary table
    print(f"\n{'='*70}")
    print(f"  {'DateTime':22s} {'Category':15s} {'Code':6s} {'Confidence':10s}")
    print(f"  {'-'*22} {'-'*15} {'-'*6} {'-'*10}")
    for r in results:
        print(f"  {r['datetime']:22s} {r['predicted_category']:15s} "
              f"{r['predicted_code']:6d} {r['confidence']:8.1f}%")
    print(f"{'='*70}")

    try:
        save_latest_prediction(
            results,
            pred_type="range_prediction",
            command_str=f'python predict_any_datetime.py --range "{start_datetime_str}" "{end_datetime_str}" {interval_hours}'
        )
        print(f"\n  ⚡ Dynamic Dashboard Sync: Result saved to results/latest_prediction.json")
        print(f"     Open http://127.0.0.1:5000 to view it live on the dashboard!")
    except Exception as e:
        print(f"\n  Warning: Could not sync to dashboard: {e}")

    return results


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--range" and len(sys.argv) >= 4:
            # Range mode: python predict_any_datetime.py --range "2025-10-25 00:00" "2025-10-27 00:00"
            interval = int(sys.argv[4]) if len(sys.argv) > 4 else 6
            predict_range(sys.argv[2], sys.argv[3], interval_hours=interval)
        else:
            predict_for_datetime(sys.argv[1])
    else:
        # Default: predict for tomorrow as an example
        tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d 12:00")
        print(f"No datetime provided.")
        print(f"Usage:")
        print(f"  python predict_any_datetime.py \"YYYY-MM-DD HH:MM\"")
        print(f"  python predict_any_datetime.py --range \"START\" \"END\" [interval_hours]")
        print(f"\nRunning default: predicting for tomorrow ({tomorrow})...\n")
        predict_for_datetime(tomorrow)
