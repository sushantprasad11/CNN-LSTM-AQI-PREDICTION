"""
Feature Engineering Module
===========================
Constructs engineered features for AQI prediction as described in the synopsis:
1. Lagged features — past pollutant values at specified hour offsets
2. Rolling window averages — multi-scale moving averages
3. Cyclic temporal encodings — sin/cos for hour-of-day and day-of-year
4. Pollutant ratio features — ratios between selected pollutant pairs

These feature groups are designed for the step-wise ablation study.
"""

import pandas as pd
import numpy as np
import os


def create_lag_features(df, columns, lag_hours):
    """
    Create lagged (shifted) features for specified columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with DatetimeIndex.
    columns : list of str
        Columns to create lag features for.
    lag_hours : list of int
        Lag offsets in hours (e.g., [1, 2, 3, 6, 12, 24]).

    Returns
    -------
    pd.DataFrame
        DataFrame with original + lag columns added.
    list of str
        Names of the new lag feature columns.
    """
    df_out = df.copy()
    new_cols = []

    for col in columns:
        if col not in df_out.columns:
            continue
        for lag in lag_hours:
            new_col_name = f"{col}_lag_{lag}h"
            df_out[new_col_name] = df_out[col].shift(lag)
            new_cols.append(new_col_name)

    print(f"[Features] Created {len(new_cols)} lag features")
    return df_out, new_cols


def create_rolling_features(df, columns, windows):
    """
    Create rolling mean features for specified columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    columns : list of str
        Columns to create rolling averages for.
    windows : list of int
        Window sizes in hours (e.g., [3, 6, 12, 24]).

    Returns
    -------
    pd.DataFrame
        DataFrame with rolling average columns added.
    list of str
        Names of the new rolling feature columns.
    """
    df_out = df.copy()
    new_cols = []

    for col in columns:
        if col not in df_out.columns:
            continue
        for window in windows:
            new_col_name = f"{col}_roll_{window}h"
            df_out[new_col_name] = df_out[col].rolling(window=window, min_periods=1).mean()
            new_cols.append(new_col_name)

    print(f"[Features] Created {len(new_cols)} rolling average features")
    return df_out, new_cols


def create_cyclic_features(df):
    """
    Create cyclic (sin/cos) temporal features for hour-of-day and day-of-year.

    These encode the cyclical nature of time: hour 23 is close to hour 0,
    and December 31 is close to January 1.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with DatetimeIndex.

    Returns
    -------
    pd.DataFrame
        DataFrame with cyclic feature columns added.
    list of str
        Names of the new cyclic feature columns.
    """
    df_out = df.copy()

    # Ensure DatetimeIndex
    if not isinstance(df_out.index, pd.DatetimeIndex):
        df_out.index = pd.to_datetime(df_out.index)

    # Hour-of-day cyclic encoding (period = 24)
    hour = df_out.index.hour
    df_out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df_out["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    # Day-of-year cyclic encoding (period = 365.25)
    day_of_year = df_out.index.dayofyear
    df_out["day_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    df_out["day_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)

    # Day-of-week cyclic encoding (period = 7)
    day_of_week = df_out.index.dayofweek
    df_out["dow_sin"] = np.sin(2 * np.pi * day_of_week / 7)
    df_out["dow_cos"] = np.cos(2 * np.pi * day_of_week / 7)

    # Month cyclic encoding (period = 12)
    month = df_out.index.month
    df_out["month_sin"] = np.sin(2 * np.pi * month / 12)
    df_out["month_cos"] = np.cos(2 * np.pi * month / 12)

    new_cols = ["hour_sin", "hour_cos", "day_sin", "day_cos",
                "dow_sin", "dow_cos", "month_sin", "month_cos"]

    print(f"[Features] Created {len(new_cols)} cyclic temporal features")
    return df_out, new_cols


def create_ratio_features(df, ratio_pairs):
    """
    Create ratio features between specified pollutant pairs.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    ratio_pairs : list of [str, str]
        Pairs of column names to compute ratios for.
        e.g., [["pm2_5", "pm10"], ["no2", "so2"], ["co", "o3"]]

    Returns
    -------
    pd.DataFrame
        DataFrame with ratio columns added.
    list of str
        Names of the new ratio feature columns.
    """
    df_out = df.copy()
    new_cols = []

    for pair in ratio_pairs:
        col_a, col_b = pair
        if col_a in df_out.columns and col_b in df_out.columns:
            new_col_name = f"ratio_{col_a}_{col_b}"
            # Add small epsilon to avoid division by zero
            df_out[new_col_name] = df_out[col_a] / (df_out[col_b] + 1e-8)
            new_cols.append(new_col_name)

    print(f"[Features] Created {len(new_cols)} ratio features")
    return df_out, new_cols


def engineer_all_features(df, config, save_path=None):
    """
    Apply all feature engineering steps to the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with pollutant + weather + AQI columns.
    config : dict
        Configuration dictionary with feature engineering parameters.
    save_path : str, optional
        If provided, save the result to this CSV path.

    Returns
    -------
    pd.DataFrame
        Fully featured DataFrame.
    dict
        Dictionary mapping feature group names to their column lists,
        used for the ablation study.
    """
    pollutant_cols = ["pm2_5", "pm10", "no2", "so2", "co", "o3"]
    pollutant_cols = [c for c in pollutant_cols if c in df.columns]

    weather_cols = ["temperature", "humidity", "wind_speed", "pressure", "dew_point"]
    weather_cols = [c for c in weather_cols if c in df.columns]

    feature_groups = {}

    # Track base pollutant columns
    feature_groups["base_pollutants"] = pollutant_cols.copy()

    # 1. Lag features
    lag_hours = config["features"]["lag_hours"]
    df, lag_cols = create_lag_features(df, pollutant_cols, lag_hours)
    feature_groups["lag_features"] = lag_cols

    # 2. Rolling average features
    rolling_windows = config["features"]["rolling_windows"]
    df, roll_cols = create_rolling_features(df, pollutant_cols, rolling_windows)
    feature_groups["rolling_features"] = roll_cols

    # 3. Cyclic temporal features
    df, cyclic_cols = create_cyclic_features(df)
    feature_groups["cyclic_features"] = cyclic_cols

    # 4. Ratio features
    ratio_pairs = config["features"]["ratio_pairs"]
    df, ratio_cols = create_ratio_features(df, ratio_pairs)
    feature_groups["ratio_features"] = ratio_cols

    # 5. Meteorological features (Extension 1)
    feature_groups["meteorological"] = weather_cols

    # Drop rows with NaN created by lag/rolling operations
    n_before = len(df)
    df = df.dropna()
    n_dropped = n_before - len(df)
    print(f"[Features] Dropped {n_dropped} rows with NaN from lag/rolling operations")
    print(f"[Features] Final shape: {df.shape}")

    # Summary
    total_features = sum(len(v) for v in feature_groups.values())
    print(f"\n[Features] Feature Group Summary:")
    for group_name, cols in feature_groups.items():
        print(f"  {group_name}: {len(cols)} features")
    print(f"  Total engineered features: {total_features}")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path)
        print(f"[Features] Saved to: {save_path}")

    return df, feature_groups
