"""
AQI Calculator — Indian CPCB Standard
=======================================
Computes the Air Quality Index using the Indian Central Pollution Control Board
(CPCB) National Ambient Air Quality Standards. Each pollutant gets a sub-index
and the overall AQI is the maximum sub-index value.

Reference: CPCB National Air Quality Index (NAQI) guidelines.
"""

import pandas as pd
import numpy as np


# ── Indian CPCB AQI Breakpoint Tables ──────────────────────────────────
# Format: list of (C_low, C_high, I_low, I_high)
# Concentrations: PM2.5/PM10 in µg/m³ (24-hr avg), SO₂/NO₂/O₃ in µg/m³, CO in mg/m³

BREAKPOINTS = {
    "pm2_5": [
        (0, 30, 0, 50),
        (31, 60, 51, 100),
        (61, 90, 101, 200),
        (91, 120, 201, 300),
        (121, 250, 301, 400),
        (251, 500, 401, 500),
    ],
    "pm10": [
        (0, 50, 0, 50),
        (51, 100, 51, 100),
        (101, 250, 101, 200),
        (251, 350, 201, 300),
        (351, 430, 301, 400),
        (431, 600, 401, 500),
    ],
    "so2": [
        (0, 40, 0, 50),
        (41, 80, 51, 100),
        (81, 380, 101, 200),
        (381, 800, 201, 300),
        (801, 1600, 301, 400),
        (1601, 2400, 401, 500),
    ],
    "no2": [
        (0, 40, 0, 50),
        (41, 80, 51, 100),
        (81, 180, 101, 200),
        (181, 280, 201, 300),
        (281, 400, 301, 400),
        (401, 600, 401, 500),
    ],
    "co": [
        (0, 1, 0, 50),
        (1.1, 2, 51, 100),
        (2.1, 10, 101, 200),
        (10.1, 17, 201, 300),
        (17.1, 34, 301, 400),
        (34.1, 50, 401, 500),
    ],
    "o3": [
        (0, 50, 0, 50),
        (51, 100, 51, 100),
        (101, 168, 101, 200),
        (169, 208, 201, 300),
        (209, 748, 301, 400),
        (749, 1000, 401, 500),
    ],
}

# AQI category definitions
AQI_CATEGORIES = [
    {"label": "Good",         "range": (0, 50),    "code": 0},
    {"label": "Satisfactory", "range": (51, 100),   "code": 1},
    {"label": "Moderate",     "range": (101, 200),  "code": 2},
    {"label": "Poor",         "range": (201, 300),  "code": 3},
    {"label": "Very Poor",    "range": (301, 400),  "code": 4},
    {"label": "Severe",       "range": (401, 500),  "code": 5},
]


def compute_sub_index(concentration, breakpoints):
    """
    Compute the sub-index for a single pollutant concentration.

    Uses linear interpolation within the matching breakpoint range:
        I = ((I_high - I_low) / (C_high - C_low)) * (C - C_low) + I_low

    Parameters
    ----------
    concentration : float
        Pollutant concentration value.
    breakpoints : list of tuples
        Breakpoint table [(C_low, C_high, I_low, I_high), ...].

    Returns
    -------
    float
        Sub-index value, or NaN if concentration is out of range.
    """
    if pd.isna(concentration) or concentration < 0:
        return np.nan

    for c_low, c_high, i_low, i_high in breakpoints:
        if c_low <= concentration <= c_high:
            sub_index = ((i_high - i_low) / (c_high - c_low)) * (concentration - c_low) + i_low
            return round(sub_index, 2)

    # If concentration exceeds the highest breakpoint, cap at 500
    if concentration > breakpoints[-1][1]:
        return 500.0

    return np.nan


def compute_aqi(row, pollutant_columns=None):
    """
    Compute the overall AQI for a single row (timestamp).

    AQI = max of all individual sub-indices (CPCB standard).

    Parameters
    ----------
    row : pd.Series
        A single row containing pollutant concentrations.
    pollutant_columns : list of str, optional
        Column names for pollutants. If None, uses all keys in BREAKPOINTS.

    Returns
    -------
    float
        Overall AQI value.
    """
    if pollutant_columns is None:
        pollutant_columns = list(BREAKPOINTS.keys())

    sub_indices = []
    for pollutant in pollutant_columns:
        if pollutant in row.index and pollutant in BREAKPOINTS:
            conc = row[pollutant]
            # CO from Open-Meteo is in µg/m³; CPCB table uses mg/m³
            if pollutant == "co":
                conc = conc / 1000.0  # Convert µg/m³ → mg/m³
            si = compute_sub_index(conc, BREAKPOINTS[pollutant])
            if not np.isnan(si):
                sub_indices.append(si)

    if sub_indices:
        return max(sub_indices)
    return np.nan


def get_aqi_category(aqi_value):
    """
    Map an AQI value to its category label and numeric code.

    Parameters
    ----------
    aqi_value : float
        AQI value.

    Returns
    -------
    tuple of (str, int)
        (category_label, category_code)
    """
    if pd.isna(aqi_value):
        return "Unknown", -1

    for cat in AQI_CATEGORIES:
        low, high = cat["range"]
        if low <= aqi_value <= high:
            return cat["label"], cat["code"]

    # If AQI > 500, classify as Severe
    if aqi_value > 500:
        return "Severe", 5

    return "Unknown", -1


def calculate_aqi_dataframe(df, save_path=None):
    """
    Calculate AQI and category for an entire DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with pollutant columns (pm2_5, pm10, no2, so2, co, o3).
    save_path : str, optional
        If provided, save the resulting DataFrame to this CSV path.

    Returns
    -------
    pd.DataFrame
        Original DataFrame with added columns: 'aqi', 'aqi_category', 'aqi_code'.
    """
    import os

    pollutant_cols = [c for c in BREAKPOINTS.keys() if c in df.columns]
    print(f"[AQI] Computing AQI using pollutants: {pollutant_cols}")

    df_result = df.copy()

    # Compute AQI for each row
    df_result["aqi"] = df_result.apply(
        lambda row: compute_aqi(row, pollutant_cols), axis=1
    )

    # Map to categories
    categories = df_result["aqi"].apply(get_aqi_category)
    df_result["aqi_category"] = categories.apply(lambda x: x[0])
    df_result["aqi_code"] = categories.apply(lambda x: x[1])

    # Drop rows where AQI couldn't be computed
    n_before = len(df_result)
    df_result = df_result.dropna(subset=["aqi"])
    n_dropped = n_before - len(df_result)
    if n_dropped > 0:
        print(f"[AQI] Dropped {n_dropped} rows with invalid AQI values")

    # Print distribution
    print(f"\n[AQI] Category Distribution:")
    print(df_result["aqi_category"].value_counts().to_string())
    print(f"\n[AQI] AQI Statistics:")
    print(f"  Min:  {df_result['aqi'].min():.2f}")
    print(f"  Max:  {df_result['aqi'].max():.2f}")
    print(f"  Mean: {df_result['aqi'].mean():.2f}")
    print(f"  Std:  {df_result['aqi'].std():.2f}")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df_result.to_csv(save_path)
        print(f"[AQI] Saved to: {save_path}")

    return df_result
