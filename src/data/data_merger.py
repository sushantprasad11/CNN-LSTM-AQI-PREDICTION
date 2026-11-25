"""
Data Merger Module
===================
Merges pollution and weather DataFrames on their common hourly timestamps.
"""

import pandas as pd
import os


def merge_pollution_weather(pollution_df, weather_df, save_path=None):
    """
    Merge pollution and weather DataFrames on datetime index.

    Both DataFrames are expected to have a DatetimeIndex. An inner join
    is used to keep only timestamps present in both datasets.

    Parameters
    ----------
    pollution_df : pd.DataFrame
        Pollution data with DatetimeIndex.
    weather_df : pd.DataFrame
        Weather data with DatetimeIndex.
    save_path : str, optional
        If provided, save the merged DataFrame to this CSV path.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame with both pollution and weather columns.
    """
    print("[Merge] Merging pollution and weather data on timestamp...")
    print(f"  Pollution shape: {pollution_df.shape}")
    print(f"  Weather shape:   {weather_df.shape}")

    # Ensure both have DatetimeIndex
    if not isinstance(pollution_df.index, pd.DatetimeIndex):
        pollution_df.index = pd.to_datetime(pollution_df.index)
    if not isinstance(weather_df.index, pd.DatetimeIndex):
        weather_df.index = pd.to_datetime(weather_df.index)

    # Inner join on index
    merged = pollution_df.join(weather_df, how="inner")

    # Remove any duplicate timestamps
    merged = merged[~merged.index.duplicated(keep="first")]

    # Sort by time
    merged = merged.sort_index()

    print(f"  Merged shape:    {merged.shape}")
    print(f"  Date range:      {merged.index.min()} to {merged.index.max()}")
    print(f"  Columns:         {list(merged.columns)}")

    # Check for any overlap in column names
    overlap = set(pollution_df.columns) & set(weather_df.columns)
    if overlap:
        print(f"  ⚠ Overlapping columns: {overlap}")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        merged.to_csv(save_path)
        print(f"  Saved to: {save_path}")

    return merged
