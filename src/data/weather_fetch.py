"""
Weather Data Fetcher — Open-Meteo Historical Weather API
=========================================================
Fetches historical hourly meteorological data (temperature, humidity,
wind speed, surface pressure, dew point) from the Open-Meteo Archive API.
Completely free, no API key required.
"""

import pandas as pd
import requests
import time
import os
from datetime import datetime, timedelta


def fetch_weather_chunk(latitude, longitude, start_date, end_date, variables):
    """
    Fetch a single chunk of historical weather data from Open-Meteo.

    Parameters
    ----------
    latitude : float
        Target latitude.
    longitude : float
        Target longitude.
    start_date : str
        Start date in 'YYYY-MM-DD' format.
    end_date : str
        End date in 'YYYY-MM-DD' format.
    variables : list of str
        Weather variable names to fetch.

    Returns
    -------
    pd.DataFrame
        Hourly weather data with 'datetime' index.
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ",".join(variables),
        "start_date": start_date,
        "end_date": end_date,
        "timezone": "auto",
    }

    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()
    data = response.json()

    if "hourly" not in data:
        raise ValueError(f"No hourly data returned for {start_date} to {end_date}")

    hourly = data["hourly"]
    df = pd.DataFrame(hourly)
    df["datetime"] = pd.to_datetime(df["time"])
    df = df.drop(columns=["time"])
    df = df.set_index("datetime")

    return df


def fetch_weather_data(latitude, longitude, start_date, end_date, variables,
                       chunk_days=90, save_path=None):
    """
    Fetch historical meteorological data in chunks to respect API limits.

    Parameters
    ----------
    latitude : float
        Target latitude (e.g., 28.6139 for Delhi).
    longitude : float
        Target longitude (e.g., 77.2090 for Delhi).
    start_date : str
        Overall start date 'YYYY-MM-DD'.
    end_date : str
        Overall end date 'YYYY-MM-DD'.
    variables : list of str
        Variable names: ['temperature_2m', 'relative_humidity_2m',
                         'wind_speed_10m', 'surface_pressure', 'dew_point_2m'].
    chunk_days : int
        Number of days per API call (default: 90).
    save_path : str, optional
        If provided, save the resulting DataFrame to this CSV path.

    Returns
    -------
    pd.DataFrame
        Complete hourly weather DataFrame.
    """
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    all_chunks = []
    current_start = start

    print(f"[Weather] Fetching meteorological data for ({latitude}, {longitude})")
    print(f"[Weather] Period: {start_date} to {end_date}")

    chunk_num = 0
    while current_start < end:
        current_end = min(current_start + timedelta(days=chunk_days - 1), end)

        chunk_start_str = current_start.strftime("%Y-%m-%d")
        chunk_end_str = current_end.strftime("%Y-%m-%d")

        chunk_num += 1
        print(f"  Chunk {chunk_num}: {chunk_start_str} → {chunk_end_str} ... ", end="")

        try:
            chunk_df = fetch_weather_chunk(
                latitude, longitude,
                chunk_start_str, chunk_end_str,
                variables
            )
            all_chunks.append(chunk_df)
            print(f"✓ ({len(chunk_df)} rows)")
        except Exception as e:
            print(f"✗ Error: {e}")

        current_start = current_end + timedelta(days=1)
        time.sleep(0.5)  # Respectful rate limiting

    if not all_chunks:
        raise RuntimeError("No weather data was fetched successfully.")

    df = pd.concat(all_chunks, axis=0)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]

    # Rename columns to shorter standardized names
    rename_map = {
        "temperature_2m": "temperature",
        "relative_humidity_2m": "humidity",
        "wind_speed_10m": "wind_speed",
        "surface_pressure": "pressure",
        "dew_point_2m": "dew_point",
    }
    df = df.rename(columns=rename_map)

    print(f"\n[Weather] Total rows fetched: {len(df)}")
    print(f"[Weather] Date range: {df.index.min()} to {df.index.max()}")
    print(f"[Weather] Columns: {list(df.columns)}")
    print(f"[Weather] Missing values:\n{df.isnull().sum()}")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path)
        print(f"[Weather] Saved to: {save_path}")

    return df
