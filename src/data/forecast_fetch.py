"""
Smart Data Fetcher — Handles Both Historical and Forecast Data
================================================================
Intelligently switches between Open-Meteo's historical archive APIs
and forecast APIs depending on whether the requested dates are in
the past or future.

API Endpoints:
  - Air Quality (past + forecast): https://air-quality-api.open-meteo.com/v1/air-quality
  - Weather Historical:            https://archive-api.open-meteo.com/v1/archive
  - Weather Forecast:              https://api.open-meteo.com/v1/forecast

The air quality API supports both historical (past_days up to 92)
and forecast (forecast_days up to 7) in a single call.

The weather API requires separate endpoints for historical vs forecast data.
"""

import pandas as pd
import requests
import time
import os
from datetime import datetime, timedelta


# ──────────────────────────────────────────────────────────────────────
#  Air Quality (Pollution) — Single API handles both past and forecast
# ──────────────────────────────────────────────────────────────────────

def fetch_pollution_smart(latitude, longitude, start_date, end_date, pollutants,
                          chunk_days=90, save_path=None):
    """
    Fetch pollution data that works for both past AND future dates.

    The Open-Meteo Air Quality API at
        https://air-quality-api.open-meteo.com/v1/air-quality
    supports both historical data (up to 92 days back with past_days)
    and forecast data (up to 7 days ahead with forecast_days).

    For convenience we use start_date/end_date which the API also accepts.
    For ranges spanning past→future, we split into a historical chunk
    and a forecast chunk, then merge.

    Parameters
    ----------
    latitude, longitude : float
        Target coordinates.
    start_date, end_date : str
        Date range in 'YYYY-MM-DD' format.  Can include future dates.
    pollutants : list of str
        Open-Meteo variable names (e.g. 'pm2_5', 'pm10', 'nitrogen_dioxide', ...).
    chunk_days : int
        Max days per API call for historical portions.
    save_path : str, optional
        If provided, save result CSV.

    Returns
    -------
    pd.DataFrame
        Hourly pollution data with DatetimeIndex.
    """
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    now = datetime.utcnow()
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)

    print(f"[Pollution-Smart] Fetching for ({latitude}, {longitude})")
    print(f"[Pollution-Smart] Requested: {start_date} to {end_date}")

    all_chunks = []

    # ── Historical portion (start_date to min(end_date, yesterday)) ──
    historical_end = min(end_dt, today - timedelta(days=1))
    if start_dt <= historical_end:
        hist_start_str = start_dt.strftime("%Y-%m-%d")
        hist_end_str = historical_end.strftime("%Y-%m-%d")
        print(f"[Pollution-Smart] Historical range: {hist_start_str} to {hist_end_str}")

        current = start_dt
        chunk_num = 0
        while current <= historical_end:
            chunk_end = min(current + timedelta(days=chunk_days - 1), historical_end)
            c_start = current.strftime("%Y-%m-%d")
            c_end = chunk_end.strftime("%Y-%m-%d")
            chunk_num += 1
            print(f"  Historical chunk {chunk_num}: {c_start} → {c_end} ... ", end="")

            try:
                df = _fetch_air_quality_chunk(latitude, longitude, c_start, c_end, pollutants)
                all_chunks.append(df)
                print(f"✓ ({len(df)} rows)")
            except Exception as e:
                print(f"✗ Error: {e}")

            current = chunk_end + timedelta(days=1)
            time.sleep(0.3)

    # ── Forecast portion (today onwards up to end_date) ──
    forecast_start = max(start_dt, today)
    if end_dt >= today:
        days_ahead = (end_dt - today).days + 1
        days_ahead = min(days_ahead, 7)  # API max: 7 days forecast
        past_days_param = max(0, min((today - start_dt).days, 92))

        print(f"[Pollution-Smart] Forecast range: {forecast_start.strftime('%Y-%m-%d')} "
              f"to {end_dt.strftime('%Y-%m-%d')} ({days_ahead} days ahead)")

        try:
            df = _fetch_air_quality_forecast(
                latitude, longitude, pollutants,
                forecast_days=days_ahead,
                past_days=min(past_days_param, 5)  # include a few past days for overlap
            )
            all_chunks.append(df)
            print(f"  Forecast: ✓ ({len(df)} rows)")
        except Exception as e:
            print(f"  Forecast: ✗ Error: {e}")

    if not all_chunks:
        raise RuntimeError("No pollution data was fetched.")

    df = pd.concat(all_chunks, axis=0)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]

    # Standardize column names
    rename_map = {
        "nitrogen_dioxide": "no2",
        "sulphur_dioxide": "so2",
        "carbon_monoxide": "co",
        "ozone": "o3",
    }
    df = df.rename(columns=rename_map)

    print(f"[Pollution-Smart] Total: {len(df)} rows, {df.index.min()} → {df.index.max()}")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path)

    return df


def _fetch_air_quality_chunk(latitude, longitude, start_date, end_date, pollutants):
    """Fetch a chunk of historical air quality data."""
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ",".join(pollutants),
        "start_date": start_date,
        "end_date": end_date,
        "timezone": "auto",
    }
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    if "hourly" not in data:
        raise ValueError(f"No hourly data for {start_date} to {end_date}")

    df = pd.DataFrame(data["hourly"])
    df["datetime"] = pd.to_datetime(df["time"])
    df = df.drop(columns=["time"]).set_index("datetime")
    return df


def _fetch_air_quality_forecast(latitude, longitude, pollutants,
                                forecast_days=5, past_days=2):
    """Fetch forecast (and recent past) air quality data."""
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ",".join(pollutants),
        "forecast_days": forecast_days,
        "past_days": past_days,
        "timezone": "auto",
    }
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    if "hourly" not in data:
        raise ValueError("No hourly forecast data returned")

    df = pd.DataFrame(data["hourly"])
    df["datetime"] = pd.to_datetime(df["time"])
    df = df.drop(columns=["time"]).set_index("datetime")
    return df


# ──────────────────────────────────────────────────────────────────────
#  Weather — Separate endpoints for historical vs forecast
# ──────────────────────────────────────────────────────────────────────

def fetch_weather_smart(latitude, longitude, start_date, end_date, variables,
                        chunk_days=90, save_path=None):
    """
    Fetch weather data that works for both past AND future dates.

    Uses two endpoints:
      - Historical: https://archive-api.open-meteo.com/v1/archive
      - Forecast:   https://api.open-meteo.com/v1/forecast

    Parameters
    ----------
    latitude, longitude : float
        Target coordinates.
    start_date, end_date : str
        Date range in 'YYYY-MM-DD' format.  Can include future dates.
    variables : list of str
        Weather variable names (e.g. 'temperature_2m', 'wind_speed_10m', ...).
    chunk_days : int
        Max days per API call for historical portions.
    save_path : str, optional
        If provided, save result CSV.

    Returns
    -------
    pd.DataFrame
        Hourly weather data with DatetimeIndex.
    """
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    now = datetime.utcnow()
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)

    print(f"[Weather-Smart] Fetching for ({latitude}, {longitude})")
    print(f"[Weather-Smart] Requested: {start_date} to {end_date}")

    all_chunks = []

    # ── Historical portion ──
    # The archive API data lags ~5 days behind current date
    archive_safe_end = today - timedelta(days=5)
    historical_end = min(end_dt, archive_safe_end)

    if start_dt <= historical_end:
        hist_end_str = historical_end.strftime("%Y-%m-%d")
        print(f"[Weather-Smart] Historical range: {start_date} to {hist_end_str}")

        current = start_dt
        chunk_num = 0
        while current <= historical_end:
            chunk_end = min(current + timedelta(days=chunk_days - 1), historical_end)
            c_start = current.strftime("%Y-%m-%d")
            c_end = chunk_end.strftime("%Y-%m-%d")
            chunk_num += 1
            print(f"  Historical chunk {chunk_num}: {c_start} → {c_end} ... ", end="")

            try:
                df = _fetch_weather_archive_chunk(latitude, longitude, c_start, c_end, variables)
                all_chunks.append(df)
                print(f"✓ ({len(df)} rows)")
            except Exception as e:
                print(f"✗ Error: {e}")

            current = chunk_end + timedelta(days=1)
            time.sleep(0.3)

    # ── Forecast / Recent portion ──
    # The forecast API covers recent past (past_days) + future (forecast_days)
    forecast_start = max(start_dt, archive_safe_end + timedelta(days=1))
    if end_dt > archive_safe_end:
        # Calculate how many past days to include for gap coverage
        past_days_param = max(0, (today - forecast_start).days + 1)
        past_days_param = min(past_days_param, 92)

        # Calculate forecast days
        forecast_days_param = max(0, (end_dt - today).days + 1)
        forecast_days_param = min(forecast_days_param, 16)  # API max: 16

        print(f"[Weather-Smart] Forecast range: past_days={past_days_param}, "
              f"forecast_days={forecast_days_param}")

        try:
            df = _fetch_weather_forecast(
                latitude, longitude, variables,
                past_days=past_days_param,
                forecast_days=forecast_days_param
            )
            all_chunks.append(df)
            print(f"  Forecast: ✓ ({len(df)} rows)")
        except Exception as e:
            print(f"  Forecast: ✗ Error: {e}")

    if not all_chunks:
        raise RuntimeError("No weather data was fetched.")

    df = pd.concat(all_chunks, axis=0)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]

    # Standardize column names (same as weather_fetch.py)
    rename_map = {
        "temperature_2m": "temperature",
        "relative_humidity_2m": "humidity",
        "wind_speed_10m": "wind_speed",
        "surface_pressure": "pressure",
        "dew_point_2m": "dew_point",
    }
    df = df.rename(columns=rename_map)

    print(f"[Weather-Smart] Total: {len(df)} rows, {df.index.min()} → {df.index.max()}")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path)

    return df


def _fetch_weather_archive_chunk(latitude, longitude, start_date, end_date, variables):
    """Fetch a chunk from the historical weather archive API."""
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ",".join(variables),
        "start_date": start_date,
        "end_date": end_date,
        "timezone": "auto",
    }
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    if "hourly" not in data:
        raise ValueError(f"No hourly archive data for {start_date} to {end_date}")

    df = pd.DataFrame(data["hourly"])
    df["datetime"] = pd.to_datetime(df["time"])
    df = df.drop(columns=["time"]).set_index("datetime")
    return df


def _fetch_weather_forecast(latitude, longitude, variables,
                            past_days=5, forecast_days=7):
    """Fetch from the weather forecast API (covers recent past + future)."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ",".join(variables),
        "past_days": past_days,
        "forecast_days": forecast_days,
        "timezone": "auto",
    }
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    if "hourly" not in data:
        raise ValueError("No hourly forecast data returned")

    df = pd.DataFrame(data["hourly"])
    df["datetime"] = pd.to_datetime(df["time"])
    df = df.drop(columns=["time"]).set_index("datetime")
    return df
