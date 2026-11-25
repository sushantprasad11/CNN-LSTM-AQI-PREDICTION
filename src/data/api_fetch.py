"""
API Fetch Module — Unified Interface
======================================
High-level interface that orchestrates fetching both pollution and weather data.
"""

from src.data.pollution_fetch import fetch_pollution_data
from src.data.weather_fetch import fetch_weather_data
from src.data.data_merger import merge_pollution_weather
from src.data.data_loader import load_config


def fetch_all_data(config_path="config/config.yaml"):
    """
    Fetch all required data (pollution + weather) and merge them.

    Parameters
    ----------
    config_path : str
        Path to the configuration YAML file.

    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame, pd.DataFrame)
        (pollution_df, weather_df, merged_df)
    """
    config = load_config(config_path)

    lat = config["data"]["latitude"]
    lon = config["data"]["longitude"]
    start = config["data"]["start_date"]
    end = config["data"]["end_date"]
    pollutants = config["data"]["pollutants"]
    weather_vars = config["data"]["weather_variables"]

    # Fetch pollution data
    pollution_df = fetch_pollution_data(
        latitude=lat, longitude=lon,
        start_date=start, end_date=end,
        pollutants=pollutants,
        save_path=config["data"]["raw_pollution_path"]
    )

    # Fetch weather data
    weather_df = fetch_weather_data(
        latitude=lat, longitude=lon,
        start_date=start, end_date=end,
        variables=weather_vars,
        save_path=config["data"]["raw_weather_path"]
    )

    # Merge on timestamp
    merged_df = merge_pollution_weather(
        pollution_df, weather_df,
        save_path=config["data"]["merged_path"]
    )

    return pollution_df, weather_df, merged_df
