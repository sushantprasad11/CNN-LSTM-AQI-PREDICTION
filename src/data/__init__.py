"""Data collection and loading subpackage."""

from src.data.pollution_fetch import fetch_pollution_data
from src.data.weather_fetch import fetch_weather_data
from src.data.data_merger import merge_pollution_weather
from src.data.data_cleaning import handle_missing_values, remove_outliers_iqr
from src.data.data_loader import load_config, load_csv
from src.data.api_fetch import fetch_all_data
from src.data.forecast_fetch import fetch_pollution_smart, fetch_weather_smart

__all__ = [
    "fetch_pollution_data",
    "fetch_weather_data",
    "fetch_pollution_smart",
    "fetch_weather_smart",
    "merge_pollution_weather",
    "handle_missing_values",
    "remove_outliers_iqr",
    "load_config",
    "load_csv",
    "fetch_all_data",
]
