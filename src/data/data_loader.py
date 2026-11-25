"""
Data Loader Module
===================
Utility functions for loading saved CSV data and configuration.
"""

import pandas as pd
import yaml
import os


def load_config(config_path="config/config.yaml"):
    """
    Load the project configuration from YAML file.

    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file.

    Returns
    -------
    dict
        Configuration dictionary.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_csv(filepath, index_col="datetime", parse_dates=True):
    """
    Load a CSV file with datetime index.

    Parameters
    ----------
    filepath : str
        Path to the CSV file.
    index_col : str
        Column to use as index.
    parse_dates : bool
        Whether to parse the index as datetime.

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")

    df = pd.read_csv(filepath, index_col=index_col, parse_dates=parse_dates)
    print(f"[Loader] Loaded {filepath}: {df.shape}")
    return df


def load_pollution_data(config):
    """Load raw pollution data using paths from config."""
    return load_csv(config["data"]["raw_pollution_path"])


def load_weather_data(config):
    """Load raw weather data using paths from config."""
    return load_csv(config["data"]["raw_weather_path"])


def load_merged_data(config):
    """Load merged pollution+weather data using paths from config."""
    return load_csv(config["data"]["merged_path"])


def load_aqi_data(config):
    """Load AQI-computed data using paths from config."""
    return load_csv(config["data"]["aqi_path"])


def load_featured_data(config):
    """Load feature-engineered data using paths from config."""
    return load_csv(config["data"]["featured_path"])
