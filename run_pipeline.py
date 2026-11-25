"""
AQI Pipeline Runner — Data Collection & Preprocessing
========================================================
Runs Steps 1-4 of the pipeline (no TensorFlow needed):
1. Fetch pollution data from Open-Meteo
2. Fetch weather data from Open-Meteo
3. Merge, clean, compute AQI
4. Engineer features, create sequences
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import warnings
warnings.filterwarnings('ignore')

from src.data.data_loader import load_config
from src.data.pollution_fetch import fetch_pollution_data
from src.data.weather_fetch import fetch_weather_data
from src.data.data_merger import merge_pollution_weather
from src.data.data_cleaning import handle_missing_values, remove_outliers_iqr, report_missing, validate_data_quality
from src.preprocessing.aqi_calculator import calculate_aqi_dataframe
from src.preprocessing.feature_engineering import engineer_all_features
from src.preprocessing.scaler import AQIScaler
from src.preprocessing.sequence_generator import create_sequences, save_sequences

import json

def main():
    print("=" * 70)
    print("  AQI PREDICTION PIPELINE — Data Collection & Preprocessing")
    print("=" * 70)

    config = load_config('config/config.yaml')

    # ── Step 1: Fetch Pollution Data ──────────────────────────────────
    print("\n\n" + "=" * 70)
    print("  STEP 1: Fetching Air Pollution Data")
    print("=" * 70)

    pollution_df = fetch_pollution_data(
        latitude=config['data']['latitude'],
        longitude=config['data']['longitude'],
        start_date=config['data']['start_date'],
        end_date=config['data']['end_date'],
        pollutants=config['data']['pollutants'],
        save_path=config['data']['raw_pollution_path']
    )

    # ── Step 2: Fetch Weather Data ────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("  STEP 2: Fetching Meteorological Data")
    print("=" * 70)

    weather_df = fetch_weather_data(
        latitude=config['data']['latitude'],
        longitude=config['data']['longitude'],
        start_date=config['data']['start_date'],
        end_date=config['data']['end_date'],
        variables=config['data']['weather_variables'],
        save_path=config['data']['raw_weather_path']
    )

    # ── Step 3: Merge + Clean + AQI ───────────────────────────────────
    print("\n\n" + "=" * 70)
    print("  STEP 3: Merge, Clean & Compute AQI")
    print("=" * 70)

    merged_df = merge_pollution_weather(
        pollution_df, weather_df,
        save_path=config['data']['merged_path']
    )

    # Missing value report
    report_missing(merged_df, 'Merged Data')

    # Handle missing values
    merged_df = handle_missing_values(merged_df, method='interpolate', limit=6)
    print(f"After interpolation: {merged_df.shape}")

    # Outlier treatment
    merged_df = remove_outliers_iqr(merged_df, factor=3.0)

    # Data quality check
    validate_data_quality(merged_df, 'Cleaned Data')

    # Save cleaned merged data
    merged_df.to_csv(config['data']['merged_path'])

    # Compute AQI
    aqi_df = calculate_aqi_dataframe(merged_df, save_path=config['data']['aqi_path'])

    # ── Step 4: Feature Engineering + Sequences ───────────────────────
    print("\n\n" + "=" * 70)
    print("  STEP 4: Feature Engineering & Sequence Creation")
    print("=" * 70)

    featured_df, feature_groups = engineer_all_features(
        aqi_df, config,
        save_path=config['data']['featured_path']
    )

    # Save feature groups for ablation study
    os.makedirs('artifacts', exist_ok=True)
    with open('artifacts/feature_groups.json', 'w') as f:
        json.dump(feature_groups, f, indent=2)
    print("[Features] Feature groups saved to artifacts/feature_groups.json")

    # Identify feature and target columns
    target_col = 'aqi_code'
    exclude_cols = ['aqi', 'aqi_category', 'aqi_code']
    feature_cols = [c for c in featured_df.columns if c not in exclude_cols]

    print(f"\nFeature columns: {len(feature_cols)}")
    print(f"Target: {target_col}")

    # Scale — fit on training portion only
    split_idx = int(len(featured_df) * 0.70)
    df_train = featured_df.iloc[:split_idx]
    df_test = featured_df.iloc[split_idx:]

    scaler = AQIScaler()
    scaler.fit(df_train, feature_cols)
    df_train_scaled = scaler.transform(df_train)
    df_test_scaled = scaler.transform(df_test)
    scaler.save('artifacts/scaler.joblib')

    # Create sequences
    window_size = config['sequence']['window_size']  # 24

    X_train, y_train = create_sequences(
        df_train_scaled[feature_cols].values,
        df_train_scaled[target_col].values.astype(int),
        window_size=window_size
    )

    X_test, y_test = create_sequences(
        df_test_scaled[feature_cols].values,
        df_test_scaled[target_col].values.astype(int),
        window_size=window_size
    )

    # Save sequences
    save_sequences(X_train, X_test, y_train, y_test,
                   config['data']['sequences_path'])

    print("\n\n" + "=" * 70)
    print("  PIPELINE COMPLETE — Ready for Model Training")
    print("=" * 70)
    print(f"  X_train: {X_train.shape}")
    print(f"  X_test:  {X_test.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  y_test:  {y_test.shape}")
    print(f"  Feature columns: {len(feature_cols)}")
    print(f"  Window size: {window_size}")
    print(f"  Num classes: {len(set(y_train))}")
    print("=" * 70)


if __name__ == '__main__':
    main()
