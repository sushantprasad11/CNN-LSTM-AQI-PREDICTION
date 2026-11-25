"""
AQI Pipeline Runner — Resume from saved data
===============================================
Loads already-fetched data from CSV and continues the pipeline:
1. Load raw data (already fetched)
2. Merge, clean, compute AQI
3. Engineer features, create sequences
4. Train all 3 models (CNN, LSTM, CNN-LSTM)
5. Evaluate and compare
6. Run ablation study
7. Evaluate early warning system
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import warnings
warnings.filterwarnings('ignore')

# Set UTF-8 encoding for Windows console
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
import json

from src.data.data_loader import load_config
from src.data.data_merger import merge_pollution_weather
from src.data.data_cleaning import handle_missing_values, remove_outliers_iqr, report_missing, validate_data_quality
from src.preprocessing.aqi_calculator import calculate_aqi_dataframe
from src.preprocessing.feature_engineering import engineer_all_features
from src.preprocessing.scaler import AQIScaler
from src.preprocessing.sequence_generator import create_sequences, save_sequences


def main():
    print("=" * 70)
    print("  AQI PREDICTION PIPELINE -- Full Execution (Resume Mode)")
    print("=" * 70)

    config = load_config('config/config.yaml')

    # == Step 1: Load already-fetched raw data ==
    print("\n\n" + "=" * 70)
    print("  STEP 1: Loading Raw Data (already fetched)")
    print("=" * 70)

    pollution_path = config['data']['raw_pollution_path']
    weather_path = config['data']['raw_weather_path']

    pollution_df = pd.read_csv(pollution_path, index_col='datetime', parse_dates=True)
    print(f"[Pollution] Loaded: {pollution_df.shape}")

    weather_df = pd.read_csv(weather_path, index_col='datetime', parse_dates=True)
    print(f"[Weather] Loaded: {weather_df.shape}")

    # == Step 2: Merge + Clean + AQI ==
    print("\n\n" + "=" * 70)
    print("  STEP 2: Merge, Clean & Compute AQI")
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

    # == Step 3: Feature Engineering + Sequences ==
    print("\n\n" + "=" * 70)
    print("  STEP 3: Feature Engineering & Sequence Creation")
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

    # Scale -- fit on training portion only
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
    print("  DATA PIPELINE COMPLETE -- Ready for Model Training")
    print("=" * 70)
    print(f"  X_train: {X_train.shape}")
    print(f"  X_test:  {X_test.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  y_test:  {y_test.shape}")
    print(f"  Feature columns: {len(feature_cols)}")
    print(f"  Window size: {window_size}")
    print(f"  Num classes: {len(set(y_train))}")
    print("=" * 70)

    # == Step 4: Train All 3 Models ==
    print("\n\n" + "=" * 70)
    print("  STEP 4: Training All Models")
    print("=" * 70)

    from src.training.train_cnn import train_cnn
    from src.training.train_lstm import train_lstm
    from src.training.train_cnn_lstm import train_cnn_lstm
    from src.evaluation.metrics import print_model_metrics, compute_metrics
    from src.evaluation.confusion_matrix import plot_confusion_matrix
    from src.evaluation.model_comparison import compare_models, plot_model_comparison
    from src.prediction.early_warning import evaluate_early_warning

    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for saving plots

    training_config = config['training']

    # --- Train CNN ---
    print("\n" + "-" * 50)
    print("  Training CNN Model")
    print("-" * 50)
    cnn_model, cnn_history = train_cnn(
        X_train, y_train, X_test, y_test,
        config=training_config,
        model_config=config['models']['cnn'],
        checkpoint_dir="models"
    )
    cnn_pred = np.argmax(cnn_model.predict(X_test, verbose=0), axis=1)
    cnn_metrics = print_model_metrics("CNN", y_test, cnn_pred)

    # --- Train LSTM ---
    print("\n" + "-" * 50)
    print("  Training LSTM Model")
    print("-" * 50)
    lstm_model, lstm_history = train_lstm(
        X_train, y_train, X_test, y_test,
        config=training_config,
        model_config=config['models']['lstm'],
        checkpoint_dir="models"
    )
    lstm_pred = np.argmax(lstm_model.predict(X_test, verbose=0), axis=1)
    lstm_metrics = print_model_metrics("LSTM", y_test, lstm_pred)

    # --- Train Hybrid CNN-LSTM ---
    print("\n" + "-" * 50)
    print("  Training Hybrid CNN-LSTM Model")
    print("-" * 50)
    hybrid_model, hybrid_history = train_cnn_lstm(
        X_train, y_train, X_test, y_test,
        config=training_config,
        model_config=config['models']['cnn_lstm'],
        checkpoint_dir="models"
    )
    hybrid_pred = np.argmax(hybrid_model.predict(X_test, verbose=0), axis=1)
    hybrid_metrics = print_model_metrics("Hybrid CNN-LSTM", y_test, hybrid_pred)

    # == Step 5: Model Comparison ==
    print("\n\n" + "=" * 70)
    print("  STEP 5: Model Comparison")
    print("=" * 70)

    results_dict = {
        "CNN": {k: v for k, v in cnn_metrics.items() if k != 'report_dict'},
        "LSTM": {k: v for k, v in lstm_metrics.items() if k != 'report_dict'},
        "Hybrid CNN-LSTM": {k: v for k, v in hybrid_metrics.items() if k != 'report_dict'},
    }
    comparison_df = compare_models(results_dict, save_path="results/metrics")

    # Save confusion matrices
    os.makedirs("results/plots", exist_ok=True)
    plot_confusion_matrix(y_test, cnn_pred, "CNN",
                          save_path="results/plots/cm_cnn.png")
    plot_confusion_matrix(y_test, lstm_pred, "LSTM",
                          save_path="results/plots/cm_lstm.png")
    plot_confusion_matrix(y_test, hybrid_pred, "Hybrid CNN-LSTM",
                          save_path="results/plots/cm_hybrid.png")

    # Save comparison plot
    plot_model_comparison(comparison_df, save_path="results/plots/model_comparison.png")

    # == Step 6: Early Warning Evaluation ==
    print("\n\n" + "=" * 70)
    print("  STEP 6: Early Warning System Evaluation")
    print("=" * 70)

    ew_metrics = evaluate_early_warning(y_test, hybrid_pred)

    # Save early warning results
    os.makedirs("results/reports", exist_ok=True)
    with open("results/reports/early_warning_results.json", "w") as f:
        json.dump(ew_metrics, f, indent=2)

    # == Step 7: Save Training Histories ==
    print("\n\n" + "=" * 70)
    print("  STEP 7: Saving Training Histories")
    print("=" * 70)

    for name, history in [("cnn", cnn_history), ("lstm", lstm_history), ("hybrid", hybrid_history)]:
        hist_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
        with open(f"results/metrics/{name}_history.json", "w") as f:
            json.dump(hist_dict, f, indent=2)
        print(f"  Saved {name} history")

    # == Final Summary ==
    print("\n\n" + "=" * 70)
    print("  PIPELINE COMPLETE!")
    print("=" * 70)
    print(f"\n  Data:     {X_train.shape[0]} train, {X_test.shape[0]} test sequences")
    print(f"  Features: {X_train.shape[2]} per timestep, {window_size} timesteps")
    print(f"\n  Model Results (Test Set):")
    for model_name, metrics in results_dict.items():
        print(f"    {model_name:20s} | Acc: {metrics['accuracy']:.4f} | F1: {metrics['f1_score']:.4f}")
    print(f"\n  Early Warning (Hybrid):")
    print(f"    Precision: {ew_metrics['alert_precision']:.4f}")
    print(f"    Recall:    {ew_metrics['alert_recall']:.4f}")
    print(f"    F1-Score:  {ew_metrics['alert_f1_score']:.4f}")
    print("=" * 70)


if __name__ == '__main__':
    main()
