"""
Meteorological Comparison Runner (Extension 1)
=================================================
Compares model performance with and without meteorological features.
Each model (CNN, LSTM, Hybrid CNN-LSTM) is trained twice:
  - Condition A: Pollutant-only features (base + lag + rolling + cyclic + ratio)
  - Condition B: Pollutant + meteorological features (all features)

This directly addresses Extension 1 from the synopsis.
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.data.data_loader import load_config
from src.preprocessing.scaler import AQIScaler
from src.preprocessing.sequence_generator import create_sequences
from src.models.cnn import build_cnn_model
from src.models.lstm import build_lstm_model
from src.models.cnn_lstm import build_cnn_lstm_model
from src.training.callbacks import get_callbacks
from src.evaluation.metrics import compute_metrics, print_model_metrics
from src.evaluation.model_comparison import compare_pollutant_vs_weather


def train_and_evaluate(X_train, y_train, X_test, y_test,
                       model_builder, model_name, model_config,
                       training_config, checkpoint_suffix):
    """Train a model and return metrics."""
    input_shape = (X_train.shape[1], X_train.shape[2])
    num_classes = len(set(y_train))

    model = model_builder(input_shape, num_classes, model_config)

    callbacks = get_callbacks(
        f"{model_name}_{checkpoint_suffix}",
        training_config,
        checkpoint_dir="models"
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=training_config.get("epochs", 100),
        batch_size=training_config.get("batch_size", 64),
        callbacks=callbacks,
        verbose=1,
    )

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    metrics = compute_metrics(y_test, y_pred)

    return model, history, metrics, y_pred


def main():
    print("=" * 70)
    print("  METEOROLOGICAL COMPARISON — Extension 1")
    print("  Pollutant-Only vs Pollutant + Weather Features")
    print("=" * 70)

    config = load_config('config/config.yaml')

    # Load data
    print("\n[1] Loading featured data...")
    featured_df = pd.read_csv(config['data']['featured_path'],
                               index_col=0, parse_dates=True)
    print(f"  Shape: {featured_df.shape}")

    with open('artifacts/feature_groups.json', 'r') as f:
        feature_groups = json.load(f)

    # Configuration
    target_col = 'aqi_code'
    window_size = config['sequence']['window_size']
    test_size = config['split']['test_size']
    training_config = config['training']

    # Define two conditions
    # Condition A: All features EXCEPT meteorological
    pollutant_only_features = []
    for group in ["base_pollutants", "lag_features", "rolling_features",
                  "cyclic_features", "ratio_features"]:
        if group in feature_groups:
            cols = [c for c in feature_groups[group] if c in featured_df.columns]
            pollutant_only_features.extend(cols)

    # Deduplicate
    seen = set()
    pollutant_only_features = [f for f in pollutant_only_features
                                if f not in seen and not seen.add(f)]

    # Condition B: All features INCLUDING meteorological
    all_features = pollutant_only_features.copy()
    if "meteorological" in feature_groups:
        meteo_cols = [c for c in feature_groups["meteorological"]
                      if c in featured_df.columns]
        all_features.extend(meteo_cols)

    # Deduplicate
    seen = set()
    all_features = [f for f in all_features
                     if f not in seen and not seen.add(f)]

    print(f"\n  Condition A (Pollutant Only): {len(pollutant_only_features)} features")
    print(f"  Condition B (Pollutant + Weather): {len(all_features)} features")
    print(f"  Meteorological features added: {len(all_features) - len(pollutant_only_features)}")

    # Models to test
    model_configs = [
        ("CNN", build_cnn_model, config['models']['cnn']),
        ("LSTM", build_lstm_model, config['models']['lstm']),
        ("Hybrid CNN-LSTM", build_cnn_lstm_model, config['models']['cnn_lstm']),
    ]

    results_pollutant_only = {}
    results_with_weather = {}

    for condition_name, feature_list, results_store, suffix in [
        ("Condition A: Pollutant Only", pollutant_only_features,
         results_pollutant_only, "pollutant"),
        ("Condition B: Pollutant + Weather", all_features,
         results_with_weather, "weather"),
    ]:
        print(f"\n\n{'=' * 70}")
        print(f"  {condition_name}")
        print(f"  Features: {len(feature_list)}")
        print(f"{'=' * 70}")

        # Split, scale, sequence
        split_idx = int(len(featured_df) * (1 - test_size))
        df_train = featured_df.iloc[:split_idx]
        df_test = featured_df.iloc[split_idx:]

        scaler = AQIScaler()
        scaler.fit(df_train, feature_list)
        df_train_scaled = scaler.transform(df_train)
        df_test_scaled = scaler.transform(df_test)

        X_train, y_train = create_sequences(
            df_train_scaled[feature_list].values,
            df_train_scaled[target_col].values.astype(int),
            window_size=window_size
        )
        X_test, y_test = create_sequences(
            df_test_scaled[feature_list].values,
            df_test_scaled[target_col].values.astype(int),
            window_size=window_size
        )

        print(f"  X_train: {X_train.shape}, X_test: {X_test.shape}")

        for model_name, builder, m_config in model_configs:
            print(f"\n  {'─' * 50}")
            print(f"  Training: {model_name} ({condition_name})")
            print(f"  {'─' * 50}")

            model, history, metrics, y_pred = train_and_evaluate(
                X_train, y_train, X_test, y_test,
                builder, model_name.lower().replace(" ", "_").replace("-", "_"),
                m_config, training_config, suffix
            )

            results_store[model_name] = {
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1_score"],
            }

            print(f"  ✓ {model_name}: Acc={metrics['accuracy']:.4f}, "
                  f"F1={metrics['f1_score']:.4f}")

    # Build comparison table
    os.makedirs("results/meteorological", exist_ok=True)

    comparison_df = compare_pollutant_vs_weather(
        results_pollutant_only, results_with_weather,
        save_path="results/meteorological"
    )

    # Calculate improvement
    print(f"\n\n{'=' * 70}")
    print(f"  METEOROLOGICAL IMPACT ANALYSIS")
    print(f"{'=' * 70}")

    improvement_rows = []
    for model_name in results_pollutant_only:
        if model_name in results_with_weather:
            po = results_pollutant_only[model_name]
            pw = results_with_weather[model_name]
            improvement = {
                "Model": model_name,
                "Acc (Poll Only)": po["accuracy"],
                "Acc (+ Weather)": pw["accuracy"],
                "Acc Δ": pw["accuracy"] - po["accuracy"],
                "F1 (Poll Only)": po["f1_score"],
                "F1 (+ Weather)": pw["f1_score"],
                "F1 Δ": pw["f1_score"] - po["f1_score"],
            }
            improvement_rows.append(improvement)
            print(f"\n  {model_name}:")
            print(f"    Accuracy: {po['accuracy']:.4f} → {pw['accuracy']:.4f} "
                  f"(Δ = {pw['accuracy'] - po['accuracy']:+.4f})")
            print(f"    F1-Score: {po['f1_score']:.4f} → {pw['f1_score']:.4f} "
                  f"(Δ = {pw['f1_score'] - po['f1_score']:+.4f})")

    improvement_df = pd.DataFrame(improvement_rows)
    improvement_df.to_csv("results/meteorological/weather_impact.csv", index=False)

    # ── Visualization ──
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    model_names = list(results_pollutant_only.keys())
    x = np.arange(len(model_names))
    width = 0.35

    # Accuracy comparison
    acc_po = [results_pollutant_only[m]["accuracy"] for m in model_names]
    acc_pw = [results_with_weather[m]["accuracy"] for m in model_names]
    bars1 = axes[0].bar(x - width/2, acc_po, width, label="Pollutant Only",
                         color="#2196F3", alpha=0.85)
    bars2 = axes[0].bar(x + width/2, acc_pw, width, label="Pollutant + Weather",
                         color="#4CAF50", alpha=0.85)
    for bar in bars1:
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                     f"{bar.get_height():.3f}", ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                     f"{bar.get_height():.3f}", ha='center', va='bottom', fontsize=9)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(model_names, fontsize=11)
    axes[0].set_ylabel("Accuracy", fontsize=12)
    axes[0].set_title("Accuracy: Pollutant-Only vs With Weather", fontsize=13)
    axes[0].legend(fontsize=10)
    axes[0].set_ylim(0, 1.08)
    axes[0].grid(True, alpha=0.2, axis='y')

    # F1 comparison
    f1_po = [results_pollutant_only[m]["f1_score"] for m in model_names]
    f1_pw = [results_with_weather[m]["f1_score"] for m in model_names]
    bars3 = axes[1].bar(x - width/2, f1_po, width, label="Pollutant Only",
                         color="#FF9800", alpha=0.85)
    bars4 = axes[1].bar(x + width/2, f1_pw, width, label="Pollutant + Weather",
                         color="#E91E63", alpha=0.85)
    for bar in bars3:
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                     f"{bar.get_height():.3f}", ha='center', va='bottom', fontsize=9)
    for bar in bars4:
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                     f"{bar.get_height():.3f}", ha='center', va='bottom', fontsize=9)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(model_names, fontsize=11)
    axes[1].set_ylabel("F1-Score", fontsize=12)
    axes[1].set_title("F1-Score: Pollutant-Only vs With Weather", fontsize=13)
    axes[1].legend(fontsize=10)
    axes[1].set_ylim(0, 1.08)
    axes[1].grid(True, alpha=0.2, axis='y')

    plt.suptitle("Extension 1 — Meteorological Feature Impact", fontsize=15, y=1.02)
    plt.tight_layout()
    fig.savefig("results/meteorological/weather_comparison_chart.png",
                dpi=150, bbox_inches="tight")
    print(f"\n[Meteo] Chart saved to: results/meteorological/weather_comparison_chart.png")
    plt.close()

    print(f"\n{'=' * 70}")
    print(f"  METEOROLOGICAL COMPARISON COMPLETE!")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
