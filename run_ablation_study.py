"""
Ablation Study Runner (Extension 2)
======================================
Runs the step-wise ablation study using the Hybrid CNN-LSTM model.
Features are added incrementally and the model is retrained at each step:
  1. Base pollutants only
  2. + Lag features
  3. + Rolling average features
  4. + Cyclic temporal features
  5. + Ratio features
  6. + Meteorological features
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
from src.models.cnn_lstm import build_cnn_lstm_model
from src.training.callbacks import get_callbacks
from src.evaluation.metrics import compute_metrics


def main():
    print("=" * 70)
    print("  ABLATION STUDY — Step-wise Feature Contribution Analysis")
    print("  Extension 2 from Synopsis")
    print("=" * 70)

    config = load_config('config/config.yaml')

    # Load the featured data
    print("\n[1] Loading featured data...")
    featured_df = pd.read_csv(config['data']['featured_path'],
                               index_col=0, parse_dates=True)
    print(f"  Shape: {featured_df.shape}")

    # Load feature groups
    with open('artifacts/feature_groups.json', 'r') as f:
        feature_groups = json.load(f)
    print(f"  Feature groups: {list(feature_groups.keys())}")

    # Configuration
    target_col = 'aqi_code'
    window_size = config['sequence']['window_size']  # 24
    test_size = config['split']['test_size']  # 0.30
    training_config = config['training']

    # Define ablation order
    group_order = [
        "base_pollutants",
        "lag_features",
        "rolling_features",
        "cyclic_features",
        "ratio_features",
        "meteorological",
    ]
    group_order = [g for g in group_order if g in feature_groups]

    results = []
    cumulative_features = []

    print(f"\n{'=' * 70}")
    print(f"  Starting Ablation Study — {len(group_order)} steps")
    print(f"{'=' * 70}")

    for step, group_name in enumerate(group_order, 1):
        group_cols = feature_groups[group_name]
        # Only include columns that exist in the DataFrame
        group_cols = [c for c in group_cols if c in featured_df.columns]
        cumulative_features.extend(group_cols)

        # Deduplicate while preserving order
        seen = set()
        unique_features = []
        for feat in cumulative_features:
            if feat not in seen:
                seen.add(feat)
                unique_features.append(feat)
        cumulative_features = unique_features

        print(f"\n{'─' * 70}")
        print(f"  Step {step}/{len(group_order)}: +{group_name} ({len(group_cols)} features)")
        print(f"  Total cumulative features: {len(cumulative_features)}")
        print(f"{'─' * 70}")

        # Split data chronologically
        split_idx = int(len(featured_df) * (1 - test_size))
        df_train = featured_df.iloc[:split_idx]
        df_test = featured_df.iloc[split_idx:]

        # Scale features (fit on train only)
        scaler = AQIScaler()
        scaler.fit(df_train, cumulative_features)
        df_train_scaled = scaler.transform(df_train)
        df_test_scaled = scaler.transform(df_test)

        # Create sequences
        X_train, y_train = create_sequences(
            df_train_scaled[cumulative_features].values,
            df_train_scaled[target_col].values.astype(int),
            window_size=window_size
        )
        X_test, y_test = create_sequences(
            df_test_scaled[cumulative_features].values,
            df_test_scaled[target_col].values.astype(int),
            window_size=window_size
        )

        # Build model
        input_shape = (X_train.shape[1], X_train.shape[2])
        num_classes = len(set(y_train))
        model = build_cnn_lstm_model(input_shape, num_classes,
                                      config['models']['cnn_lstm'])

        # Training callbacks
        callbacks = get_callbacks(f"ablation_step_{step}", training_config,
                                   checkpoint_dir="models")

        # Train
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=training_config.get("epochs", 100),
            batch_size=training_config.get("batch_size", 64),
            callbacks=callbacks,
            verbose=1,
        )

        # Evaluate
        y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
        metrics = compute_metrics(y_test, y_pred)

        step_result = {
            "step": step,
            "feature_group": group_name,
            "n_features": len(cumulative_features),
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"],
            "epochs_trained": len(history.history['loss']),
            "best_val_loss": float(min(history.history['val_loss'])),
        }
        results.append(step_result)

        print(f"\n  ✓ Step {step} Results:")
        print(f"    Accuracy:  {metrics['accuracy']:.4f}")
        print(f"    Precision: {metrics['precision']:.4f}")
        print(f"    Recall:    {metrics['recall']:.4f}")
        print(f"    F1-Score:  {metrics['f1_score']:.4f}")

    # Save results
    results_df = pd.DataFrame(results)
    os.makedirs("results/ablation", exist_ok=True)
    results_df.to_csv("results/ablation/ablation_results.csv", index=False)
    print(f"\n[Ablation] Results saved to: results/ablation/ablation_results.csv")
    print(f"\n{results_df.to_string(index=False)}")

    # ── Plot 1: Accuracy and F1 progression ──
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    x_labels = [f"Step {r['step']}\n+{r['feature_group']}" for _, r in results_df.iterrows()]

    # Performance progression
    axes[0].plot(x_labels, results_df["accuracy"], "bo-", linewidth=2.5,
                 markersize=10, label="Accuracy", zorder=5)
    axes[0].plot(x_labels, results_df["f1_score"], "rs-", linewidth=2.5,
                 markersize=10, label="F1-Score", zorder=5)
    axes[0].plot(x_labels, results_df["precision"], "g^--", linewidth=1.5,
                 markersize=8, alpha=0.7, label="Precision", zorder=4)
    axes[0].plot(x_labels, results_df["recall"], "mD--", linewidth=1.5,
                 markersize=8, alpha=0.7, label="Recall", zorder=4)

    # Add value labels
    for _, row in results_df.iterrows():
        axes[0].annotate(f"{row['accuracy']:.3f}",
                         (row['step'] - 1, row['accuracy']),
                         textcoords="offset points", xytext=(0, 12),
                         ha='center', fontsize=8, color='blue')
        axes[0].annotate(f"{row['f1_score']:.3f}",
                         (row['step'] - 1, row['f1_score']),
                         textcoords="offset points", xytext=(0, -15),
                         ha='center', fontsize=8, color='red')

    axes[0].set_ylabel("Score", fontsize=12)
    axes[0].set_title("Ablation Study — Performance by Feature Group", fontsize=13)
    axes[0].legend(fontsize=10, loc='lower right')
    axes[0].grid(True, alpha=0.3)
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].set_ylim(0, 1.05)

    # Incremental improvement
    acc_diff = results_df["accuracy"].diff().fillna(results_df["accuracy"].iloc[0])
    f1_diff = results_df["f1_score"].diff().fillna(results_df["f1_score"].iloc[0])

    x = np.arange(len(results_df))
    width = 0.35
    bars1 = axes[1].bar(x - width/2, acc_diff * 100, width,
                         label="Accuracy Δ (%)", color="#2196F3", alpha=0.85)
    bars2 = axes[1].bar(x + width/2, f1_diff * 100, width,
                         label="F1-Score Δ (%)", color="#E91E63", alpha=0.85)

    # Add value labels on bars
    for bar in bars1:
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                     f"{bar.get_height():.2f}%", ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                     f"{bar.get_height():.2f}%", ha='center', va='bottom', fontsize=8)

    axes[1].set_xticks(x)
    axes[1].set_xticklabels([r["feature_group"] for _, r in results_df.iterrows()],
                             rotation=25, ha="right")
    axes[1].set_ylabel("Incremental Change (%)", fontsize=12)
    axes[1].set_title("Feature Group Contribution (Incremental)", fontsize=13)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].axhline(y=0, color='black', linewidth=0.5)

    plt.suptitle("Step-wise Ablation Study — Hybrid CNN-LSTM", fontsize=15, y=1.02)
    plt.tight_layout()
    fig.savefig("results/ablation/ablation_chart.png", dpi=150, bbox_inches="tight")
    print(f"[Ablation] Chart saved to: results/ablation/ablation_chart.png")
    plt.close()

    # ── Plot 2: Feature count vs performance ──
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.bar(x, results_df["n_features"], color="#FF9800", alpha=0.4, label="Feature Count")
    ax2_twin = ax2.twinx()
    ax2_twin.plot(x, results_df["f1_score"], "ro-", linewidth=2.5, markersize=10,
                  label="F1-Score")
    ax2.set_xticks(x)
    ax2.set_xticklabels([r["feature_group"] for _, r in results_df.iterrows()],
                         rotation=25, ha="right")
    ax2.set_ylabel("Number of Features", fontsize=12)
    ax2_twin.set_ylabel("F1-Score", fontsize=12, color='red')
    ax2.set_title("Feature Count vs Model Performance", fontsize=13)
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='lower right')
    ax2.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    fig2.savefig("results/ablation/feature_count_vs_performance.png", dpi=150,
                 bbox_inches="tight")
    print(f"[Ablation] Chart saved to: results/ablation/feature_count_vs_performance.png")
    plt.close()

    print(f"\n{'=' * 70}")
    print(f"  ABLATION STUDY COMPLETE!")
    print(f"{'=' * 70}")
    print(f"\n  Summary:")
    for _, r in results_df.iterrows():
        delta = ""
        if r['step'] > 1:
            prev = results_df.iloc[r['step'] - 2]
            d = r['f1_score'] - prev['f1_score']
            delta = f" (Δ = {d:+.4f})"
        print(f"    Step {r['step']}: +{r['feature_group']:20s} | "
              f"F1: {r['f1_score']:.4f}{delta}")

    print(f"\n  Best configuration: Step {results_df.loc[results_df['f1_score'].idxmax(), 'step']} "
          f"({results_df.loc[results_df['f1_score'].idxmax(), 'feature_group']})")
    print(f"  Best F1: {results_df['f1_score'].max():.4f}")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
