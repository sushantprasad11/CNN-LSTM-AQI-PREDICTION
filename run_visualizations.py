"""
Generate Training Visualizations & Statistical Tests
=======================================================
Creates:
1. Training history curves (loss + accuracy) for all 3 models
2. McNemar's statistical tests between model pairs
3. Combined results summary report
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
from src.preprocessing.sequence_generator import load_sequences
from src.evaluation.statistical_tests import mcnemar_test


def plot_training_histories():
    """Plot training history curves for all 3 models."""
    print("\n[1] Generating Training History Plots...")

    histories = {}
    colors = {
        "cnn": {"train": "#2196F3", "val": "#90CAF9", "label": "CNN"},
        "lstm": {"train": "#FF9800", "val": "#FFCC80", "label": "LSTM"},
        "hybrid": {"train": "#4CAF50", "val": "#A5D6A7", "label": "Hybrid CNN-LSTM"},
    }

    for name in ["cnn", "lstm", "hybrid"]:
        path = f"results/metrics/{name}_history.json"
        if os.path.exists(path):
            with open(path, 'r') as f:
                histories[name] = json.load(f)
            print(f"  Loaded {name} history: {len(histories[name]['loss'])} epochs")

    if not histories:
        print("  No training histories found!")
        return

    os.makedirs("results/plots", exist_ok=True)

    # ── Plot 1: Individual model training curves ──
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))

    for idx, (name, hist) in enumerate(histories.items()):
        c = colors[name]
        epochs = range(1, len(hist['loss']) + 1)

        # Loss
        axes[0][idx].plot(epochs, hist['loss'], color=c['train'],
                          linewidth=2, label='Train Loss')
        axes[0][idx].plot(epochs, hist['val_loss'], color=c['val'],
                          linewidth=2, linestyle='--', label='Val Loss')
        axes[0][idx].set_title(f"{c['label']} — Loss", fontsize=12)
        axes[0][idx].set_xlabel("Epoch")
        axes[0][idx].set_ylabel("Loss")
        axes[0][idx].legend(fontsize=9)
        axes[0][idx].grid(True, alpha=0.3)

        # Mark best epoch
        best_epoch = np.argmin(hist['val_loss']) + 1
        best_val_loss = min(hist['val_loss'])
        axes[0][idx].axvline(x=best_epoch, color='red', linestyle=':',
                              alpha=0.7, label=f'Best: Epoch {best_epoch}')
        axes[0][idx].scatter([best_epoch], [best_val_loss], color='red',
                              s=80, zorder=5, marker='*')

        # Accuracy
        axes[1][idx].plot(epochs, hist['accuracy'], color=c['train'],
                          linewidth=2, label='Train Acc')
        axes[1][idx].plot(epochs, hist['val_accuracy'], color=c['val'],
                          linewidth=2, linestyle='--', label='Val Acc')
        axes[1][idx].set_title(f"{c['label']} — Accuracy", fontsize=12)
        axes[1][idx].set_xlabel("Epoch")
        axes[1][idx].set_ylabel("Accuracy")
        axes[1][idx].legend(fontsize=9)
        axes[1][idx].grid(True, alpha=0.3)

        best_acc_epoch = np.argmax(hist['val_accuracy']) + 1
        best_val_acc = max(hist['val_accuracy'])
        axes[1][idx].axvline(x=best_acc_epoch, color='red', linestyle=':',
                              alpha=0.7)
        axes[1][idx].scatter([best_acc_epoch], [best_val_acc], color='red',
                              s=80, zorder=5, marker='*')

    plt.suptitle("Training History — All Models", fontsize=15, y=1.02)
    plt.tight_layout()
    fig.savefig("results/plots/training_history.png", dpi=150, bbox_inches="tight")
    print(f"  Saved: results/plots/training_history.png")
    plt.close()

    # ── Plot 2: Overlay comparison ──
    fig2, axes2 = plt.subplots(1, 2, figsize=(16, 6))

    for name, hist in histories.items():
        c = colors[name]
        epochs = range(1, len(hist['val_loss']) + 1)
        axes2[0].plot(epochs, hist['val_loss'], linewidth=2,
                      color=c['train'], label=c['label'])
        axes2[1].plot(epochs, hist['val_accuracy'], linewidth=2,
                      color=c['train'], label=c['label'])

    axes2[0].set_title("Validation Loss Comparison", fontsize=13)
    axes2[0].set_xlabel("Epoch")
    axes2[0].set_ylabel("Validation Loss")
    axes2[0].legend(fontsize=10)
    axes2[0].grid(True, alpha=0.3)

    axes2[1].set_title("Validation Accuracy Comparison", fontsize=13)
    axes2[1].set_xlabel("Epoch")
    axes2[1].set_ylabel("Validation Accuracy")
    axes2[1].legend(fontsize=10)
    axes2[1].grid(True, alpha=0.3)

    plt.suptitle("Model Convergence Comparison", fontsize=15, y=1.02)
    plt.tight_layout()
    fig2.savefig("results/plots/convergence_comparison.png", dpi=150,
                 bbox_inches="tight")
    print(f"  Saved: results/plots/convergence_comparison.png")
    plt.close()

    # ── Plot 3: Learning rate schedule ──
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    for name, hist in histories.items():
        c = colors[name]
        if 'learning_rate' in hist:
            epochs = range(1, len(hist['learning_rate']) + 1)
            ax3.plot(epochs, hist['learning_rate'], linewidth=2,
                     color=c['train'], label=c['label'], marker='o', markersize=3)
    ax3.set_title("Learning Rate Schedule", fontsize=13)
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Learning Rate")
    ax3.set_yscale('log')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    fig3.savefig("results/plots/learning_rate_schedule.png", dpi=150,
                 bbox_inches="tight")
    print(f"  Saved: results/plots/learning_rate_schedule.png")
    plt.close()

    return histories


def run_statistical_tests():
    """Run McNemar's test between all model pairs."""
    print("\n[2] Running Statistical Tests (McNemar's)...")

    config = load_config('config/config.yaml')

    # Load sequences and models to get predictions
    sequences_path = config['data']['sequences_path']
    if not os.path.exists(sequences_path):
        print("  Sequences file not found, skipping statistical tests.")
        return None

    X_train, X_test, y_train, y_test = load_sequences(sequences_path)

    # Load each model and get predictions
    import tensorflow as tf

    predictions = {}
    model_paths = {
        "CNN": "models/cnn/best_model.keras",
        "LSTM": "models/lstm/best_model.keras",
        "Hybrid CNN-LSTM": "models/cnn_lstm/best_model.keras",
    }

    for model_name, path in model_paths.items():
        if os.path.exists(path):
            print(f"  Loading {model_name} from {path}...")
            model = tf.keras.models.load_model(path)
            y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
            predictions[model_name] = y_pred
            print(f"    {model_name}: {len(y_pred)} predictions generated")
        else:
            print(f"  Model not found: {path}")

    if len(predictions) < 2:
        print("  Need at least 2 models for statistical tests.")
        return None

    # Pairwise McNemar's tests
    model_names = list(predictions.keys())
    test_results = []

    print(f"\n{'=' * 60}")
    print(f"  McNEMAR'S STATISTICAL TESTS")
    print(f"{'=' * 60}")

    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            m1, m2 = model_names[i], model_names[j]
            result = mcnemar_test(y_test, predictions[m1], predictions[m2],
                                   model_name_1=m1, model_name_2=m2)
            test_results.append(result)

    # Save results
    os.makedirs("results/reports", exist_ok=True)
    test_df = pd.DataFrame(test_results)
    test_df.to_csv("results/reports/mcnemar_tests.csv", index=False)
    print(f"\n  Results saved to: results/reports/mcnemar_tests.csv")

    return test_results, predictions, y_test


def generate_summary_report():
    """Generate a comprehensive summary report."""
    print("\n[3] Generating Summary Report...")

    os.makedirs("results/reports", exist_ok=True)

    report = {
        "project": "AQI Prediction using Hybrid CNN-LSTM",
        "city": "Delhi, India",
        "data_period": "2022-01-01 to 2025-12-31",
    }

    # Load model comparison
    if os.path.exists("results/metrics/model_comparison.csv"):
        comp = pd.read_csv("results/metrics/model_comparison.csv")
        report["model_comparison"] = comp.to_dict(orient='records')

    # Load early warning results
    if os.path.exists("results/reports/early_warning_results.json"):
        with open("results/reports/early_warning_results.json", 'r') as f:
            report["early_warning"] = json.load(f)

    # Load ablation results
    if os.path.exists("results/ablation/ablation_results.csv"):
        ablation = pd.read_csv("results/ablation/ablation_results.csv")
        report["ablation_study"] = ablation.to_dict(orient='records')

    # Load meteorological comparison
    if os.path.exists("results/meteorological/weather_impact.csv"):
        meteo = pd.read_csv("results/meteorological/weather_impact.csv")
        report["meteorological_comparison"] = meteo.to_dict(orient='records')

    # Training summary
    training_summary = {}
    for name in ["cnn", "lstm", "hybrid"]:
        path = f"results/metrics/{name}_history.json"
        if os.path.exists(path):
            with open(path, 'r') as f:
                hist = json.load(f)
            training_summary[name] = {
                "total_epochs": len(hist['loss']),
                "best_val_loss": float(min(hist['val_loss'])),
                "best_val_accuracy": float(max(hist['val_accuracy'])),
                "final_train_accuracy": float(hist['accuracy'][-1]),
                "final_val_accuracy": float(hist['val_accuracy'][-1]),
            }
    report["training_summary"] = training_summary

    with open("results/reports/full_results_summary.json", 'w') as f:
        json.dump(report, f, indent=2)

    print(f"  Summary saved to: results/reports/full_results_summary.json")

    # Print summary
    print(f"\n{'=' * 70}")
    print(f"  FULL RESULTS SUMMARY")
    print(f"{'=' * 70}")

    if "model_comparison" in report:
        print(f"\n  Model Performance (Test Set):")
        for m in report["model_comparison"]:
            print(f"    {m['Model']:20s} | Acc: {m['accuracy']:.4f} | "
                  f"P: {m['precision']:.4f} | R: {m['recall']:.4f} | "
                  f"F1: {m['f1_score']:.4f}")

    if "early_warning" in report:
        ew = report["early_warning"]
        print(f"\n  Early Warning System (Hybrid CNN-LSTM):")
        print(f"    Precision: {ew['alert_precision']:.4f}")
        print(f"    Recall:    {ew['alert_recall']:.4f}")
        print(f"    F1-Score:  {ew['alert_f1_score']:.4f}")

    print(f"\n  Training Summary:")
    for name, ts in training_summary.items():
        print(f"    {name:8s} | Epochs: {ts['total_epochs']:3d} | "
              f"Best Val Loss: {ts['best_val_loss']:.4f} | "
              f"Best Val Acc: {ts['best_val_accuracy']:.4f}")

    print(f"{'=' * 70}")
    return report


def main():
    print("=" * 70)
    print("  POST-TRAINING ANALYSIS — Visualizations & Statistical Tests")
    print("=" * 70)

    # Step 1: Training visualizations
    histories = plot_training_histories()

    # Step 2: Statistical tests
    test_results = run_statistical_tests()

    # Step 3: Summary report
    report = generate_summary_report()

    print(f"\n{'=' * 70}")
    print(f"  ALL ANALYSIS COMPLETE!")
    print(f"{'=' * 70}")
    print(f"\n  Generated Files:")
    print(f"    results/plots/training_history.png")
    print(f"    results/plots/convergence_comparison.png")
    print(f"    results/plots/learning_rate_schedule.png")
    print(f"    results/reports/mcnemar_tests.csv")
    print(f"    results/reports/full_results_summary.json")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
