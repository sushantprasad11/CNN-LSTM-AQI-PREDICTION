"""
AQI Pipeline — Training & Evaluation Only
============================================
Loads pre-processed sequences and runs model training + evaluation.
Data processing was already completed successfully.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import warnings
warnings.filterwarnings('ignore')

import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import json
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend — must be before pyplot import

from src.preprocessing.sequence_generator import load_sequences
from src.training.train_cnn import train_cnn
from src.training.train_lstm import train_lstm
from src.training.train_cnn_lstm import train_cnn_lstm
from src.evaluation.metrics import print_model_metrics
from src.evaluation.confusion_matrix import plot_confusion_matrix
from src.evaluation.model_comparison import compare_models, plot_model_comparison
from src.prediction.early_warning import evaluate_early_warning
from src.data.data_loader import load_config


def main():
    print("=" * 70)
    print("  AQI PREDICTION — Model Training & Evaluation")
    print("=" * 70)

    config = load_config('config/config.yaml')

    # Load pre-processed sequences
    print("\n[Loading] Pre-processed sequences...")
    X_train, X_test, y_train, y_test = load_sequences(config['data']['sequences_path'])
    print(f"  X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"  Classes in train: {sorted(set(y_train))}")
    print(f"  Classes in test:  {sorted(set(y_test))}")

    training_config = config['training']

    # ================================================================
    #  TRAIN CNN
    # ================================================================
    print("\n\n" + "=" * 70)
    print("  Training CNN Model")
    print("=" * 70)
    cnn_model, cnn_history = train_cnn(
        X_train, y_train, X_test, y_test,
        config=training_config,
        model_config=config['models']['cnn'],
        checkpoint_dir="models"
    )
    cnn_pred = np.argmax(cnn_model.predict(X_test, verbose=0), axis=1)
    cnn_metrics = print_model_metrics("CNN", y_test, cnn_pred)

    # ================================================================
    #  TRAIN LSTM
    # ================================================================
    print("\n\n" + "=" * 70)
    print("  Training LSTM Model")
    print("=" * 70)
    lstm_model, lstm_history = train_lstm(
        X_train, y_train, X_test, y_test,
        config=training_config,
        model_config=config['models']['lstm'],
        checkpoint_dir="models"
    )
    lstm_pred = np.argmax(lstm_model.predict(X_test, verbose=0), axis=1)
    lstm_metrics = print_model_metrics("LSTM", y_test, lstm_pred)

    # ================================================================
    #  TRAIN HYBRID CNN-LSTM
    # ================================================================
    print("\n\n" + "=" * 70)
    print("  Training Hybrid CNN-LSTM Model")
    print("=" * 70)
    hybrid_model, hybrid_history = train_cnn_lstm(
        X_train, y_train, X_test, y_test,
        config=training_config,
        model_config=config['models']['cnn_lstm'],
        checkpoint_dir="models"
    )
    hybrid_pred = np.argmax(hybrid_model.predict(X_test, verbose=0), axis=1)
    hybrid_metrics = print_model_metrics("Hybrid CNN-LSTM", y_test, hybrid_pred)

    # ================================================================
    #  MODEL COMPARISON
    # ================================================================
    print("\n\n" + "=" * 70)
    print("  Model Comparison")
    print("=" * 70)

    results_dict = {
        "CNN": {k: v for k, v in cnn_metrics.items() if k != 'report_dict'},
        "LSTM": {k: v for k, v in lstm_metrics.items() if k != 'report_dict'},
        "Hybrid CNN-LSTM": {k: v for k, v in hybrid_metrics.items() if k != 'report_dict'},
    }

    os.makedirs("results/metrics", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)
    os.makedirs("results/reports", exist_ok=True)

    comparison_df = compare_models(results_dict, save_path="results/metrics")

    # Confusion matrices
    plot_confusion_matrix(y_test, cnn_pred, "CNN",
                          save_path="results/plots/cm_cnn.png")
    plot_confusion_matrix(y_test, lstm_pred, "LSTM",
                          save_path="results/plots/cm_lstm.png")
    plot_confusion_matrix(y_test, hybrid_pred, "Hybrid CNN-LSTM",
                          save_path="results/plots/cm_hybrid.png")

    # Comparison chart
    plot_model_comparison(comparison_df,
                          save_path="results/plots/model_comparison.png")

    # ================================================================
    #  EARLY WARNING EVALUATION
    # ================================================================
    print("\n\n" + "=" * 70)
    print("  Early Warning System Evaluation")
    print("=" * 70)

    ew_metrics = evaluate_early_warning(y_test, hybrid_pred)
    with open("results/reports/early_warning_results.json", "w") as f:
        json.dump(ew_metrics, f, indent=2)

    # ================================================================
    #  SAVE TRAINING HISTORIES
    # ================================================================
    for name, history in [("cnn", cnn_history), ("lstm", lstm_history), ("hybrid", hybrid_history)]:
        hist_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
        with open(f"results/metrics/{name}_history.json", "w") as f:
            json.dump(hist_dict, f, indent=2)
        print(f"[Saved] {name} training history")

    # ================================================================
    #  FINAL SUMMARY
    # ================================================================
    print("\n\n" + "=" * 70)
    print("  ALL COMPLETE!")
    print("=" * 70)
    print(f"\n  Data: {X_train.shape[0]} train / {X_test.shape[0]} test sequences")
    print(f"  Shape: ({X_train.shape[1]} timesteps x {X_train.shape[2]} features)")
    print(f"\n  Model Results:")
    for model_name, metrics in results_dict.items():
        print(f"    {model_name:20s} | Acc: {metrics['accuracy']:.4f} | F1: {metrics['f1_score']:.4f}")
    print(f"\n  Early Warning (Hybrid CNN-LSTM):")
    print(f"    Precision: {ew_metrics['alert_precision']:.4f}")
    print(f"    Recall:    {ew_metrics['alert_recall']:.4f}")
    print(f"    F1:        {ew_metrics['alert_f1_score']:.4f}")
    print("=" * 70)


if __name__ == '__main__':
    main()
