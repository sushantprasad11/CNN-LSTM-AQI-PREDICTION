"""
Model Comparison Module
=========================
Compares performance across CNN, LSTM, and Hybrid CNN-LSTM models,
and between pollutant-only vs pollutant+meteorological conditions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def compare_models(results_dict, save_path=None):
    """
    Create a comparison table and visualization across models.

    Parameters
    ----------
    results_dict : dict
        Mapping from model_name to metrics dict.
        e.g., {"CNN": {"accuracy": 0.85, "f1_score": 0.84, ...}, ...}
    save_path : str, optional
        Directory to save comparison outputs.

    Returns
    -------
    pd.DataFrame
        Comparison DataFrame.
    """
    comparison = pd.DataFrame(results_dict).T
    comparison.index.name = "Model"

    print(f"\n{'=' * 70}")
    print(f"  MODEL COMPARISON")
    print(f"{'=' * 70}")
    print(comparison.to_string())
    print(f"{'=' * 70}")

    # Find best model
    best_model = comparison["f1_score"].idxmax()
    print(f"\n  Best model (by F1): {best_model} "
          f"(F1={comparison.loc[best_model, 'f1_score']:.4f})")

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        comparison.to_csv(os.path.join(save_path, "model_comparison.csv"))

    return comparison


def plot_model_comparison(comparison_df, save_path=None):
    """
    Plot a grouped bar chart comparing models across metrics.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        Comparison DataFrame from compare_models().
    save_path : str, optional
        Save path for the figure.
    """
    metrics = ["accuracy", "precision", "recall", "f1_score"]
    available_metrics = [m for m in metrics if m in comparison_df.columns]

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(comparison_df))
    width = 0.8 / len(available_metrics)

    colors = ["#2196F3", "#FF9800", "#4CAF50", "#E91E63"]

    for i, metric in enumerate(available_metrics):
        offset = (i - len(available_metrics) / 2 + 0.5) * width
        bars = ax.bar(x + offset, comparison_df[metric], width,
                      label=metric.replace("_", " ").title(),
                      color=colors[i % len(colors)], alpha=0.85)
        # Add value labels on bars
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Performance Comparison", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(comparison_df.index, fontsize=11)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.08)
    ax.grid(True, alpha=0.2, axis="y")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Comparison] Figure saved to: {save_path}")

    plt.show()
    return fig


def compare_pollutant_vs_weather(results_pollutant_only, results_with_weather,
                                  save_path=None):
    """
    Compare model performance: pollutant-only features vs pollutant+weather.

    This is part of Extension 1 (meteorological extension).

    Parameters
    ----------
    results_pollutant_only : dict
        Model name → metrics dict (pollutant-only features).
    results_with_weather : dict
        Model name → metrics dict (pollutant + weather features).
    save_path : str, optional
        Save directory.

    Returns
    -------
    pd.DataFrame
        Comparison DataFrame with both conditions.
    """
    rows = []
    for model_name in results_pollutant_only:
        row = {"Model": model_name, "Condition": "Pollutant Only"}
        row.update(results_pollutant_only[model_name])
        rows.append(row)

        if model_name in results_with_weather:
            row2 = {"Model": model_name, "Condition": "Pollutant + Weather"}
            row2.update(results_with_weather[model_name])
            rows.append(row2)

    comparison = pd.DataFrame(rows)

    print(f"\n{'=' * 70}")
    print(f"  POLLUTANT-ONLY vs POLLUTANT + WEATHER COMPARISON")
    print(f"{'=' * 70}")
    print(comparison.to_string(index=False))
    print(f"{'=' * 70}")

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        comparison.to_csv(os.path.join(save_path, "weather_comparison.csv"), index=False)

    return comparison
