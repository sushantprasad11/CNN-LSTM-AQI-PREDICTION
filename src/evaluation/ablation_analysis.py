"""
Ablation Study Module (Extension 2)
======================================
Performs a step-wise ablation analysis to quantify the incremental
contribution of each engineered feature category to model performance.

As per the synopsis: features are added incrementally in this order:
1. Base pollutants only
2. + Lag features
3. + Rolling average features
4. + Cyclic temporal features
5. + Ratio features
6. + Meteorological features (Extension 1)

At each step, the model is retrained and evaluated.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


def run_ablation_study(df, feature_groups, target_col, model_builder,
                       train_config, window_size=24, test_size=0.30,
                       results_dir="results"):
    """
    Execute the step-wise ablation study.

    Parameters
    ----------
    df : pd.DataFrame
        Full featured DataFrame (scaled).
    feature_groups : dict
        Mapping from group name to list of column names.
        Order matters: features are added incrementally.
    target_col : str
        Target column name (e.g., 'aqi_code').
    model_builder : callable
        Function that builds a model: model_builder(input_shape, num_classes) -> model.
    train_config : dict
        Training configuration (epochs, batch_size, etc.).
    window_size : int
        Sequence window size.
    test_size : float
        Fraction for test set.
    results_dir : str
        Directory to save results.

    Returns
    -------
    pd.DataFrame
        Ablation results with metrics at each step.
    """
    from src.preprocessing.sequence_generator import create_sequences, split_sequences
    from src.evaluation.metrics import compute_metrics
    from src.training.callbacks import get_callbacks

    # Define the order of feature groups for ablation
    group_order = [
        "base_pollutants",
        "lag_features",
        "rolling_features",
        "cyclic_features",
        "ratio_features",
        "meteorological",
    ]

    # Filter to only groups that exist
    group_order = [g for g in group_order if g in feature_groups]

    results = []
    cumulative_features = []

    print(f"\n{'=' * 70}")
    print(f"  ABLATION STUDY — Step-wise Feature Contribution Analysis")
    print(f"{'=' * 70}")

    for step, group_name in enumerate(group_order, 1):
        group_cols = feature_groups[group_name]
        # Only include columns that exist in the DataFrame
        group_cols = [c for c in group_cols if c in df.columns]
        cumulative_features.extend(group_cols)

        # Deduplicate while preserving order
        seen = set()
        unique_features = []
        for f in cumulative_features:
            if f not in seen:
                seen.add(f)
                unique_features.append(f)
        cumulative_features = unique_features

        print(f"\n--- Step {step}: +{group_name} ({len(group_cols)} features) ---")
        print(f"  Total features so far: {len(cumulative_features)}")

        # Extract features and target
        X_data = df[cumulative_features].values
        y_data = df[target_col].values.astype(int)

        # Create sequences
        X_seq, y_seq = create_sequences(X_data, y_data, window_size)

        # Split
        split_idx = int(len(X_seq) * (1 - test_size))
        X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
        y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]

        # Build and train model
        input_shape = (X_train.shape[1], X_train.shape[2])
        num_classes = len(set(y_train))
        model = model_builder(input_shape, num_classes)

        callbacks = get_callbacks(f"ablation_step_{step}", train_config)

        model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=train_config.get("epochs", 100),
            batch_size=train_config.get("batch_size", 64),
            callbacks=callbacks,
            verbose=0,
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
        }
        results.append(step_result)

        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1-Score: {metrics['f1_score']:.4f}")

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Save results
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, "ablation_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\n[Ablation] Results saved to: {results_path}")

    return results_df


def plot_ablation_results(results_df, save_path=None):
    """
    Visualize ablation study results showing incremental improvement.

    Parameters
    ----------
    results_df : pd.DataFrame
        Ablation results with columns: step, feature_group, metrics.
    save_path : str, optional
        Save path for the figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    x_labels = [f"Step {r['step']}\n+{r['feature_group']}"
                for _, r in results_df.iterrows()]

    # Plot 1: Accuracy and F1
    axes[0].plot(x_labels, results_df["accuracy"], "bo-", linewidth=2, label="Accuracy")
    axes[0].plot(x_labels, results_df["f1_score"], "rs-", linewidth=2, label="F1-Score")
    axes[0].set_ylabel("Score")
    axes[0].set_title("Ablation Study — Performance by Feature Group")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].tick_params(axis="x", rotation=30)

    # Plot 2: Incremental improvement
    acc_diff = results_df["accuracy"].diff().fillna(results_df["accuracy"].iloc[0])
    f1_diff = results_df["f1_score"].diff().fillna(results_df["f1_score"].iloc[0])

    x = np.arange(len(results_df))
    width = 0.35
    axes[1].bar(x - width / 2, acc_diff, width, label="Accuracy Δ", color="steelblue")
    axes[1].bar(x + width / 2, f1_diff, width, label="F1-Score Δ", color="indianred")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([r["feature_group"] for _, r in results_df.iterrows()],
                            rotation=30, ha="right")
    axes[1].set_ylabel("Incremental Change")
    axes[1].set_title("Feature Group Contribution (Incremental)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Step-wise Ablation Study Results", fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Ablation] Figure saved to: {save_path}")

    plt.show()
    return fig
