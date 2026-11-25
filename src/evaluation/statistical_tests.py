"""
Statistical Tests Module
==========================
Optional statistical tests for comparing model performances
(e.g., McNemar's test between classifiers).
"""

import numpy as np
from scipy.stats import chi2


def mcnemar_test(y_true, y_pred_1, y_pred_2, model_name_1="Model A", model_name_2="Model B"):
    """
    Perform McNemar's test to compare two classifiers.

    Tests whether the disagreements between two classifiers are symmetric.

    Parameters
    ----------
    y_true : array-like
        True labels.
    y_pred_1 : array-like
        Predictions from model 1.
    y_pred_2 : array-like
        Predictions from model 2.
    model_name_1, model_name_2 : str
        Model names for display.

    Returns
    -------
    dict
        Test results with statistic and p-value.
    """
    correct_1 = (y_pred_1 == y_true)
    correct_2 = (y_pred_2 == y_true)

    # Build contingency table
    # b = Model 1 correct, Model 2 wrong
    # c = Model 1 wrong, Model 2 correct
    b = np.sum(correct_1 & ~correct_2)
    c = np.sum(~correct_1 & correct_2)

    # McNemar statistic (with continuity correction)
    if b + c == 0:
        statistic = 0.0
        p_value = 1.0
    else:
        statistic = (abs(b - c) - 1) ** 2 / (b + c)
        p_value = 1 - chi2.cdf(statistic, df=1)

    result = {
        "model_1": model_name_1,
        "model_2": model_name_2,
        "b (M1 right, M2 wrong)": int(b),
        "c (M1 wrong, M2 right)": int(c),
        "chi2_statistic": round(statistic, 4),
        "p_value": round(p_value, 6),
        "significant_at_0.05": p_value < 0.05,
    }

    print(f"\n[McNemar Test] {model_name_1} vs {model_name_2}")
    print(f"  b = {b}, c = {c}")
    print(f"  χ² = {statistic:.4f}, p = {p_value:.6f}")
    if p_value < 0.05:
        print(f"  → Statistically significant difference (p < 0.05)")
    else:
        print(f"  → No statistically significant difference (p ≥ 0.05)")

    return result
