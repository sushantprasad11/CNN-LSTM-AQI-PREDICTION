"""
Data Cleaning Module
=====================
Handles missing values, outlier detection, and data quality checks
for the raw pollution and weather datasets.
"""

import pandas as pd
import numpy as np


def report_missing(df, dataset_name="Dataset"):
    """
    Print a summary of missing values in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    dataset_name : str
        Name label for the report.

    Returns
    -------
    pd.DataFrame
        Summary with count and percentage of missing values per column.
    """
    missing = df.isnull().sum()
    pct = (missing / len(df)) * 100
    summary = pd.DataFrame({
        "missing_count": missing,
        "missing_pct": pct.round(2)
    })
    summary = summary[summary["missing_count"] > 0].sort_values(
        "missing_pct", ascending=False
    )

    print(f"\n{'=' * 50}")
    print(f"Missing Value Report — {dataset_name}")
    print(f"{'=' * 50}")
    print(f"Total rows: {len(df)}")
    if len(summary) == 0:
        print("No missing values found! ✓")
    else:
        print(summary.to_string())
    print(f"{'=' * 50}\n")

    return summary


def handle_missing_values(df, method="interpolate", limit=6):
    """
    Handle missing values using interpolation (best for time series).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with datetime index.
    method : str
        Method to use: 'interpolate', 'ffill', 'bfill', 'drop'.
    limit : int
        Maximum number of consecutive NaNs to fill (for interpolate/ffill/bfill).

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame.
    """
    df_clean = df.copy()

    # Ensure all numeric-looking columns are cast to numeric dtype
    for col in df_clean.columns:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

    if method == "interpolate":
        # Linear interpolation — ideal for continuous time series
        # Only interpolate numeric columns to avoid TypeError on object dtypes
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        df_clean[numeric_cols] = df_clean[numeric_cols].interpolate(
            method="linear", limit=limit, limit_direction="both"
        )
    elif method == "ffill":
        df_clean = df_clean.ffill(limit=limit)
    elif method == "bfill":
        df_clean = df_clean.bfill(limit=limit)
    elif method == "drop":
        df_clean = df_clean.dropna()
    else:
        raise ValueError(f"Unknown method: {method}")

    # Drop any remaining NaNs at the edges
    df_clean = df_clean.dropna()

    return df_clean


def remove_outliers_iqr(df, columns=None, factor=3.0):
    """
    Cap extreme outliers using the IQR method (winsorization).

    Instead of removing rows (which breaks time series continuity),
    we cap values at Q1 - factor*IQR and Q3 + factor*IQR.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    columns : list of str, optional
        Columns to check. If None, applies to all numeric columns.
    factor : float
        IQR multiplier (3.0 = very conservative, only extreme outliers).

    Returns
    -------
    pd.DataFrame
        DataFrame with outliers capped.
    """
    df_clean = df.copy()

    if columns is None:
        columns = df_clean.select_dtypes(include=[np.number]).columns.tolist()

    outlier_report = {}
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR

        # Count outliers before capping
        n_lower = (df_clean[col] < lower).sum()
        n_upper = (df_clean[col] > upper).sum()

        if n_lower + n_upper > 0:
            outlier_report[col] = {
                "lower_bound": round(lower, 4),
                "upper_bound": round(upper, 4),
                "n_lower_outliers": int(n_lower),
                "n_upper_outliers": int(n_upper),
            }

        # Cap (winsorize) — do NOT clip values that shouldn't be negative below 0
        real_lower = max(lower, 0) if df_clean[col].min() >= 0 else lower
        df_clean[col] = df_clean[col].clip(lower=real_lower, upper=upper)

    if outlier_report:
        print("\n[Outlier Report]")
        for col, info in outlier_report.items():
            print(f"  {col}: {info['n_lower_outliers']} low, "
                  f"{info['n_upper_outliers']} high "
                  f"(bounds: [{info['lower_bound']}, {info['upper_bound']}])")

    return df_clean


def validate_data_quality(df, dataset_name="Dataset"):
    """
    Run basic data quality checks and print results.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    dataset_name : str
        Name label for the report.

    Returns
    -------
    dict
        Quality check results.
    """
    results = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "duplicated_rows": int(df.duplicated().sum()),
        "missing_values": int(df.isnull().sum().sum()),
        "negative_values": {},
    }

    # Check for negative concentrations (should not exist for pollutants)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        n_neg = (df[col] < 0).sum()
        if n_neg > 0:
            results["negative_values"][col] = int(n_neg)

    print(f"\n{'=' * 50}")
    print(f"Data Quality Report — {dataset_name}")
    print(f"{'=' * 50}")
    print(f"Rows: {results['total_rows']}")
    print(f"Columns: {results['total_columns']}")
    print(f"Duplicates: {results['duplicated_rows']}")
    print(f"Missing: {results['missing_values']}")
    if results["negative_values"]:
        print(f"Negative values: {results['negative_values']}")
    else:
        print("Negative values: None ✓")
    print(f"{'=' * 50}\n")

    return results
