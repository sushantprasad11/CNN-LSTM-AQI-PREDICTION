"""
Scaler Module — Z-Score Normalization
======================================
Applies Z-score (StandardScaler) normalization to feature columns,
preserving the target variable (AQI code) unchanged.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os


class AQIScaler:
    """
    Z-score normalizer for AQI prediction features.

    Wraps sklearn's StandardScaler and provides methods specifically
    tailored for the AQI prediction pipeline:
    - Fits on training data only (to avoid data leakage)
    - Transforms both train and test sets
    - Saves/loads fitted scaler for reproducibility
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.is_fitted = False

    def fit(self, df, feature_columns):
        """
        Fit the scaler on training data.

        Parameters
        ----------
        df : pd.DataFrame
            Training DataFrame.
        feature_columns : list of str
            Column names to normalize (excludes target columns).

        Returns
        -------
        self
        """
        self.feature_columns = feature_columns
        self.scaler.fit(df[feature_columns])
        self.is_fitted = True
        print(f"[Scaler] Fitted on {len(feature_columns)} features, "
              f"{len(df)} samples")
        return self

    def transform(self, df):
        """
        Transform feature columns using the fitted scaler.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to transform.

        Returns
        -------
        pd.DataFrame
            Transformed DataFrame (non-feature columns preserved).
        """
        if not self.is_fitted:
            raise RuntimeError("Scaler not fitted. Call fit() first.")

        df_out = df.copy()
        df_out[self.feature_columns] = self.scaler.transform(df[self.feature_columns])
        return df_out

    def fit_transform(self, df, feature_columns):
        """Fit and transform in a single step."""
        self.fit(df, feature_columns)
        return self.transform(df)

    def inverse_transform(self, df):
        """
        Inverse transform to get original scale values.

        Parameters
        ----------
        df : pd.DataFrame
            Scaled DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with feature columns in original scale.
        """
        if not self.is_fitted:
            raise RuntimeError("Scaler not fitted. Call fit() first.")

        df_out = df.copy()
        df_out[self.feature_columns] = self.scaler.inverse_transform(
            df[self.feature_columns]
        )
        return df_out

    def save(self, filepath):
        """
        Save the fitted scaler to disk.

        Parameters
        ----------
        filepath : str
            Path to save the scaler (e.g., 'artifacts/scaler.joblib').
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        save_data = {
            "scaler": self.scaler,
            "feature_columns": self.feature_columns,
        }
        joblib.dump(save_data, filepath)
        print(f"[Scaler] Saved to: {filepath}")

    def load(self, filepath):
        """
        Load a previously saved scaler from disk.

        Parameters
        ----------
        filepath : str
            Path to the saved scaler file.

        Returns
        -------
        self
        """
        save_data = joblib.load(filepath)
        self.scaler = save_data["scaler"]
        self.feature_columns = save_data["feature_columns"]
        self.is_fitted = True
        print(f"[Scaler] Loaded from: {filepath}")
        return self

    def get_stats(self):
        """
        Get the mean and standard deviation used for scaling.

        Returns
        -------
        pd.DataFrame
            DataFrame with 'mean' and 'std' for each feature.
        """
        if not self.is_fitted:
            raise RuntimeError("Scaler not fitted.")

        return pd.DataFrame({
            "feature": self.feature_columns,
            "mean": self.scaler.mean_,
            "std": self.scaler.scale_,
        }).set_index("feature")
