"""
Tests for Preprocessing Modules
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import numpy as np
import pandas as pd
from src.preprocessing.feature_engineering import (
    create_lag_features, create_rolling_features,
    create_cyclic_features, create_ratio_features
)
from src.preprocessing.scaler import AQIScaler
from src.preprocessing.sequence_generator import create_sequences, split_sequences


class TestLagFeatures(unittest.TestCase):
    """Test lag feature creation."""

    def setUp(self):
        dates = pd.date_range("2024-01-01", periods=100, freq="h")
        self.df = pd.DataFrame({
            "pm2_5": np.random.rand(100) * 100,
            "pm10": np.random.rand(100) * 200,
        }, index=dates)

    def test_lag_columns_created(self):
        df_out, new_cols = create_lag_features(self.df, ["pm2_5"], [1, 2, 3])
        self.assertEqual(len(new_cols), 3)
        self.assertIn("pm2_5_lag_1h", df_out.columns)

    def test_lag_values_correct(self):
        df_out, _ = create_lag_features(self.df, ["pm2_5"], [1])
        # At index 1, lag_1h should equal value at index 0
        self.assertAlmostEqual(
            df_out["pm2_5_lag_1h"].iloc[1],
            self.df["pm2_5"].iloc[0],
            places=5
        )


class TestCyclicFeatures(unittest.TestCase):
    """Test cyclic temporal feature creation."""

    def setUp(self):
        dates = pd.date_range("2024-01-01", periods=48, freq="h")
        self.df = pd.DataFrame({"val": np.random.rand(48)}, index=dates)

    def test_cyclic_columns_exist(self):
        df_out, new_cols = create_cyclic_features(self.df)
        self.assertIn("hour_sin", df_out.columns)
        self.assertIn("hour_cos", df_out.columns)
        self.assertIn("day_sin", df_out.columns)
        self.assertEqual(len(new_cols), 8)

    def test_values_bounded(self):
        df_out, _ = create_cyclic_features(self.df)
        self.assertTrue((df_out["hour_sin"] >= -1).all())
        self.assertTrue((df_out["hour_sin"] <= 1).all())


class TestScaler(unittest.TestCase):
    """Test AQI scaler."""

    def test_fit_transform(self):
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [10, 20, 30, 40, 50]})
        scaler = AQIScaler()
        df_scaled = scaler.fit_transform(df, ["a", "b"])
        # Mean should be ~0 and std ~1
        self.assertAlmostEqual(df_scaled["a"].mean(), 0, places=5)
        self.assertAlmostEqual(df_scaled["a"].std(ddof=0), 1, places=5)


class TestSequences(unittest.TestCase):
    """Test sequence generation."""

    def test_sequence_shapes(self):
        features = np.random.rand(100, 10)
        targets = np.random.randint(0, 6, 100)
        X, y = create_sequences(features, targets, window_size=24)
        self.assertEqual(X.shape[0], 76)  # 100 - 24
        self.assertEqual(X.shape[1], 24)
        self.assertEqual(X.shape[2], 10)
        self.assertEqual(y.shape[0], 76)

    def test_split(self):
        X = np.random.rand(100, 24, 10)
        y = np.random.randint(0, 6, 100)
        X_tr, X_te, y_tr, y_te = split_sequences(X, y, test_size=0.3)
        self.assertEqual(len(X_tr), 70)
        self.assertEqual(len(X_te), 30)


if __name__ == '__main__':
    unittest.main()
