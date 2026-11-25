"""
Tests for AQI Calculator Module
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import numpy as np
import pandas as pd
from src.preprocessing.aqi_calculator import (
    compute_sub_index, compute_aqi, get_aqi_category,
    BREAKPOINTS, AQI_CATEGORIES
)


class TestSubIndex(unittest.TestCase):
    """Test sub-index computation."""

    def test_pm25_good(self):
        """PM2.5 = 15 µg/m³ should give sub-index in Good range."""
        si = compute_sub_index(15, BREAKPOINTS["pm2_5"])
        self.assertGreaterEqual(si, 0)
        self.assertLessEqual(si, 50)

    def test_pm25_severe(self):
        """PM2.5 = 300 µg/m³ should give sub-index in Severe range."""
        si = compute_sub_index(300, BREAKPOINTS["pm2_5"])
        self.assertGreater(si, 400)

    def test_negative_concentration(self):
        """Negative concentration should return NaN."""
        si = compute_sub_index(-10, BREAKPOINTS["pm2_5"])
        self.assertTrue(np.isnan(si))

    def test_nan_concentration(self):
        """NaN concentration should return NaN."""
        si = compute_sub_index(np.nan, BREAKPOINTS["pm2_5"])
        self.assertTrue(np.isnan(si))

    def test_boundary_value(self):
        """Test exact breakpoint boundary."""
        si = compute_sub_index(30, BREAKPOINTS["pm2_5"])
        self.assertAlmostEqual(si, 50.0, places=1)


class TestAQICategory(unittest.TestCase):
    """Test AQI category mapping."""

    def test_good_category(self):
        label, code = get_aqi_category(25)
        self.assertEqual(label, "Good")
        self.assertEqual(code, 0)

    def test_severe_category(self):
        label, code = get_aqi_category(450)
        self.assertEqual(label, "Severe")
        self.assertEqual(code, 5)

    def test_nan_value(self):
        label, code = get_aqi_category(np.nan)
        self.assertEqual(label, "Unknown")
        self.assertEqual(code, -1)

    def test_above_500(self):
        label, code = get_aqi_category(600)
        self.assertEqual(label, "Severe")
        self.assertEqual(code, 5)


class TestComputeAQI(unittest.TestCase):
    """Test full AQI computation from a row."""

    def test_basic_aqi(self):
        row = pd.Series({
            "pm2_5": 15.0, "pm10": 40.0,
            "no2": 20.0, "so2": 10.0,
            "co": 500.0, "o3": 30.0  # CO in µg/m³
        })
        aqi = compute_aqi(row)
        self.assertIsInstance(aqi, float)
        self.assertGreater(aqi, 0)

    def test_all_nan(self):
        row = pd.Series({
            "pm2_5": np.nan, "pm10": np.nan,
            "no2": np.nan, "so2": np.nan,
            "co": np.nan, "o3": np.nan
        })
        aqi = compute_aqi(row)
        self.assertTrue(np.isnan(aqi))


if __name__ == '__main__':
    unittest.main()
