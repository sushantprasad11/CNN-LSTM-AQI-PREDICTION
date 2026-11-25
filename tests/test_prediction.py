"""
Tests for Prediction & Early Warning Modules
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import numpy as np
from src.prediction.early_warning import (
    generate_alerts, evaluate_early_warning, format_alert_message
)
from src.prediction.health_risk import map_health_risk, HEALTH_RISK_MAP


class TestEarlyWarning(unittest.TestCase):
    """Test early warning alert generation."""

    def test_alert_generation(self):
        categories = np.array([0, 1, 2, 3, 4, 5, 4, 3, 2])
        alerts = generate_alerts(categories)
        # Categories 4 and 5 should trigger alerts
        expected = np.array([0, 0, 0, 0, 1, 1, 1, 0, 0])
        np.testing.assert_array_equal(alerts, expected)

    def test_no_alerts(self):
        categories = np.array([0, 1, 2, 3])
        alerts = generate_alerts(categories)
        self.assertEqual(alerts.sum(), 0)

    def test_all_alerts(self):
        categories = np.array([4, 5, 4, 5])
        alerts = generate_alerts(categories)
        self.assertEqual(alerts.sum(), 4)

    def test_evaluate_metrics(self):
        true_cats = np.array([0, 1, 4, 5, 3, 4])
        pred_cats = np.array([0, 1, 4, 4, 3, 3])
        metrics = evaluate_early_warning(true_cats, pred_cats)
        self.assertIn('alert_precision', metrics)
        self.assertIn('alert_recall', metrics)
        self.assertIn('alert_f1_score', metrics)


class TestAlertMessage(unittest.TestCase):
    """Test alert message formatting."""

    def test_severe_alert(self):
        alert = format_alert_message(5)
        self.assertIsNotNone(alert)
        self.assertEqual(alert['level'], 'CRITICAL')
        self.assertEqual(alert['category'], 'Severe')

    def test_very_poor_alert(self):
        alert = format_alert_message(4)
        self.assertIsNotNone(alert)
        self.assertEqual(alert['level'], 'WARNING')

    def test_no_alert_for_good(self):
        alert = format_alert_message(0)
        self.assertIsNone(alert)

    def test_no_alert_for_moderate(self):
        alert = format_alert_message(2)
        self.assertIsNone(alert)


class TestHealthRisk(unittest.TestCase):
    """Test health risk mapping."""

    def test_all_categories_mapped(self):
        for code in range(6):
            risk = map_health_risk(code)
            self.assertIn('risk_level', risk)
            self.assertIn('risk_score', risk)
            self.assertNotEqual(risk['risk_level'], 'Unknown')

    def test_risk_score_ordering(self):
        scores = [map_health_risk(i)['risk_score'] for i in range(6)]
        self.assertEqual(scores, sorted(scores))

    def test_unknown_category(self):
        risk = map_health_risk(99)
        self.assertEqual(risk['risk_level'], 'Unknown')


if __name__ == '__main__':
    unittest.main()
