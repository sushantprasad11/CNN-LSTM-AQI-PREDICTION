"""
Tests for Model Building
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import numpy as np

# Suppress TensorFlow warnings for cleaner test output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class TestCNNModel(unittest.TestCase):
    """Test CNN model building."""

    def test_build(self):
        from src.models.cnn import build_cnn_model
        model = build_cnn_model(input_shape=(24, 10), num_classes=6)
        self.assertEqual(model.output_shape, (None, 6))

    def test_predict_shape(self):
        from src.models.cnn import build_cnn_model
        model = build_cnn_model(input_shape=(24, 10), num_classes=6)
        X = np.random.rand(5, 24, 10).astype(np.float32)
        pred = model.predict(X, verbose=0)
        self.assertEqual(pred.shape, (5, 6))


class TestLSTMModel(unittest.TestCase):
    """Test LSTM model building."""

    def test_build(self):
        from src.models.lstm import build_lstm_model
        model = build_lstm_model(input_shape=(24, 10), num_classes=6)
        self.assertEqual(model.output_shape, (None, 6))


class TestCNNLSTMModel(unittest.TestCase):
    """Test Hybrid CNN-LSTM model building."""

    def test_build(self):
        from src.models.cnn_lstm import build_cnn_lstm_model
        model = build_cnn_lstm_model(input_shape=(24, 10), num_classes=6)
        self.assertEqual(model.output_shape, (None, 6))

    def test_predict_probabilities(self):
        from src.models.cnn_lstm import build_cnn_lstm_model
        model = build_cnn_lstm_model(input_shape=(24, 10), num_classes=6)
        X = np.random.rand(3, 24, 10).astype(np.float32)
        pred = model.predict(X, verbose=0)
        # Probabilities should sum to 1
        for row in pred:
            self.assertAlmostEqual(row.sum(), 1.0, places=5)


if __name__ == '__main__':
    unittest.main()
