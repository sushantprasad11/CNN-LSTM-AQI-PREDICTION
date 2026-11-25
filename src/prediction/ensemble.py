"""
Ensemble Predictor — Maximum Accuracy via Multi-Model Fusion
================================================================
Combines predictions from all 3 trained models (CNN, LSTM, Hybrid CNN-LSTM)
to produce more accurate and robust AQI category predictions.

Ensemble Strategy:
  - Weighted soft-voting: Average class probabilities from all models,
    weighting by model quality (Hybrid CNN-LSTM gets highest weight).
  - Confidence calibration: Aggregated confidence is more reliable than
    any single model's confidence.

This approach reduces variance and exploits the complementary strengths
of each architecture:
  - CNN:  captures local spatial patterns in the feature space
  - LSTM: captures long-range temporal dependencies
  - Hybrid: combines both — best individual model

When all 3 models agree, confidence is very high.
When they disagree, the ensemble picks the weighted majority view,
and the confidence is naturally lower — a built-in uncertainty signal.
"""

import numpy as np
import os

# Suppress TF noise
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf

AQI_LABELS = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]

# Default ensemble weights — Hybrid gets highest because it typically
# has the best F1/accuracy.  These can be overridden.
DEFAULT_WEIGHTS = {
    "cnn": 0.20,
    "lstm": 0.25,
    "cnn_lstm": 0.55,
}


class EnsemblePredictor:
    """
    Multi-model ensemble predictor for maximum AQI forecast accuracy.

    Loads CNN, LSTM, and Hybrid CNN-LSTM models and combines their
    probability outputs using weighted soft-voting.
    """

    def __init__(self, models_dir="models", weights=None):
        """
        Parameters
        ----------
        models_dir : str
            Base directory containing model subdirectories (cnn/, lstm/, cnn_lstm/).
        weights : dict, optional
            Custom weights for each model. Keys: 'cnn', 'lstm', 'cnn_lstm'.
            Values should sum to 1.0. If None, uses DEFAULT_WEIGHTS.
        """
        self.models_dir = models_dir
        self.weights = weights or DEFAULT_WEIGHTS.copy()
        self.models = {}
        self.loaded = False

    def load(self):
        """Load all available models from disk."""
        model_specs = {
            "cnn":      os.path.join(self.models_dir, "cnn",      "best_model.keras"),
            "lstm":     os.path.join(self.models_dir, "lstm",     "best_model.keras"),
            "cnn_lstm": os.path.join(self.models_dir, "cnn_lstm", "best_model.keras"),
        }

        for name, path in model_specs.items():
            if os.path.exists(path):
                try:
                    self.models[name] = tf.keras.models.load_model(path)
                    print(f"  [Ensemble] Loaded {name:10s} from {path}")
                except Exception as e:
                    print(f"  [Ensemble] Failed to load {name}: {e}")
            else:
                print(f"  [Ensemble] Not found: {path}")

        if not self.models:
            raise RuntimeError("No models could be loaded for ensemble.")

        # Renormalize weights to only include loaded models
        total = sum(self.weights.get(name, 0) for name in self.models)
        if total > 0:
            self.weights = {
                name: self.weights.get(name, 0) / total
                for name in self.models
            }

        self.loaded = True
        print(f"  [Ensemble] Active models: {list(self.models.keys())}")
        print(f"  [Ensemble] Weights: {self.weights}")
        return self

    def predict(self, X, return_individual=False):
        """
        Generate ensemble predictions for input sequences.

        Parameters
        ----------
        X : np.ndarray
            Input array of shape (n_samples, timesteps, features).
        return_individual : bool
            If True, also return each model's individual probabilities.

        Returns
        -------
        dict with keys:
            'predicted_codes' : np.ndarray of ints (n_samples,)
            'predicted_labels': list of str
            'probabilities'   : np.ndarray (n_samples, 6)  — ensemble probs
            'confidences'     : np.ndarray (n_samples,)     — max prob per sample
            'agreement'       : np.ndarray (n_samples,)     — fraction of models
                                agreeing on the top class (0..1)
            'individual'      : dict (only if return_individual=True)
                model_name -> probabilities array
        """
        if not self.loaded:
            self.load()

        individual_probs = {}
        individual_preds = {}

        for name, model in self.models.items():
            probs = model.predict(X, verbose=0)
            individual_probs[name] = probs
            individual_preds[name] = np.argmax(probs, axis=1)

        # Weighted soft-voting
        ensemble_probs = np.zeros_like(list(individual_probs.values())[0])
        for name, probs in individual_probs.items():
            ensemble_probs += self.weights[name] * probs

        predicted_codes = np.argmax(ensemble_probs, axis=1)
        confidences = np.max(ensemble_probs, axis=1)
        predicted_labels = [AQI_LABELS[c] for c in predicted_codes]

        # Compute model agreement — how many models agree on the top class
        pred_matrix = np.column_stack(list(individual_preds.values()))
        agreement = np.array([
            np.mean(pred_matrix[i] == predicted_codes[i])
            for i in range(len(predicted_codes))
        ])

        result = {
            "predicted_codes": predicted_codes,
            "predicted_labels": predicted_labels,
            "probabilities": ensemble_probs,
            "confidences": confidences,
            "agreement": agreement,
        }

        if return_individual:
            result["individual"] = individual_probs

        return result

    def predict_single(self, X_single):
        """
        Predict a single sample with detailed breakdown.

        Parameters
        ----------
        X_single : np.ndarray
            Shape (1, timesteps, features) or (timesteps, features).

        Returns
        -------
        dict
            Detailed prediction result for the single sample.
        """
        if X_single.ndim == 2:
            X_single = X_single[np.newaxis, ...]

        result = self.predict(X_single, return_individual=True)

        return {
            "predicted_code": int(result["predicted_codes"][0]),
            "predicted_label": result["predicted_labels"][0],
            "confidence": float(result["confidences"][0]),
            "agreement": float(result["agreement"][0]),
            "ensemble_probs": {
                AQI_LABELS[i]: float(result["probabilities"][0][i])
                for i in range(len(AQI_LABELS))
            },
            "individual_predictions": {
                name: {
                    "predicted": AQI_LABELS[np.argmax(probs[0])],
                    "confidence": float(np.max(probs[0])),
                    "probs": {
                        AQI_LABELS[i]: float(probs[0][i])
                        for i in range(len(AQI_LABELS))
                    },
                }
                for name, probs in result["individual"].items()
            },
        }
