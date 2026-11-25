"""Evaluation subpackage."""

from src.evaluation.metrics import compute_metrics, get_classification_report, print_model_metrics
from src.evaluation.confusion_matrix import plot_confusion_matrix, plot_side_by_side_cm
from src.evaluation.ablation_analysis import run_ablation_study, plot_ablation_results
from src.evaluation.model_comparison import compare_models, plot_model_comparison
from src.evaluation.statistical_tests import mcnemar_test

__all__ = [
    "compute_metrics", "get_classification_report", "print_model_metrics",
    "plot_confusion_matrix", "plot_side_by_side_cm",
    "run_ablation_study", "plot_ablation_results",
    "compare_models", "plot_model_comparison",
    "mcnemar_test",
]
