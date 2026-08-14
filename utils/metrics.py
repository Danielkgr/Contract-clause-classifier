"""
Metrics calculation module for evaluating clause classifiers.
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    classification_report, confusion_matrix
)

from config import config


@dataclass
class ClassificationMetrics:
    """Classification metrics for a single clause type."""
    precision: float
    recall: float
    f1: float
    accuracy: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    
    @classmethod
    def from_predictions(
        cls, y_true: List[int], y_pred: List[int]
    ) -> "ClassificationMetrics":
        """Calculate metrics from true and predicted labels."""
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
        tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
        
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)
        
        return cls(
            precision=precision,
            recall=recall,
            f1=f1,
            accuracy=accuracy,
            true_positives=tp,
            false_positives=fp,
            true_negatives=tn,
            false_negatives=fn
        )


@dataclass
class InferenceStats:
    """Statistics for inference on multiple samples."""
    total_latency_ms: float
    avg_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    total_cost_usd: float
    avg_cost_usd: float
    input_tokens: int
    output_tokens: int


def calculate_metrics(
    y_true: List[int],
    y_pred: List[int],
    clause_type: Optional[str] = None
) -> ClassificationMetrics:
    """Calculate classification metrics.
    
    Args:
        y_true: Ground truth labels (0 or 1)
        y_pred: Predicted labels (0 or 1)
        clause_type: Optional clause type name for logging
        
    Returns:
        ClassificationMetrics object
    """
    metrics = ClassificationMetrics.from_predictions(y_true, y_pred)
    
    if clause_type:
        print(f"\n{clause_type}:")
        print(f"  Precision: {metrics.precision:.4f}")
        print(f"  Recall: {metrics.recall:.4f}")
        print(f"  F1 Score: {metrics.f1:.4f}")
        print(f"  Accuracy: {metrics.accuracy:.4f}")
        print(f"  TP: {metrics.true_positives}, FP: {metrics.false_positives}")
        print(f"  TN: {metrics.true_negatives}, FN: {metrics.false_negatives}")
    
    return metrics


def calculate_cost(
    input_tokens: int,
    output_tokens: int,
    model: Optional[str] = None
) -> float:
    """Calculate API cost for LLM inference.
    
    Args:
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        model: Model name (uses config if None)
        
    Returns:
        Cost in USD
    """
    model = model or config.llm.model
    cost_per_million = config.llm.get_cost_per_million()
    
    input_cost = (input_tokens / 1_000_000) * cost_per_million[0]
    output_cost = (output_tokens / 1_000_000) * cost_per_million[1]
    
    return input_cost + output_cost


def calculate_latency(
    start_times: List[float],
    end_times: List[float]
) -> InferenceStats:
    """Calculate latency statistics.
    
    Args:
        start_times: List of start timestamps
        end_times: List of end timestamps
        
    Returns:
        InferenceStats object
    """
    latencies = [(e - s) * 1000 for s, e in zip(start_times, end_times)]
    
    return InferenceStats(
        total_latency_ms=sum(latencies),
        avg_latency_ms=np.mean(latencies),
        min_latency_ms=min(latencies),
        max_latency_ms=max(latencies),
        total_cost_usd=0,  # Will be calculated separately
        avg_cost_usd=0,
        input_tokens=0,
        output_tokens=0
    )


def aggregate_metrics(
    all_metrics: Dict[str, ClassificationMetrics]
) -> Dict[str, float]:
    """Aggregate metrics across all clause types.
    
    Args:
        all_metrics: Dict mapping clause type to metrics
        
    Returns:
        Dict with aggregate metrics
    """
    if not all_metrics:
        return {}
    
    metrics_list = list(all_metrics.values())
    
    return {
        "avg_precision": np.mean([m.precision for m in metrics_list]),
        "avg_recall": np.mean([m.recall for m in metrics_list]),
        "avg_f1": np.mean([m.f1 for m in metrics_list]),
        "avg_accuracy": np.mean([m.accuracy for m in metrics_list]),
        "min_precision": min([m.precision for m in metrics_list]),
        "max_precision": max([m.precision for m in metrics_list]),
    }


def get_classification_report(
    y_true: List[int],
    y_pred: List[int],
    clause_type: str
) -> str:
    """Generate a classification report string.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        clause_type: Clause type name
        
    Returns:
        Formatted report string
    """
    return classification_report(
        y_true, y_pred,
        target_names=[f"No {clause_type}", f"{clause_type}"],
        digits=4
    )


def print_comparison_table(
    zero_shot_metrics: Dict[str, ClassificationMetrics],
    fine_tuned_metrics: Dict[str, ClassificationMetrics]
):
    """Print a comparison table between two classifiers.
    
    Args:
        zero_shot_metrics: Metrics for zero-shot LLM
        fine_tuned_metrics: Metrics for fine-tuned model
    """
    print("\n" + "=" * 80)
    print("CLASSIFIER COMPARISON")
    print("=" * 80)
    
    clause_types = list(zero_shot_metrics.keys())
    if not clause_types:
        clause_types = list(fine_tuned_metrics.keys())
    
    # Header
    print(f"{'Clause Type':<25} {'Method':<15} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Acc':>10}")
    print("-" * 80)
    
    # Data rows
    for clause_type in clause_types:
        zs = zero_shot_metrics.get(clause_type)
        ft = fine_tuned_metrics.get(clause_type)
        
        if zs:
            print(f"{clause_type:<25} {'Zero-Shot':<15} "
                  f"{zs.precision:>10.4f} {zs.recall:>10.4f} {zs.f1:>10.4f} {zs.accuracy:>10.4f}")
        
        if ft:
            print(f"{clause_type:<25} {'Fine-Tuned':<15} "
                  f"{ft.precision:>10.4f} {ft.recall:>10.4f} {ft.f1:>10.4f} {ft.accuracy:>10.4f}")
    
    print("=" * 80)
