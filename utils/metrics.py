"""
Metrics for evaluating clause classifiers.
"""

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


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
    def from_predictions(cls, y_true: list[int], y_pred: list[int]) -> "ClassificationMetrics":
        """Calculate metrics from true and predicted labels."""
        pairs = list(zip(y_true, y_pred, strict=True))
        return cls(
            precision=precision_score(y_true, y_pred, zero_division=0),
            recall=recall_score(y_true, y_pred, zero_division=0),
            f1=f1_score(y_true, y_pred, zero_division=0),
            accuracy=accuracy_score(y_true, y_pred),
            true_positives=sum(1 for t, p in pairs if t == 1 and p == 1),
            false_positives=sum(1 for t, p in pairs if t == 0 and p == 1),
            true_negatives=sum(1 for t, p in pairs if t == 0 and p == 0),
            false_negatives=sum(1 for t, p in pairs if t == 1 and p == 0),
        )


@dataclass
class InferenceStats:
    """Latency and cost measured per document.

    Cost fields are None when the method has no price, which is reported as
    not priced rather than as zero.
    """

    total_latency_ms: float
    avg_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    total_cost_usd: float | None
    avg_cost_usd: float | None
    input_tokens: int
    output_tokens: int
    documents: int = 0


def summarise_documents(
    latencies_ms: list[float],
    costs_usd: list[float] | None = None,
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> InferenceStats:
    """Build InferenceStats from one measured latency, and cost, per document."""
    priced = costs_usd is not None
    if not latencies_ms:
        zero = 0.0 if priced else None
        return InferenceStats(0, 0, 0, 0, zero, zero, input_tokens, output_tokens, 0)
    return InferenceStats(
        total_latency_ms=float(sum(latencies_ms)),
        avg_latency_ms=float(np.mean(latencies_ms)),
        min_latency_ms=float(min(latencies_ms)),
        max_latency_ms=float(max(latencies_ms)),
        total_cost_usd=float(sum(costs_usd)) if priced else None,
        avg_cost_usd=float(np.mean(costs_usd)) if priced else None,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        documents=len(latencies_ms),
    )


def calculate_metrics(y_true: list[int], y_pred: list[int]) -> ClassificationMetrics:
    """Precision, recall, F1, accuracy, and confusion counts for one clause type."""
    return ClassificationMetrics.from_predictions(y_true, y_pred)


def aggregate_metrics(all_metrics: dict[str, ClassificationMetrics]) -> dict[str, float]:
    """Mean of each metric across clause types, with the lowest and highest precision."""
    if not all_metrics:
        return {}
    metrics = list(all_metrics.values())
    return {
        "avg_precision": float(np.mean([m.precision for m in metrics])),
        "avg_recall": float(np.mean([m.recall for m in metrics])),
        "avg_f1": float(np.mean([m.f1 for m in metrics])),
        "avg_accuracy": float(np.mean([m.accuracy for m in metrics])),
        "min_precision": min(m.precision for m in metrics),
        "max_precision": max(m.precision for m in metrics),
    }
