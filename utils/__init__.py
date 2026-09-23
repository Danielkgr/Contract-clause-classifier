"""Utility modules for Contract Clause Classifier.

Exports load lazily, so importing utils.data_loader or utils.llm_client does
not pull in torch and transformers through utils.classifier.
"""

import importlib

_EXPORTS = {
    "LLMClient": "llm_client",
    "load_cuad_dataset": "data_loader",
    "evaluate_zero_shot": "evaluation",
    "evaluate_fine_tuned": "evaluation",
    "calculate_metrics": "metrics",
    "calculate_cost": "metrics",
    "calculate_latency": "metrics",
    "FineTunedClassifier": "classifier",
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        module = importlib.import_module(f".{_EXPORTS[name]}", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
