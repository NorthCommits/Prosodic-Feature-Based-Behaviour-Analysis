"""Evaluation and metrics modules."""

from .evaluator import Evaluator

try:
    from .interpretability import InterpretabilityAnalyzer
    __all__ = ["Evaluator", "InterpretabilityAnalyzer"]
except ImportError:
    __all__ = ["Evaluator"]

