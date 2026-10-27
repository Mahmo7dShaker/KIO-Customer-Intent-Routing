"""Model adapters must implement these signatures."""

from typing import Protocol

from . import SALES, SUPPORT


class Predictor(Protocol):
    def __call__(self, message: str) -> int: ...


def normalize_decision_label(label: str) -> int:
    """Map *validated* decision-service categories to router flags."""
    mapping = {"customer_service": SUPPORT, "sales": SALES}
    if not isinstance(label, str) or label not in mapping:
        raise ValueError("decision label must be 'customer_service' or 'sales'")
    return mapping[label]
