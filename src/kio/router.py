"""Shared KIO router: models are injected and must predict 0 or 1."""

from collections.abc import Callable

from . import SALES, SUPPORT


def support_agent() -> None:
    print("Support Agent")


def sales_agent() -> None:
    print("Sales Agent")


def route_message(message: str, predict: Callable[[str], int]) -> int:
    """Predict the route and invoke the corresponding simulated agent.

    A failed model or invalid prediction raises an exception instead of
    silently turning it into an incorrect class label.
    """
    if not isinstance(message, str) or not message.strip():
        raise ValueError("message must be a nonempty string")
    prediction = predict(message)
    if type(prediction) is not int or prediction not in (SUPPORT, SALES):
        raise ValueError("model must return integer 0 (support) or 1 (sales)")
    (support_agent if prediction == SUPPORT else sales_agent)()
    return prediction
