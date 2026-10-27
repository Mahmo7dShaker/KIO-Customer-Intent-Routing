import pytest

from kio.contracts import normalize_decision_label
from kio.router import route_message


def test_support_route(capsys):
    assert route_message("My service is down", lambda _: 0) == 0
    assert capsys.readouterr().out == "Support Agent\n"


def test_sales_route(capsys):
    assert route_message("I want a quote", lambda _: 1) == 1
    assert capsys.readouterr().out == "Sales Agent\n"


@pytest.mark.parametrize("prediction", [None, -1, 2, True, "0", 1.0])
def test_invalid_prediction_is_rejected(prediction, capsys):
    with pytest.raises(ValueError):
        route_message("Test", lambda _: prediction)
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("message", ["", "   ", None])
def test_invalid_message_is_rejected(message):
    with pytest.raises(ValueError):
        route_message(message, lambda _: 0)


def test_decision_labels():
    assert normalize_decision_label("customer_service") == 0
    assert normalize_decision_label("sales") == 1
    with pytest.raises(ValueError):
        normalize_decision_label("support")
