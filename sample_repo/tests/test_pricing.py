import pytest

from shop.pricing import line_total, order_total


def test_line_total_simple():
    assert line_total(250, 3) == 750


def test_line_total_zero_quantity():
    assert line_total(250, 0) == 0


def test_negative_inputs_rejected():
    with pytest.raises(ValueError):
        line_total(-1, 2)
    with pytest.raises(ValueError):
        line_total(1, -2)


def test_order_total_sums_lines():
    assert order_total([(100, 2), (50, 1)]) == 250
