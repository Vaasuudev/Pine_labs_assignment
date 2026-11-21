import pytest

from shop.cart import Cart


def test_empty_cart_is_zero():
    assert Cart().total_cents() == 0


def test_cart_totals_lines():
    cart = Cart()
    cart.add("PEN-01", 2)
    cart.add("NBK-02")
    assert cart.total_cents() == 2 * 500 + 2500


def test_unknown_sku_rejected():
    with pytest.raises(KeyError):
        Cart().add("NOPE")


def test_non_positive_quantity_rejected():
    with pytest.raises(ValueError):
        Cart().add("PEN-01", 0)
