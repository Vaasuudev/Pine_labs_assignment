"""Pricing rules. All money is integer cents."""


def line_total(unit_price_cents: int, quantity: int) -> int:
    """Total for one cart line."""
    if unit_price_cents < 0 or quantity < 0:
        raise ValueError("unit price and quantity must be non-negative")
    return unit_price_cents * quantity


def order_total(lines):
    """Sum of line totals for (unit_price_cents, quantity) pairs."""
    return sum(line_total(price, qty) for price, qty in lines)
