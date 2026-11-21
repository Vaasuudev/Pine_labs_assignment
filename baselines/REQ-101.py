"""Held-out reference for REQ-101 (bulk discount). NEVER shown to any agent.

A drop-in replacement for shop/pricing.py. Uses exact fractions rather than the obvious integer
formulation, so it is an independent oracle. It is used twice:
  1. differential check: candidate vs this file over a grid of inputs
  2. test-validity check: the QA agent's tests must PASS against this file (a test that fails on
     the reference is a wrong test, not a wrong implementation)."""
import math
from fractions import Fraction

TARGET = "shop/pricing.py"
FUNCTION = "line_total"


def cases():
    for price in list(range(0, 400)) + [4999, 8999, 12345]:
        for qty in range(0, 31):
            yield (price, qty)


def line_total(unit_price_cents: int, quantity: int) -> int:
    if unit_price_cents < 0 or quantity < 0:
        raise ValueError("unit price and quantity must be non-negative")
    gross = unit_price_cents * quantity
    if quantity >= 10:
        return math.floor(Fraction(gross * 9, 10) + Fraction(1, 2))
    return gross


def order_total(lines):
    return sum(line_total(price, qty) for price, qty in lines)
