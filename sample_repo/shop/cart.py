"""Shopping cart."""
from .catalog import PRODUCTS
from .pricing import line_total


class Cart:
    def __init__(self):
        self._items = {}

    def add(self, sku, quantity=1):
        if sku not in PRODUCTS:
            raise KeyError(f"unknown sku: {sku}")
        if quantity <= 0:
            raise ValueError("quantity must be positive")
        self._items[sku] = self._items.get(sku, 0) + quantity

    def total_cents(self):
        return sum(
            line_total(PRODUCTS[sku]["price_cents"], qty)
            for sku, qty in self._items.items()
        )
