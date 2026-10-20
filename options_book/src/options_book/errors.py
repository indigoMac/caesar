"""Failures the engine either skips or latches."""

from __future__ import annotations


class OptionsBookError(Exception):
    """Base error for this package."""


class HaltError(OptionsBookError):
    """The book must stop sending orders until an operator clears the latch."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


class OrdersDisabled(HaltError):
    """placeOrder was reached while submit_orders is false."""

    def __init__(self) -> None:
        super().__init__("submit_orders is false")


class SelectionError(OptionsBookError):
    """This snapshot has no order. The latch stays clear."""
