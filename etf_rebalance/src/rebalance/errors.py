"""Explicit failures for data-contract and look-ahead violations."""

from __future__ import annotations


class DataContractError(ValueError):
    """Raised when inputs violate the research data contract."""


class LookaheadError(ValueError):
    """Raised when a feature or fill uses information that was not yet public."""


class MissingPriceError(DataContractError):
    """Raised when a required raw price bar is absent. Prices are never filled."""
