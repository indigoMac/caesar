"""Shared fixtures for calendar-aware research tests."""

from __future__ import annotations

import pandas as pd
import pytest

from rebalance.calendar import TradingCalendar


@pytest.fixture(scope="session")
def xcal() -> TradingCalendar:
    return TradingCalendar("XNYS", "America/New_York")


def et(value: str) -> pd.Timestamp:
    """Parse a New York clock time and return UTC."""
    return pd.Timestamp(value, tz="America/New_York").tz_convert("UTC")


def price_row(
    security_id: str,
    day: str,
    open_price: float,
    close: float,
    *,
    volume: float = 1_000_000.0,
    split_factor: float = 1.0,
    dividend: float = 0.0,
) -> dict[str, object]:
    return {
        "security_id": security_id,
        "date": pd.Timestamp(day),
        "open_raw": open_price,
        "high_raw": max(open_price, close) + 0.01,
        "low_raw": min(open_price, close) - 0.01,
        "close_raw": close,
        "volume_raw": volume,
        "split_factor": split_factor,
        "dividend": dividend,
    }


def flat_prices(security_id: str, sessions: pd.DatetimeIndex, price: float) -> pd.DataFrame:
    return pd.DataFrame(
        [
            price_row(security_id, pd.Timestamp(session).strftime("%Y-%m-%d"), price, price)
            for session in sessions
        ]
    )
