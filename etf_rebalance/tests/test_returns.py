"""Total return, split, and dividend conventions."""

from __future__ import annotations

import pandas as pd
import pytest

from rebalance.features.returns import (
    close_to_close_total_return,
    holding_period_total_return,
    index_prices,
    open_to_close_return,
)
from tests.conftest import price_row


def test_close_to_close_split_and_dividend() -> None:
    assert close_to_close_total_return(100, 50, 2, 0) == 0
    assert close_to_close_total_return(100, 99, 1, 1) == 0
    assert close_to_close_total_return(100, 49, 2, 1) == 0


def test_open_to_close_return() -> None:
    assert open_to_close_return(100, 110) == pytest.approx(0.1)


def test_buyer_at_the_open_does_not_receive_that_days_dividend(xcal) -> None:
    frame = index_prices(
        pd.DataFrame([price_row("AAA", "2024-06-10", 100, 100, dividend=5)])
    )["AAA"]
    assert holding_period_total_return(frame, pd.Timestamp("2024-06-10"), pd.Timestamp("2024-06-10"), xcal) == 0


def test_overnight_holder_receives_next_session_dividend(xcal) -> None:
    frame = index_prices(
        pd.DataFrame(
            [
                price_row("AAA", "2024-06-10", 100, 100, dividend=0),
                price_row("AAA", "2024-06-11", 95, 95, dividend=5),
            ]
        )
    )["AAA"]
    result = holding_period_total_return(frame, pd.Timestamp("2024-06-10"), pd.Timestamp("2024-06-11"), xcal)
    assert result == 0


def test_split_holding_period_is_flat(xcal) -> None:
    frame = index_prices(
        pd.DataFrame(
            [
                price_row("AAA", "2024-06-10", 100, 100, split_factor=1),
                price_row("AAA", "2024-06-11", 50, 50, split_factor=2, volume=2_000_000),
            ]
        )
    )["AAA"]
    result = holding_period_total_return(frame, pd.Timestamp("2024-06-10"), pd.Timestamp("2024-06-11"), xcal)
    assert result == 0
