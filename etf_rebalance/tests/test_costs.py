"""Flat cost scenarios and buy-and-hold portfolio netting."""

from __future__ import annotations

import pandas as pd
import pytest

from rebalance.config import load_cost_config, repo_root
from rebalance.portfolio.costs import CostContext, FlatBpsCostModel, bps_to_fraction, net_return
from rebalance.portfolio.simulator import simulate_portfolio
from rebalance.portfolio.sizing import size_positions
from tests.conftest import et, price_row


def test_cost_scenarios_and_netting() -> None:
    costs = load_cost_config(repo_root() / "config" / "costs.yaml")
    assert costs.scenarios_bps == {"optimistic": 5, "base": 15, "conservative": 30, "hostile": 50}
    assert bps_to_fraction(15) == pytest.approx(0.0015)
    assert net_return(0.02, entry_cost=0.0015, exit_cost=0.0015) == pytest.approx(0.017)
    model = FlatBpsCostModel(bps=50, name="hostile")
    assert model.one_way_cost(CostContext(security_id="AAA", leg="entry")) == pytest.approx(0.005)


def test_equal_weight_cap_and_liquidity_cap() -> None:
    names = pd.DataFrame(
        {
            "security_id": ["A", "B"],
            "action": ["ADD", "ADD"],
            "pressure_score": [1.0, 3.0],
            "adv20": [1_000_000.0, 1_000_000.0],
        }
    )
    capped = size_positions(names, mode="long_additions", scheme="equal", max_name_weight=0.4, gross_exposure_cap=1.0)
    assert capped.weights.tolist() == pytest.approx([0.4, 0.4])
    assert capped.achieved_gross == pytest.approx(0.8)
    pressured = size_positions(names, mode="long_additions", scheme="pressure", max_name_weight=1.0, gross_exposure_cap=1.0)
    assert pressured.weights.tolist() == pytest.approx([0.25, 0.75])
    limited = size_positions(
        names,
        mode="long_additions",
        scheme="equal",
        max_name_weight=0.5,
        gross_exposure_cap=1.0,
        portfolio_nav=1_000_000.0,
        max_adv_participation=0.01,
    )
    assert limited.weights.tolist() == pytest.approx([0.01, 0.01])
    assert limited.liquidity_constraint == "active"


def _book(xcal) -> tuple[pd.DataFrame, pd.DataFrame]:
    known = et("2024-06-07 17:00")
    earliest = xcal.earliest_regular_open_after(known)
    entry = xcal.entry_session(earliest)
    signals = pd.DataFrame(
        [
            {
                "rebalance_id": "2024-06",
                "security_id": "AAA",
                "action": "ADD",
                "signal_known_at": known,
                "feature_asof": known,
                "earliest_trade_at": earliest,
                "entry_session": entry,
                "effective_close": pd.Timestamp("2024-06-12"),
                "pressure_score": 1.0,
                "adv20": 1_000_000.0,
            },
            {
                "rebalance_id": "2024-06",
                "security_id": "DDD",
                "action": "DELETE",
                "signal_known_at": known,
                "feature_asof": known,
                "earliest_trade_at": earliest,
                "entry_session": entry,
                "effective_close": pd.Timestamp("2024-06-12"),
                "pressure_score": 1.0,
                "adv20": 1_000_000.0,
            },
        ]
    )
    prices = pd.DataFrame(
        [
            price_row("AAA", "2024-06-10", 100, 110),
            price_row("AAA", "2024-06-11", 110, 121),
            price_row("AAA", "2024-06-12", 121, 121),
            price_row("DDD", "2024-06-10", 200, 180),
            price_row("DDD", "2024-06-11", 180, 162),
            price_row("DDD", "2024-06-12", 162, 162),
            price_row("BENCH", "2024-06-10", 50, 50),
            price_row("BENCH", "2024-06-11", 50, 50),
            price_row("BENCH", "2024-06-12", 50, 50),
        ]
    )
    return signals, prices


def test_long_and_short_costs_and_theoretical_borrow(xcal) -> None:
    signals, prices = _book(xcal)
    model = FlatBpsCostModel(bps=15, name="base")
    long_book = simulate_portfolio(
        signals,
        prices,
        xcal,
        mode="long_additions",
        scheme="equal",
        cost_model=model,
        max_name_weight=1.0,
        gross_exposure_cap=1.0,
        benchmark_security_id="BENCH",
    )
    long_row = long_book.rebalance_returns.iloc[0]
    assert long_row["gross_return"] == pytest.approx(0.21)
    assert long_row["net_return"] == pytest.approx(0.21 - 0.003)
    assert long_book.contains_theoretical_shorts is False

    short_book = simulate_portfolio(
        signals,
        prices,
        xcal,
        mode="short_deletions",
        scheme="equal",
        cost_model=model,
        max_name_weight=1.0,
        gross_exposure_cap=1.0,
        benchmark_security_id="BENCH",
    )
    short_row = short_book.rebalance_returns.iloc[0]
    # Underlying deletion return compounds to -0.19. The short weight flips the sign.
    assert short_row["gross_return"] == pytest.approx(0.19)
    assert short_row["net_return"] == pytest.approx(0.19 - 0.003)
    assert short_book.contains_theoretical_shorts is True
    assert short_book.trades.iloc[0]["borrow_status"] == "theoretical"


def test_hedged_book_charges_the_etf_leg(xcal) -> None:
    signals, prices = _book(xcal)
    result = simulate_portfolio(
        signals,
        prices,
        xcal,
        mode="long_additions_hedged",
        scheme="equal",
        cost_model=FlatBpsCostModel(bps=15, name="base"),
        max_name_weight=1.0,
        gross_exposure_cap=1.0,
        benchmark_security_id="BENCH",
    )
    row = result.rebalance_returns.iloc[0]
    assert row["gross_return"] == pytest.approx(0.105)
    assert row["net_return"] == pytest.approx(0.105 - 0.003)
    assert set(result.trades["side"]) == {"long", "hedge"}
