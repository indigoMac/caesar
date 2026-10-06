"""Look-ahead guards fail closed."""

from __future__ import annotations

import pandas as pd
import pytest

from rebalance.data.validation import assert_output_not_raw
from rebalance.errors import DataContractError, LookaheadError
from rebalance.events.timeline import assert_fill_time, assert_no_lookahead
from rebalance.features.liquidity import compute_liquidity_features
from rebalance.features.pressure import compute_pressure
from rebalance.portfolio.simulator import borrow_status_for
from tests.conftest import et, flat_prices


def test_feature_timestamp_cannot_exceed_signal_timestamp() -> None:
    known = et("2024-06-07 17:00")
    earliest = et("2024-06-10 09:30")
    assert_no_lookahead(known, known, earliest)
    with pytest.raises(LookaheadError):
        assert_no_lookahead(known + pd.Timedelta(nanoseconds=1), known, earliest)


def test_trade_cannot_occur_before_signal_is_public() -> None:
    known = et("2024-06-07 17:00")
    earliest = et("2024-06-10 09:30")
    with pytest.raises(LookaheadError):
        assert_no_lookahead(known, earliest, earliest)
    with pytest.raises(LookaheadError):
        assert_fill_time(known, earliest, known, known)
    assert_fill_time(earliest, earliest, known, known)


def test_liquidity_ignores_bars_after_the_signal(xcal) -> None:
    sessions = xcal.sessions_between("2024-03-01", "2024-06-20")
    prices = pd.concat(
        [
            flat_prices("AAA", sessions, 20.0),
            flat_prices("BENCH", sessions, 50.0),
        ],
        ignore_index=True,
    )
    known = et("2024-06-07 17:00")
    earliest = xcal.earliest_regular_open_after(known)
    signals = pd.DataFrame(
        [
            {
                "rebalance_id": "2024-06",
                "security_id": "AAA",
                "action": "ADD",
                "signal_known_at": known,
                "earliest_trade_at": earliest,
            }
        ]
    )
    before, _ = compute_liquidity_features(signals, prices, xcal, "BENCH")
    spiked = prices.copy()
    spike_day = pd.Timestamp("2024-06-10")
    spiked.loc[(spiked["security_id"] == "AAA") & (spiked["date"] == spike_day), "volume_raw"] = 1e15
    after, _ = compute_liquidity_features(signals, spiked, xcal, "BENCH")
    assert before.iloc[0]["adv20"] == pytest.approx(after.iloc[0]["adv20"])
    assert before.iloc[0]["feature_asof"] <= known


def test_fundamental_published_after_the_signal_is_ignored(xcal) -> None:
    sessions = xcal.sessions_between("2024-03-01", "2024-06-20")
    prices = pd.concat(
        [flat_prices("AAA", sessions, 20.0), flat_prices("BENCH", sessions, 50.0)],
        ignore_index=True,
    )
    known = et("2024-06-07 17:00")
    signals = pd.DataFrame(
        [
            {
                "rebalance_id": "2024-06",
                "security_id": "AAA",
                "action": "ADD",
                "signal_known_at": known,
                "earliest_trade_at": xcal.earliest_regular_open_after(known),
            }
        ]
    )
    fundamentals = pd.DataFrame(
        [
            {"security_id": "AAA", "asof": et("2024-05-01 16:00"), "market_cap": 10.0, "delta_weight": 0.01, "estimated_flow": pd.NA},
            {"security_id": "AAA", "asof": et("2024-06-10 16:00"), "market_cap": 9e12, "delta_weight": 5.0, "estimated_flow": 1e9},
        ]
    )
    features, _ = compute_liquidity_features(signals, prices, xcal, "BENCH", fundamentals)
    assert features.iloc[0]["market_cap"] == 10.0
    assert features.iloc[0]["delta_weight"] == pytest.approx(0.01)
    assert pd.isna(features.iloc[0]["estimated_flow"])
    assert features.iloc[0]["adv20"] == pytest.approx(20.0 * 1_000_000.0)


def test_borrow_record_after_the_signal_is_not_used() -> None:
    known = et("2024-06-07 17:00")
    borrow = pd.DataFrame(
        [{"security_id": "AAA", "asof": et("2024-06-10 09:00"), "borrow_available": True}]
    )
    assert borrow_status_for("AAA", known, borrow) == "theoretical_no_record"
    earlier = pd.DataFrame(
        [{"security_id": "AAA", "asof": et("2024-06-01 16:00"), "borrow_available": False}]
    )
    assert borrow_status_for("AAA", known, earlier) == "unavailable"
    assert borrow_status_for("AAA", known, None) == "theoretical"


def test_proxy_pressure_is_not_stored_as_estimated_flow(xcal) -> None:
    sessions = xcal.sessions_between("2024-03-01", "2024-06-20")
    prices = pd.concat(
        [flat_prices("AAA", sessions, 20.0), flat_prices("BENCH", sessions, 50.0)],
        ignore_index=True,
    )
    known = et("2024-06-07 17:00")
    signals = pd.DataFrame(
        [
            {
                "rebalance_id": "2024-06",
                "security_id": "AAA",
                "action": "ADD",
                "signal_known_at": known,
                "earliest_trade_at": xcal.earliest_regular_open_after(known),
            }
        ]
    )
    fundamentals = pd.DataFrame(
        [
            {
                "security_id": "AAA",
                "asof": et("2024-05-01 16:00"),
                "market_cap": 10.0,
                "delta_weight": 0.01,
                "estimated_flow": pd.NA,
            }
        ]
    )
    features, _ = compute_liquidity_features(signals, prices, xcal, "BENCH", fundamentals)
    scored = compute_pressure(features)
    assert scored.iloc[0]["pressure_kind"] == "abs_delta_weight_over_adv20"
    assert bool(scored.iloc[0]["pressure_is_proxy"]) is True
    assert pd.isna(scored.iloc[0]["estimated_flow"])
    scored.loc[0, "estimated_flow"] = 2_000_000.0
    directed = compute_pressure(scored)
    assert directed.iloc[0]["pressure_kind"] == "net_flow_over_adv20"
    assert bool(directed.iloc[0]["pressure_is_proxy"]) is False
    assert directed.iloc[0]["pressure_score"] == pytest.approx(2_000_000.0 / scored.iloc[0]["adv20"])


def test_refuses_writes_under_raw(tmp_path) -> None:
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    with pytest.raises(DataContractError):
        assert_output_not_raw(raw / "russell" / "events.csv", raw)
