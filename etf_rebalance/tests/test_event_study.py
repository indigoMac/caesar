"""Event panel, tradable entry return, and addition versus deletion CARs."""

from __future__ import annotations

import pandas as pd
import pytest

from rebalance.config import WindowSpec
from rebalance.errors import DataContractError
from rebalance.events.timeline import select_tradable_events
from rebalance.research.event_study import build_event_panel, mean_abnormal_return_by_offset, window_cars
from tests.conftest import et, flat_prices, price_row


def _signals_and_prices(xcal):
    known = et("2024-06-07 17:00")
    earliest = xcal.earliest_regular_open_after(known)
    entry = xcal.entry_session(earliest)
    assert entry == pd.Timestamp("2024-06-10")
    base = {
        "rebalance_id": "2024-06",
        "index_name": "RUSSELL_2000",
        "ticker_published": "X",
        "signal_known_at": known,
        "feature_asof": known,
        "earliest_trade_at": earliest,
        "entry_session": entry,
        "effective_close": pd.Timestamp("2024-06-12"),
    }
    signals = pd.DataFrame(
        [
            {**base, "security_id": "AAA", "ticker_published": "AAA", "action": "ADD"},
            {**base, "security_id": "DDD", "ticker_published": "DDD", "action": "DELETE"},
        ]
    )
    sessions = xcal.sessions_between("2024-06-03", "2024-07-15")
    prices = pd.concat(
        [
            flat_prices("AAA", sessions, 90.0),
            flat_prices("DDD", sessions, 200.0),
            flat_prices("BENCH", sessions, 50.0),
        ],
        ignore_index=True,
    )
    replacements = {
        ("AAA", "2024-06-07"): (90.0, 90.0),
        ("AAA", "2024-06-10"): (100.0, 110.0),
        ("AAA", "2024-06-11"): (110.0, 121.0),
        ("AAA", "2024-06-12"): (121.0, 121.0),
        ("AAA", "2024-06-13"): (121.0, 108.9),
        ("DDD", "2024-06-10"): (200.0, 180.0),
        ("DDD", "2024-06-11"): (180.0, 162.0),
        ("DDD", "2024-06-12"): (162.0, 162.0),
    }
    for (security_id, day), (open_price, close) in replacements.items():
        mask = (prices["security_id"] == security_id) & (prices["date"] == pd.Timestamp(day))
        prices.loc[mask, "open_raw"] = open_price
        prices.loc[mask, "close_raw"] = close
        prices.loc[mask, "high_raw"] = max(open_price, close) + 0.01
        prices.loc[mask, "low_raw"] = min(open_price, close) - 0.01
    return signals, prices


def test_panel_uses_open_on_entry_and_separates_actions(xcal) -> None:
    signals, prices = _signals_and_prices(xcal)
    panel, diagnostics = build_event_panel(signals, prices, xcal, "BENCH", pre_sessions=5, post_sessions=1)
    assert diagnostics.empty
    addition = panel[(panel["security_id"] == "AAA") & (panel["date"] == pd.Timestamp("2024-06-10"))].iloc[0]
    assert addition["return_basis"] == "open_to_close"
    assert addition["stock_return"] == pytest.approx(0.10)
    assert addition["days_from_announcement"] == 0
    assert addition["days_from_rebalance"] == -2
    # Close-to-close from the prior close of 90 would have been 110/90 - 1, which is not tradable.
    assert addition["stock_return"] != pytest.approx(110 / 90 - 1)

    windows = [
        WindowSpec(name="ann_to_rebalance", start_anchor="entry", start_offset=0, end_anchor="rebalance", end_offset=0),
        WindowSpec(name="rebalance_to_plus_1", start_anchor="rebalance", start_offset=1, end_anchor="rebalance", end_offset=1),
        WindowSpec(name="ann_plus_5_to_rebalance", start_anchor="entry", start_offset=5, end_anchor="rebalance", end_offset=0),
    ]
    cars = window_cars(panel, xcal, windows)
    add_car = cars[(cars["security_id"] == "AAA") & (cars["window"] == "ann_to_rebalance")].iloc[0]
    delete_car = cars[(cars["security_id"] == "DDD") & (cars["window"] == "ann_to_rebalance")].iloc[0]
    post = cars[(cars["security_id"] == "AAA") & (cars["window"] == "rebalance_to_plus_1")].iloc[0]
    too_short = cars[(cars["security_id"] == "AAA") & (cars["window"] == "ann_plus_5_to_rebalance")].iloc[0]
    assert add_car["car"] == pytest.approx(0.20)
    assert delete_car["car"] == pytest.approx(-0.20)
    assert delete_car["action"] == "DELETE"
    assert post["car"] == pytest.approx(-0.10)
    assert too_short["window_status"] == "empty_window"
    assert pd.isna(too_short["car"])

    averaged = mean_abnormal_return_by_offset(panel)
    entry_rows = averaged[averaged["days_from_rebalance"] == -2]
    assert set(entry_rows["action"]) == {"ADD", "DELETE"}
    pandas_mean = panel.groupby(["action", "days_from_rebalance"], sort=False)["abnormal_return"].mean()
    for row in averaged.itertuples(index=False):
        assert row.mean_abnormal_return == pytest.approx(float(pandas_mean.loc[(row.action, row.days_from_rebalance)]))


def test_missing_stock_nulls_that_event_and_missing_benchmark_raises(xcal) -> None:
    signals, prices = _signals_and_prices(xcal)
    broken = prices[~((prices["security_id"] == "AAA") & (prices["date"] == pd.Timestamp("2024-06-11")))].copy()
    panel, diagnostics = build_event_panel(signals, broken, xcal, "BENCH", pre_sessions=0, post_sessions=0)
    windows = [WindowSpec(name="ann_to_rebalance", start_anchor="entry", start_offset=0, end_anchor="rebalance", end_offset=0)]
    cars = window_cars(panel, xcal, windows)
    add = cars[cars["security_id"] == "AAA"].iloc[0]
    delete = cars[cars["security_id"] == "DDD"].iloc[0]
    assert add["window_status"] == "missing_return"
    assert pd.isna(add["car"])
    assert delete["car"] == pytest.approx(-0.20)
    assert not diagnostics.empty

    missing_benchmark = prices[prices["security_id"] != "BENCH"]
    with pytest.raises(DataContractError, match="benchmark"):
        build_event_panel(signals, missing_benchmark, xcal, "BENCH", pre_sessions=0, post_sessions=0)


def test_cancellation_before_the_open_is_not_traded(xcal) -> None:
    rows = [
        {
            "rebalance_id": "2024-06",
            "index_name": "RUSSELL_2000",
            "security_id": security_id,
            "ticker_published": security_id,
            "company_name_published": security_id,
            "action": action,
            "published_at": published,
            "effective_close": pd.Timestamp("2024-06-28"),
            "version": version,
            "source_url": "https://example.invalid/test",
            "source_hash": f"{security_id}-{version}",
            "retrieved_at": et("2024-08-01 12:00"),
        }
        for security_id, action, published, version in [
            ("AAA", "ADD", et("2024-05-24 17:00"), 1),
            ("XXX", "ADD", et("2024-05-24 17:00"), 1),
            ("XXX", "NONE", et("2024-05-27 18:00"), 2),
        ]
    ]
    trades, excluded = select_tradable_events(pd.DataFrame(rows), xcal)
    assert list(trades["security_id"]) == ["AAA"]
    assert trades.iloc[0]["entry_session"] == pd.Timestamp("2024-05-28")
    assert excluded.iloc[0]["security_id"] == "XXX"
    assert excluded.iloc[0]["reason"] == "cancelled_before_entry"


def test_corporate_action_merge_does_not_change_raw_prices() -> None:
    from rebalance.data.prices import merge_corporate_actions

    prices = pd.DataFrame(
        [
            price_row("AAA", "2024-06-10", 100, 50, split_factor=1),
            price_row("AAA", "2024-06-11", 50, 50, split_factor=1),
        ]
    )
    actions = pd.DataFrame(
        [{"security_id": "AAA", "date": "2024-06-10", "split_factor": 2.0, "dividend": 0.0}]
    )
    merged = merge_corporate_actions(prices, actions)
    assert merged.loc[0, "split_factor"] == 2.0
    assert merged.loc[0, "close_raw"] == 50
    assert merged.loc[0, "open_raw"] == 100
