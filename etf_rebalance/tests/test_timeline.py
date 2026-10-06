"""Calendar execution rules: weekends, holidays, and post-close announcements."""

from __future__ import annotations

import pandas as pd

from rebalance.events.timeline import select_tradable_events
from tests.conftest import et


def _announcement(published: pd.Timestamp, effective: str, action: str = "ADD", security_id: str = "AAA") -> dict:
    return {
        "rebalance_id": "2024-06",
        "index_name": "RUSSELL_2000",
        "security_id": security_id,
        "ticker_published": security_id,
        "company_name_published": security_id,
        "action": action,
        "published_at": published,
        "effective_close": pd.Timestamp(effective),
        "version": 1,
        "source_url": "https://example.invalid/test",
        "source_hash": "hash",
        "retrieved_at": et("2024-08-01 12:00"),
    }


def test_friday_after_close_skips_memorial_day(xcal) -> None:
    known = et("2024-05-24 17:00")
    earliest = xcal.earliest_regular_open_after(known)
    local = earliest.tz_convert("America/New_York")
    assert local == pd.Timestamp("2024-05-28 09:30", tz="America/New_York")
    assert xcal.entry_session(earliest) == pd.Timestamp("2024-05-28")
    assert not xcal.is_session("2024-05-27")
    assert xcal.session_distance("2024-05-24", "2024-05-28") == 1


def test_ordinary_friday_after_close_trades_monday_open(xcal) -> None:
    earliest = xcal.earliest_regular_open_after(et("2024-06-07 17:00"))
    assert earliest.tz_convert("America/New_York") == pd.Timestamp("2024-06-10 09:30", tz="America/New_York")


def test_intraday_announcement_cannot_use_that_sessions_open(xcal) -> None:
    earliest = xcal.earliest_regular_open_after(et("2024-06-07 11:00"))
    assert earliest.tz_convert("America/New_York") == pd.Timestamp("2024-06-10 09:30", tz="America/New_York")


def test_preopen_announcement_can_trade_the_same_sessions_open(xcal) -> None:
    earliest = xcal.earliest_regular_open_after(et("2024-06-10 08:00"))
    assert earliest.tz_convert("America/New_York") == pd.Timestamp("2024-06-10 09:30", tz="America/New_York")


def test_good_friday_and_independence_day(xcal) -> None:
    after_thursday = xcal.earliest_regular_open_after(et("2024-03-28 17:00"))
    assert after_thursday.tz_convert("America/New_York") == pd.Timestamp("2024-04-01 09:30", tz="America/New_York")
    assert xcal.session_distance("2024-03-28", "2024-04-01") == 1
    after_july3 = xcal.earliest_regular_open_after(et("2024-07-03 17:00"))
    assert after_july3.tz_convert("America/New_York") == pd.Timestamp("2024-07-05 09:30", tz="America/New_York")
    assert xcal.session_distance("2024-07-03", "2024-07-05") == 1
    assert not xcal.is_session("2024-07-04")


def test_holiday_week_event_offsets_skip_the_closed_session(xcal) -> None:
    events = pd.DataFrame([_announcement(et("2024-07-03 17:00"), "2024-07-08")])
    trades, excluded = select_tradable_events(events, xcal)
    assert excluded.empty
    trade = trades.iloc[0]
    assert trade["entry_session"] == pd.Timestamp("2024-07-05")
    assert xcal.session_distance(trade["entry_session"], trade["effective_close"]) == 1


def test_non_session_effective_close_is_excluded(xcal) -> None:
    events = pd.DataFrame([_announcement(et("2024-06-07 17:00"), "2024-07-04")])
    trades, excluded = select_tradable_events(events, xcal)
    assert trades.empty
    assert excluded.iloc[0]["reason"] == "effective_close_not_session"
