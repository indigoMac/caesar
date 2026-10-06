"""Announcement revisions stay in the log and only affect state after publication."""

from __future__ import annotations

import pandas as pd

from rebalance.events.revisions import event_state_asof
from tests.conftest import et


def _row(published: str, action: str, version: int) -> dict[str, object]:
    return {
        "rebalance_id": "2024-06",
        "index_name": "RUSSELL_2000",
        "security_id": "ABC",
        "ticker_published": "ABC",
        "company_name_published": "ABC Corp",
        "action": action,
        "published_at": et(published),
        "effective_close": pd.Timestamp("2024-06-28"),
        "version": version,
        "source_url": "https://example.invalid/revisions",
        "source_hash": f"{action}-{version}",
        "retrieved_at": et("2024-08-01 12:00"),
    }


def _log() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _row("2024-05-22 16:30", "ADD", 1),
            _row("2024-05-29 16:30", "ADD", 2),
            _row("2024-06-05 16:30", "NONE", 3),
        ]
    )


def test_revision_sequence_preserves_history() -> None:
    events = _log()
    original = events.copy()
    may_28 = event_state_asof(events, et("2024-05-28 12:00"))
    may_30 = event_state_asof(events, et("2024-05-30 12:00"))
    june_8 = event_state_asof(events, et("2024-06-08 12:00"))

    assert list(may_28["security_id"]) == ["ABC"]
    assert may_28.iloc[0]["action"] == "ADD"
    assert int(may_28.iloc[0]["version"]) == 1
    assert list(may_30["security_id"]) == ["ABC"]
    assert int(may_30.iloc[0]["version"]) == 2
    assert "ABC" not in set(june_8["security_id"])
    pd.testing.assert_frame_equal(events, original)
    assert (events["action"] == "ADD").sum() == 2


def test_same_timestamp_higher_version_wins() -> None:
    events = pd.DataFrame(
        [
            _row("2024-05-22 16:30", "ADD", 1),
            _row("2024-05-22 16:30", "DELETE", 2),
        ]
    )
    state = event_state_asof(events, et("2024-05-22 16:30"))
    assert state.iloc[0]["action"] == "DELETE"
    earlier = event_state_asof(events, et("2024-05-22 16:29"))
    assert earlier.empty


def test_future_rows_are_invisible() -> None:
    events = _log()
    state = event_state_asof(events, et("2024-05-21 12:00"))
    assert state.empty
