"""Signal timestamps and the first legal execution.

Look-ahead rule, enforced on every tradable signal:

    feature_asof <= signal_known_at < earliest_trade_at

`earliest_trade_at` is the next regular-session open strictly after the
announcement that determines the trade. A Friday post-close announcement
cannot fill at Friday's close.
"""

from __future__ import annotations

import pandas as pd

from rebalance.calendar import TradingCalendar, as_utc
from rebalance.errors import DataContractError, LookaheadError
from rebalance.events.revisions import event_state_asof

SIGNAL_COLUMNS = (
    "rebalance_id",
    "index_name",
    "security_id",
    "ticker_published",
    "company_name_published",
    "action",
    "signal_known_at",
    "feature_asof",
    "earliest_trade_at",
    "entry_session",
    "effective_close",
    "version",
    "source_url",
    "source_hash",
    "first_published_at",
)


def assert_no_lookahead(
    feature_asof: pd.Timestamp,
    signal_known_at: pd.Timestamp,
    earliest_trade_at: pd.Timestamp,
) -> None:
    """Reject features dated after the signal, or trades that are not strictly later."""
    feature = as_utc(feature_asof)
    known = as_utc(signal_known_at)
    earliest = as_utc(earliest_trade_at)
    if feature > known:
        raise LookaheadError(
            f"feature_asof {feature.isoformat()} is after signal_known_at {known.isoformat()}"
        )
    if known >= earliest:
        raise LookaheadError(
            f"signal_known_at {known.isoformat()} is not strictly before "
            f"earliest_trade_at {earliest.isoformat()}"
        )


def assert_fill_time(
    fill_at: pd.Timestamp,
    earliest_trade_at: pd.Timestamp,
    signal_known_at: pd.Timestamp,
    feature_asof: pd.Timestamp,
) -> None:
    """Reject a fill that occurs before the signal is public and tradable."""
    assert_no_lookahead(feature_asof, signal_known_at, earliest_trade_at)
    fill = as_utc(fill_at)
    earliest = as_utc(earliest_trade_at)
    if fill < earliest:
        raise LookaheadError(
            f"fill {fill.isoformat()} is before the first legal trade {earliest.isoformat()}"
        )


def select_tradable_events(
    events: pd.DataFrame,
    calendar: TradingCalendar,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Choose the action that was public at the first open after it was known.

    Revisions published before that open replace the earlier action. A NONE
    published before the open drops the name. Revisions are not applied
    before their `published_at`, and earlier rows stay in `events`.
    """
    if events.empty:
        raise DataContractError("no announcements to select")
    trades: list[dict[str, object]] = []
    excluded: list[dict[str, object]] = []
    grouped = events.groupby(["rebalance_id", "index_name", "security_id"], sort=False)
    for (rebalance_id, index_name, security_id), group in grouped:
        subset = group.sort_values(["published_at", "version", "retrieved_at"], kind="mergesort")
        first_active = subset[subset["action"].isin(["ADD", "DELETE"])]
        base = {
            "rebalance_id": rebalance_id,
            "index_name": index_name,
            "security_id": security_id,
            "ticker_published": subset.iloc[0]["ticker_published"],
            "company_name_published": subset.iloc[0]["company_name_published"],
        }
        if first_active.empty:
            excluded.append({**base, "reason": "no_add_or_delete"})
            continue
        cursor = first_active.iloc[0]
        first_published_at = as_utc(cursor["published_at"])
        resolved: pd.Series | None = None
        earliest = None
        for _ in range(len(subset) + 1):
            signal_known_at = as_utc(cursor["published_at"])
            earliest = calendar.earliest_regular_open_after(signal_known_at)
            decision_time = earliest - pd.Timedelta(nanoseconds=1)
            visible = subset.loc[pd.to_datetime(subset["published_at"], utc=True) <= decision_time]
            if visible.empty:
                resolved = None
                break
            latest = visible.iloc[-1]
            if latest["action"] == "NONE":
                resolved = None
                break
            same_row = as_utc(latest["published_at"]) == signal_known_at and int(latest["version"]) == int(
                cursor["version"]
            )
            if same_row:
                resolved = latest
                break
            cursor = latest
        if resolved is None or earliest is None:
            excluded.append(
                {
                    **base,
                    "reason": "cancelled_before_entry",
                    "first_published_at": first_published_at,
                }
            )
            continue
        signal_known_at = as_utc(resolved["published_at"])
        feature_asof = signal_known_at
        entry_session = calendar.entry_session(earliest)
        try:
            effective_close = calendar.require_session(resolved["effective_close"])
        except DataContractError as exc:
            excluded.append({**base, "reason": "effective_close_not_session", "detail": str(exc)})
            continue
        exit_close = calendar.session_close(effective_close)
        if earliest >= exit_close:
            excluded.append(
                {
                    **base,
                    "reason": "announced_after_effective_close",
                    "signal_known_at": signal_known_at,
                    "earliest_trade_at": earliest,
                }
            )
            continue
        assert_no_lookahead(feature_asof, signal_known_at, earliest)
        trades.append(
            {
                **base,
                "ticker_published": resolved["ticker_published"],
                "company_name_published": resolved["company_name_published"],
                "action": resolved["action"],
                "signal_known_at": signal_known_at,
                "feature_asof": feature_asof,
                "earliest_trade_at": earliest,
                "entry_session": entry_session,
                "effective_close": effective_close,
                "version": int(resolved["version"]),
                "source_url": resolved["source_url"],
                "source_hash": resolved["source_hash"],
                "first_published_at": first_published_at,
            }
        )
    trade_frame = pd.DataFrame(trades, columns=list(SIGNAL_COLUMNS))
    exclusion_frame = pd.DataFrame(excluded)
    return trade_frame, exclusion_frame
