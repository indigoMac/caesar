"""Total-return math on raw bars.

Convention, applied consistently and tested directly:

* Prices on an ex-date are already in post-split units.
* `split_factor` is new shares per old share (2.0 for a 2-for-1). It is 1
  when there is no split. It converts the previous close into today's units.
* `dividend` is cash per post-split share with that session as the ex-date.
* Close-to-close total return on session t is

      (close_t + dividend_t) / (close_{t-1} / split_factor_t) - 1

  Compounding these returns reinvests cash dividends.
* A buyer at the open does not receive that session's dividend and does not
  apply that session's split to the open, because the open is already
  post-split. Same-session open-to-close return is `close / open - 1`.
* The first session of a holding period that starts at the open uses the
  open-to-close return. Later sessions use close-to-close total return.
"""

from __future__ import annotations

import pandas as pd

from rebalance.calendar import TradingCalendar, session_label
from rebalance.errors import DataContractError, MissingPriceError


def close_to_close_total_return(
    previous_close: float,
    close: float,
    split_factor: float,
    dividend: float,
) -> float:
    if split_factor <= 0:
        raise DataContractError("split_factor must be positive")
    if previous_close <= 0 or close <= 0:
        raise DataContractError("close prices must be positive")
    if dividend < 0:
        raise DataContractError("dividend cannot be negative")
    denominator = previous_close / split_factor
    return (close + dividend) / denominator - 1.0


def open_to_close_return(open_price: float, close: float) -> float:
    if open_price <= 0 or close <= 0:
        raise DataContractError("open and close must be positive")
    return close / open_price - 1.0


def index_prices(prices: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Index raw bars by security_id and session date."""
    indexed: dict[str, pd.DataFrame] = {}
    for security_id, frame in prices.groupby("security_id", sort=False):
        ordered = frame.sort_values("date").set_index("date")
        ordered.index = pd.DatetimeIndex(session_label(pd.Timestamp(value)) for value in ordered.index)
        if ordered.index.has_duplicates:
            raise DataContractError(f"duplicate sessions for {security_id}")
        indexed[str(security_id)] = ordered
    return indexed


def session_total_return(
    frame: pd.DataFrame,
    session: pd.Timestamp,
    basis: str,
    calendar: TradingCalendar,
) -> float:
    label = calendar.require_session(session)
    if label not in frame.index:
        raise MissingPriceError(f"missing price bar on {label.date()}")
    row = frame.loc[label]
    if basis == "open_to_close":
        return open_to_close_return(float(row["open_raw"]), float(row["close_raw"]))
    if basis != "close_to_close":
        raise DataContractError(f"unknown return basis {basis}")
    previous = calendar.shift_session(label, -1)
    if previous not in frame.index:
        raise MissingPriceError(f"missing previous close on {previous.date()} for return on {label.date()}")
    return close_to_close_total_return(
        float(frame.loc[previous, "close_raw"]),
        float(row["close_raw"]),
        float(row["split_factor"]),
        float(row["dividend"]),
    )


def holding_period_total_return(
    frame: pd.DataFrame,
    entry_session: pd.Timestamp,
    exit_session: pd.Timestamp,
    calendar: TradingCalendar,
) -> float:
    """Buy-and-hold total return from the entry open to the exit close."""
    entry = calendar.require_session(entry_session)
    exit_label = calendar.require_session(exit_session)
    if exit_label < entry:
        raise DataContractError(f"exit session {exit_label.date()} is before entry {entry.date()}")
    sessions = list(calendar.sessions_between(entry, exit_label))
    missing = [session.date().isoformat() for session in sessions if session not in frame.index]
    if missing:
        raise MissingPriceError(f"holding period is missing sessions: {missing}")
    first = frame.loc[entry]
    if exit_label == entry:
        return open_to_close_return(float(first["open_raw"]), float(first["close_raw"]))
    wealth = float(first["close_raw"]) / float(first["open_raw"])
    previous_close = float(first["close_raw"])
    for session in sessions[1:]:
        row = frame.loc[session]
        split_factor = float(row["split_factor"])
        wealth *= (float(row["close_raw"]) + float(row["dividend"])) / (previous_close / split_factor)
        previous_close = float(row["close_raw"])
    return wealth - 1.0
