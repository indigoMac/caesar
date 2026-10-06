"""Historical ticker mappings. A ticker is never the permanent identifier."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from rebalance.calendar import as_utc
from rebalance.data.validation import validate_security_master
from rebalance.errors import DataContractError

OPEN_ENDED = pd.Timestamp("2200-01-01")


def _parse_date(value: object, field: str) -> pd.Timestamp:
    if value is None or (isinstance(value, float) and pd.isna(value)) or pd.isna(value):
        raise DataContractError(f"{field} is missing")
    parsed = pd.Timestamp(value)
    if parsed.tzinfo is not None:
        raise DataContractError(f"{field} must be a calendar date, not a timezone-aware instant")
    normalized = parsed.normalize()
    if parsed != normalized:
        raise DataContractError(f"{field} must be a calendar date")
    return normalized


def load_security_master(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "valid_to" in frame.columns:
        frame["valid_to"] = frame["valid_to"].where(frame["valid_to"].notna(), OPEN_ENDED)
    frame["valid_from"] = frame["valid_from"].map(lambda value: _parse_date(value, "valid_from"))
    frame["valid_to"] = frame["valid_to"].map(lambda value: _parse_date(value, "valid_to"))
    frame["security_id"] = frame["security_id"].astype(str)
    frame["ticker"] = frame["ticker"].astype(str)
    frame["company_name"] = frame["company_name"].astype(str)
    validate_security_master(frame)
    _assert_no_ticker_overlaps(frame)
    return frame.sort_values(["ticker", "valid_from"]).reset_index(drop=True)


def _assert_no_ticker_overlaps(master: pd.DataFrame) -> None:
    for ticker, group in master.groupby("ticker", sort=False):
        ordered = group.sort_values(["valid_from", "valid_to"])
        previous_end: pd.Timestamp | None = None
        for row in ordered.itertuples(index=False):
            if previous_end is not None and row.valid_from <= previous_end:
                raise DataContractError(f"overlapping ticker interval for {ticker}")
            previous_end = row.valid_to


def resolve_security_id(
    master: pd.DataFrame,
    ticker: str,
    asof: pd.Timestamp,
    timezone: str = "America/New_York",
) -> str:
    """Map a published ticker to security_id using only intervals covering `asof`."""
    day = as_utc(asof).tz_convert(timezone).normalize().tz_localize(None)
    hits = master[(master["ticker"] == str(ticker)) & (master["valid_from"] <= day) & (master["valid_to"] >= day)]
    if len(hits) != 1:
        raise DataContractError(
            f"ticker {ticker} on {day.date()} matched {len(hits)} security master rows; expected exactly one"
        )
    return str(hits.iloc[0]["security_id"])
