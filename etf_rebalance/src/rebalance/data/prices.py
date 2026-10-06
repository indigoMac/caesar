"""Raw price bars and optional corporate-action attachment.

Raw open, high, low, close, and volume are never replaced with adjusted
prices. `split_factor` and `dividend` sit beside them. Total returns are
computed in `rebalance.features.returns`.

TODO: Ingest a licensed price and corporate-action source. This module only
reads the local CSV contract below. It does not download market data.

Price columns:
    security_id, date, open_raw, high_raw, low_raw, close_raw, volume_raw,
    split_factor, dividend

`date` is an exchange session label. `split_factor` is 1 when the session is
not an ex-split date. `dividend` is cash per share in that session's price
units (post-split), and 0 when there is no dividend.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from rebalance.calendar import TradingCalendar, session_label
from rebalance.data.validation import PRICE_COLUMNS, validate_prices
from rebalance.errors import DataContractError


def load_prices(path: Path, calendar: TradingCalendar | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = [column for column in PRICE_COLUMNS if column not in frame.columns]
    if missing:
        raise DataContractError(f"price CSV is missing columns: {missing}")
    frame["security_id"] = frame["security_id"].astype(str)
    frame["date"] = frame["date"].map(lambda value: session_label(value))
    for column in ("open_raw", "high_raw", "low_raw", "close_raw", "volume_raw", "split_factor", "dividend"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    validate_prices(frame)
    if calendar is not None:
        bad = sorted({pd.Timestamp(day).date().isoformat() for day in frame["date"].unique() if not calendar.is_session(day)})
        if bad:
            preview = ", ".join(bad[:8])
            raise DataContractError(f"price file contains non-session dates: {preview}")
    return frame.sort_values(["security_id", "date"]).reset_index(drop=True)


def load_fundamentals(path: Path, naive_timestamps_tz: str | None = None) -> pd.DataFrame:
    """Load optional point-in-time market cap, weight changes, and estimated flow.

    `asof` is the instant the row became public. Blank `estimated_flow` stays
    missing. This loader does not invent index demand.
    """
    from rebalance.data.russell import _parse_instant

    frame = pd.read_csv(path)
    if "security_id" not in frame.columns or "asof" not in frame.columns:
        raise DataContractError("fundamentals CSV requires security_id and asof")
    frame["security_id"] = frame["security_id"].astype(str)
    frame["asof"] = frame["asof"].map(lambda value: _parse_instant(value, "asof", naive_timestamps_tz))
    for column in ("market_cap", "delta_weight", "estimated_flow"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        else:
            frame[column] = pd.NA
    if frame.duplicated(["security_id", "asof"]).any():
        raise DataContractError("fundamentals has duplicate (security_id, asof) rows")
    return frame.sort_values(["security_id", "asof"]).reset_index(drop=True)


def load_borrow(path: Path, naive_timestamps_tz: str | None = None) -> pd.DataFrame:
    """Load optional point-in-time borrow availability. Absence is not availability."""
    from rebalance.data.russell import _parse_instant

    frame = pd.read_csv(path)
    for column in ("security_id", "asof", "borrow_available"):
        if column not in frame.columns:
            raise DataContractError(f"borrow CSV is missing {column}")
    frame["security_id"] = frame["security_id"].astype(str)
    frame["asof"] = frame["asof"].map(lambda value: _parse_instant(value, "asof", naive_timestamps_tz))
    frame["borrow_available"] = frame["borrow_available"].map(_parse_bool)
    if frame.duplicated(["security_id", "asof"]).any():
        raise DataContractError("borrow file has duplicate (security_id, asof) rows")
    return frame.sort_values(["security_id", "asof"]).reset_index(drop=True)


def _parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "false", "f", "no", "n"}:
        return False
    raise DataContractError(f"borrow_available value {value!r} is not boolean")


def merge_corporate_actions(prices: pd.DataFrame, actions: pd.DataFrame) -> pd.DataFrame:
    """Fill split_factor and dividend from a corporate-action file.

    Open, high, low, close, and volume are copied through unchanged. If the
    price bar already records a different split or dividend, this raises
    instead of overwriting raw prices or silently summing factors.
    """
    required = {"security_id", "date", "split_factor", "dividend"}
    missing = required - set(actions.columns)
    if missing:
        raise DataContractError(f"corporate actions missing columns: {sorted(missing)}")
    merged = prices.copy()
    actions = actions.copy()
    actions["security_id"] = actions["security_id"].astype(str)
    actions["date"] = actions["date"].map(lambda value: session_label(value))
    actions["split_factor"] = pd.to_numeric(actions["split_factor"], errors="coerce")
    actions["dividend"] = pd.to_numeric(actions["dividend"], errors="coerce")
    if actions[["split_factor", "dividend"]].isna().any().any():
        raise DataContractError("corporate actions contain non-numeric split_factor or dividend")
    lookup = actions.set_index(["security_id", "date"])
    if lookup.index.has_duplicates:
        raise DataContractError("corporate actions has duplicate (security_id, date) rows")
    for position, row in merged.iterrows():
        key = (row["security_id"], row["date"])
        if key not in lookup.index:
            continue
        action = lookup.loc[key]
        for column in ("split_factor", "dividend"):
            current = float(row[column])
            incoming = float(action[column])
            if current == 1.0 and column == "split_factor" and incoming != 1.0:
                merged.at[position, column] = incoming
            elif current == 0.0 and column == "dividend" and incoming != 0.0:
                merged.at[position, column] = incoming
            elif current != incoming:
                raise DataContractError(
                    f"{column} conflict for {key[0]} on {pd.Timestamp(key[1]).date()}: "
                    f"prices have {current}, actions have {incoming}"
                )
    validate_prices(merged)
    for column in ("open_raw", "high_raw", "low_raw", "close_raw", "volume_raw"):
        if not merged[column].equals(prices[column]):
            raise DataContractError(f"corporate-action merge changed raw column {column}")
    return merged
