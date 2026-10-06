"""Schema checks for announcements, prices, and the security master.

Validators raise on contract violations. They do not coerce bad rows into
usable ones and they do not overwrite raw price fields with adjusted prices.
"""

from __future__ import annotations

import pandas as pd

from rebalance.errors import DataContractError

ANNOUNCEMENT_COLUMNS = (
    "rebalance_id",
    "security_id",
    "ticker_published",
    "company_name_published",
    "index_name",
    "action",
    "published_at",
    "effective_close",
    "version",
    "source_url",
    "source_hash",
    "retrieved_at",
)

PRICE_COLUMNS = (
    "security_id",
    "date",
    "open_raw",
    "high_raw",
    "low_raw",
    "close_raw",
    "volume_raw",
    "split_factor",
    "dividend",
)

SECURITY_MASTER_COLUMNS = (
    "security_id",
    "ticker",
    "valid_from",
    "valid_to",
    "company_name",
)

FUNDAMENTAL_COLUMNS = (
    "security_id",
    "asof",
)

ACTIONS = frozenset({"ADD", "DELETE", "NONE"})


def require_columns(frame: pd.DataFrame, columns: tuple[str, ...], kind: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise DataContractError(f"{kind} is missing columns: {missing}")


def validate_announcements(frame: pd.DataFrame) -> pd.DataFrame:
    require_columns(frame, ANNOUNCEMENT_COLUMNS, "announcements")
    if frame.empty:
        raise DataContractError("announcement file has no rows")
    invalid = sorted(set(frame["action"].astype(str)) - ACTIONS)
    if invalid:
        raise DataContractError(f"action must be ADD, DELETE, or NONE; found {invalid}")
    if frame["published_at"].isna().any() or frame["retrieved_at"].isna().any():
        raise DataContractError("published_at and retrieved_at are required on every row")
    if frame["effective_close"].isna().any():
        raise DataContractError("effective_close is required on every row")
    if (frame["version"].astype(int) < 1).any():
        raise DataContractError("version must be a positive integer")
    keys = ["rebalance_id", "index_name", "security_id", "published_at", "version"]
    duplicates = frame.duplicated(keys, keep=False)
    if duplicates.any():
        sample = frame.loc[duplicates, keys].head(5)
        raise DataContractError(f"duplicate announcement keys:\n{sample}")
    return frame


def validate_prices(frame: pd.DataFrame) -> pd.DataFrame:
    require_columns(frame, PRICE_COLUMNS, "prices")
    if frame.empty:
        raise DataContractError("price file has no rows")
    duplicates = frame.duplicated(["security_id", "date"], keep=False)
    if duplicates.any():
        sample = frame.loc[duplicates, ["security_id", "date"]].head(5)
        raise DataContractError(f"duplicate price bars:\n{sample}")
    for column in ("open_raw", "high_raw", "low_raw", "close_raw"):
        if (frame[column] <= 0).any() or frame[column].isna().any():
            raise DataContractError(f"{column} must be positive on every bar")
    if (frame["volume_raw"] < 0).any() or frame["volume_raw"].isna().any():
        raise DataContractError("volume_raw must be zero or positive")
    if (frame["split_factor"] <= 0).any() or frame["split_factor"].isna().any():
        raise DataContractError("split_factor must be positive; supply 1 when there is no split")
    if (frame["dividend"] < 0).any() or frame["dividend"].isna().any():
        raise DataContractError("dividend must be zero or positive")
    high_ok = frame["high_raw"] + 1e-9 >= frame[["open_raw", "close_raw", "low_raw"]].max(axis=1)
    low_ok = frame["low_raw"] - 1e-9 <= frame[["open_raw", "close_raw", "high_raw"]].min(axis=1)
    if not bool(high_ok.all() and low_ok.all()):
        raise DataContractError("high_raw/low_raw are inconsistent with open_raw/close_raw")
    return frame


def validate_security_master(frame: pd.DataFrame) -> pd.DataFrame:
    require_columns(frame, SECURITY_MASTER_COLUMNS, "security master")
    if frame.empty:
        raise DataContractError("security master has no rows")
    if (frame["valid_to"] < frame["valid_from"]).any():
        raise DataContractError("security master has valid_to before valid_from")
    return frame


def assert_output_not_raw(path: object, raw_root: object) -> None:
    """Refuse writes under data/raw. Derived artifacts belong in interim, processed, or runs."""
    from pathlib import Path

    target = Path(str(path)).resolve()
    raw = Path(str(raw_root)).resolve()
    if target == raw or raw in target.parents:
        raise DataContractError(f"refusing to write under immutable raw data: {target}")
