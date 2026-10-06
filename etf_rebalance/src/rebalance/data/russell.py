"""Normalize public index-announcement files into the research schema.

The loader does not know FTSE Russell's delivery format. Feed it a CSV that
already uses the columns below, or adapt the alias map when a real extract
arrives. Missing history must stay missing.

TODO: Map licensed FTSE Russell preliminary, updated, and final reconstitution
files into this schema. Do not invent historical additions, deletions, or
effective dates.

Required CSV columns:
    published_at, effective_close, ticker, company_name, index_name, action,
    version, rebalance_id

`published_at` and `retrieved_at` must include a timezone offset unless the
caller explicitly passes `naive_timestamps_tz`. `effective_close` is a
calendar date. `action` is ADD, DELETE, or NONE.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from rebalance.calendar import as_utc
from rebalance.data.securities import resolve_security_id
from rebalance.data.validation import validate_announcements
from rebalance.errors import DataContractError

COLUMN_ALIASES = {
    "ticker": "ticker_published",
    "company_name": "company_name_published",
    "company": "company_name_published",
}

HASH_FIELDS = (
    "rebalance_id",
    "ticker_published",
    "company_name_published",
    "index_name",
    "action",
    "published_at",
    "effective_close",
    "version",
    "source_url",
)


def _parse_instant(value: object, field: str, naive_tz: str | None) -> pd.Timestamp:
    if value is None or pd.isna(value):
        raise DataContractError(f"{field} is missing")
    parsed = pd.Timestamp(value)
    if parsed.tzinfo is None:
        if naive_tz is None:
            raise DataContractError(
                f"{field} value {value!r} has no timezone. Add an offset or pass "
                "naive_timestamps_tz explicitly. The loader will not assume one."
            )
        parsed = parsed.tz_localize(naive_tz)
    return as_utc(parsed)


def _parse_effective_close(value: object) -> pd.Timestamp:
    if value is None or pd.isna(value):
        raise DataContractError("effective_close is missing")
    parsed = pd.Timestamp(value)
    if parsed.tzinfo is not None:
        raise DataContractError("effective_close must be a calendar date, not an instant")
    normalized = parsed.normalize()
    if parsed != normalized:
        raise DataContractError("effective_close must be a calendar date")
    return normalized


def _canonical_hash(row: pd.Series) -> str:
    payload = {field: None if pd.isna(row[field]) else str(row[field]) for field in HASH_FIELDS}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_announcements(
    path: Path,
    security_master: pd.DataFrame,
    *,
    naive_timestamps_tz: str | None = None,
    timezone: str = "America/New_York",
) -> pd.DataFrame:
    """Read a public-announcement CSV and attach point-in-time security ids."""
    frame = pd.read_csv(path)
    frame = frame.rename(columns={key: value for key, value in COLUMN_ALIASES.items() if key in frame.columns})
    required = [
        "published_at",
        "effective_close",
        "ticker_published",
        "company_name_published",
        "index_name",
        "action",
        "version",
        "rebalance_id",
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise DataContractError(f"announcement CSV is missing columns: {missing}")
    if "retrieved_at" not in frame.columns:
        raise DataContractError(
            "announcement CSV must include retrieved_at. The loader does not default it to now()."
        )
    if "source_url" not in frame.columns:
        raise DataContractError("announcement CSV must include source_url")

    frame["action"] = frame["action"].astype(str).str.strip().str.upper()
    frame["version"] = frame["version"].astype(int)
    frame["published_at"] = frame["published_at"].map(
        lambda value: _parse_instant(value, "published_at", naive_timestamps_tz)
    )
    frame["retrieved_at"] = frame["retrieved_at"].map(
        lambda value: _parse_instant(value, "retrieved_at", naive_timestamps_tz)
    )
    frame["effective_close"] = frame["effective_close"].map(_parse_effective_close)
    for column in ("rebalance_id", "ticker_published", "company_name_published", "index_name", "source_url"):
        frame[column] = frame[column].astype(str)

    if "source_hash" not in frame.columns:
        frame["source_hash"] = frame.apply(_canonical_hash, axis=1)
    else:
        missing_hash = frame["source_hash"].isna() | (frame["source_hash"].astype(str).str.len() == 0)
        frame.loc[missing_hash, "source_hash"] = frame.loc[missing_hash].apply(_canonical_hash, axis=1)
        frame["source_hash"] = frame["source_hash"].astype(str)

    supplied_ids = frame["security_id"].astype(str) if "security_id" in frame.columns else None
    resolved: list[str] = []
    for row in frame.itertuples(index=False):
        security_id = resolve_security_id(
            security_master,
            row.ticker_published,
            row.published_at,
            timezone=timezone,
        )
        resolved.append(security_id)
    if supplied_ids is not None:
        for published, resolved_id, supplied in zip(frame["ticker_published"], resolved, supplied_ids, strict=True):
            if supplied not in {"", "nan", "None"} and supplied != resolved_id:
                raise DataContractError(
                    f"security_id {supplied} does not match ticker {published} -> {resolved_id}"
                )
    frame["security_id"] = resolved
    validate_announcements(frame)
    return frame.sort_values(
        ["rebalance_id", "index_name", "security_id", "published_at", "version"]
    ).reset_index(drop=True)
