"""Build synthetic fixtures for the research pipeline.

These files are not FTSE Russell history and they are not a price vendor
extract. They exist so the event-study code can be run and audited before
licensed point-in-time data is available.

TODO: Replace every file this script writes with a normalized extract of
real public announcements, a security master, and raw prices.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from rebalance.calendar import TradingCalendar
from rebalance.data.prices import load_prices
from rebalance.data.russell import load_announcements
from rebalance.data.securities import load_security_master

ROOT = Path(__file__).resolve().parents[1]
ADDS = ["A01", "A02", "A03", "A04", "A05"]
DELETES = ["D01", "D02", "D03", "D04", "D05"]
NAMES = ADDS + DELETES + ["X01", "BENCH"]
SOURCE_URL = "https://example.invalid/synthetic-reconstitution"
RETRIEVED_AT = "2024-08-15T12:00:00-04:00"


def main() -> None:
    calendar = TradingCalendar("XNYS", "America/New_York")
    split_session = calendar.require_session("2024-02-09")
    dividend_session = calendar.require_session("2024-03-15")
    sessions = list(calendar.sessions_between("2022-11-01", "2024-08-30"))
    _write_security_master()
    _write_announcements()
    _write_fundamentals()
    _write_prices(sessions, split_session, dividend_session)
    _write_corporate_actions(split_session, dividend_session)
    master = load_security_master(ROOT / "data/raw/russell/sample_security_master.csv")
    load_announcements(ROOT / "data/raw/russell/sample_announcements.csv", master)
    load_prices(ROOT / "data/raw/prices/sample_prices.csv", calendar=calendar)
    print("wrote synthetic fixtures")


def _write_security_master() -> None:
    rows = [
        ("A01", "OLD_A01", "2020-01-01", "2022-12-31", "Alpha One"),
        ("A01", "A01", "2023-01-01", "2200-01-01", "Alpha One"),
    ]
    for security_id in NAMES:
        if security_id == "A01":
            continue
        company = "Synthetic Benchmark" if security_id == "BENCH" else f"Synthetic {security_id}"
        rows.append((security_id, security_id, "2020-01-01", "2200-01-01", company))
    frame = pd.DataFrame(rows, columns=["security_id", "ticker", "valid_from", "valid_to", "company_name"])
    frame["data_status"] = "SYNTHETIC"
    _write(frame, ROOT / "data/raw/russell/sample_security_master.csv")


def _write_announcements() -> None:
    rows: list[dict[str, object]] = []
    for security_id in ADDS + DELETES:
        action = "ADD" if security_id in ADDS else "DELETE"
        rows.append(_event("2023-06", "2023-05-19T17:00:00-04:00", "2023-06-23", security_id, action, 1))
        rows.append(_event("2024-06", "2024-05-24T17:00:00-04:00", "2024-06-28", security_id, action, 1))
    rows.append(_event("2024-06", "2024-05-24T17:00:00-04:00", "2024-06-28", "X01", "ADD", 1))
    rows.append(_event("2024-06", "2024-05-27T18:00:00-04:00", "2024-06-28", "X01", "NONE", 2))
    rows.append(_event("2024-06", "2024-05-31T17:00:00-04:00", "2024-06-28", "A01", "ADD", 2))
    _write(pd.DataFrame(rows), ROOT / "data/raw/russell/sample_announcements.csv")


def _event(
    rebalance_id: str,
    published_at: str,
    effective_close: str,
    ticker: str,
    action: str,
    version: int,
) -> dict[str, object]:
    return {
        "published_at": published_at,
        "effective_close": effective_close,
        "ticker": ticker,
        "company_name": f"Synthetic {ticker}",
        "index_name": "RUSSELL_2000",
        "action": action,
        "version": version,
        "rebalance_id": rebalance_id,
        "source_url": SOURCE_URL,
        "retrieved_at": RETRIEVED_AT,
        "data_status": "SYNTHETIC",
    }


def _write_fundamentals() -> None:
    rows: list[dict[str, object]] = []
    for offset, names in ((1, ADDS), (1, DELETES)):
        del offset
        for index, security_id in enumerate(names, start=1):
            sign = 1.0 if security_id in ADDS else -1.0
            for asof in ("2023-05-01T16:00:00-04:00", "2024-05-01T16:00:00-04:00"):
                rows.append(
                    {
                        "security_id": security_id,
                        "asof": asof,
                        "market_cap": 1_000_000_000.0 * index,
                        "delta_weight": sign * 0.0001 * index,
                        "estimated_flow": "",
                        "data_status": "SYNTHETIC",
                    }
                )
    rows.append(
        {
            "security_id": "A01",
            "asof": "2024-06-01T16:00:00-04:00",
            "market_cap": 9_000_000_000_000.0,
            "delta_weight": 0.5,
            "estimated_flow": 1_000_000_000.0,
            "data_status": "SYNTHETIC_LOOKAHEAD_TRAP",
        }
    )
    _write(pd.DataFrame(rows), ROOT / "data/raw/russell/sample_fundamentals.csv")


def _write_prices(sessions: list[pd.Timestamp], split_session: pd.Timestamp, dividend_session: pd.Timestamp) -> None:
    frames = [
        _path(security_id, index, sessions, split_session, dividend_session)
        for index, security_id in enumerate(NAMES)
    ]
    _write(pd.concat(frames, ignore_index=True), ROOT / "data/raw/prices/sample_prices.csv")


def _path(
    security_id: str,
    index: int,
    sessions: list[pd.Timestamp],
    split_session: pd.Timestamp,
    dividend_session: pd.Timestamp,
) -> pd.DataFrame:
    rng = np.random.default_rng(42 + index)
    previous: float | None = None
    volume = 1_000_000.0
    rows: list[dict[str, object]] = []
    for session in sessions:
        split = 2.0 if security_id == "A01" and session == split_session else 1.0
        dividend = 0.5 if security_id == "D01" and session == dividend_session else 0.0
        if previous is None:
            open_price = 40.0
            close = 40.0
        else:
            shock = float(rng.normal(0.0, 0.01))
            theoretical = previous / split
            open_price = max(theoretical - dividend, 1.0)
            close = max((1.0 + shock) * theoretical - dividend, 1.0)
        if split != 1.0:
            volume *= split
        high = max(open_price, close) * 1.001
        low = min(open_price, close) * 0.999
        rows.append(
            {
                "security_id": security_id,
                "date": session.strftime("%Y-%m-%d"),
                "open_raw": open_price,
                "high_raw": high,
                "low_raw": low,
                "close_raw": close,
                "volume_raw": volume,
                "split_factor": split,
                "dividend": dividend,
                "data_status": "SYNTHETIC",
            }
        )
        previous = close
    return pd.DataFrame(rows)


def _write_corporate_actions(split_session: pd.Timestamp, dividend_session: pd.Timestamp) -> None:
    frame = pd.DataFrame(
        [
            {
                "security_id": "A01",
                "date": split_session.strftime("%Y-%m-%d"),
                "split_factor": 2.0,
                "dividend": 0.0,
                "data_status": "SYNTHETIC",
            },
            {
                "security_id": "D01",
                "date": dividend_session.strftime("%Y-%m-%d"),
                "split_factor": 1.0,
                "dividend": 0.5,
                "data_status": "SYNTHETIC",
            },
        ]
    )
    _write(frame, ROOT / "data/raw/corporate_actions/sample_corporate_actions.csv")


def _write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


if __name__ == "__main__":
    main()
