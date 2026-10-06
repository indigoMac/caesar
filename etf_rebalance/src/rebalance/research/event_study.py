"""Event-study panel and window cumulative abnormal returns.

`stock_return` on the entry session is the open-to-close return, because the
position is entered at that open. Every other session in the panel uses
close-to-close total return. The benchmark uses the same basis on the same
session. Abnormal return is the difference. Additions and deletions are not
sign-flipped here; the portfolio layer applies the deletion sign.
"""

from __future__ import annotations

from dataclasses import dataclass

import duckdb
import pandas as pd

from rebalance.calendar import TradingCalendar
from rebalance.config import WindowSpec
from rebalance.errors import DataContractError, MissingPriceError
from rebalance.events.timeline import assert_no_lookahead
from rebalance.features.returns import index_prices, session_total_return


@dataclass(frozen=True)
class EventStudyOutputs:
    panel: pd.DataFrame
    diagnostics: pd.DataFrame
    window_cars: pd.DataFrame


def build_event_panel(
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    calendar: TradingCalendar,
    benchmark_security_id: str,
    *,
    pre_sessions: int = 5,
    post_sessions: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the session panel around each tradable announcement."""
    if signals.empty:
        raise DataContractError("no tradable signals for the event panel")
    if pre_sessions < 0 or post_sessions < 0:
        raise DataContractError("pre_sessions and post_sessions cannot be negative")
    indexed = index_prices(prices)
    if benchmark_security_id not in indexed:
        raise DataContractError(f"benchmark {benchmark_security_id} is missing from prices")
    benchmark = indexed[benchmark_security_id]
    rows: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    for signal in signals.to_dict(orient="records"):
        assert_no_lookahead(signal["feature_asof"], signal["signal_known_at"], signal["earliest_trade_at"])
        security_id = str(signal["security_id"])
        if security_id not in indexed:
            diagnostics.append({**_diagnostic_key(signal), "reason": "missing_security_prices"})
            continue
        stock = indexed[security_id]
        entry = calendar.require_session(signal["entry_session"])
        rebalance = calendar.require_session(signal["effective_close"])
        start = min(entry, calendar.shift_session(rebalance, -pre_sessions))
        end = calendar.shift_session(rebalance, post_sessions)
        sessions = list(calendar.sessions_between(start, end))
        cumulative = 0.0
        cumulative_valid = True
        missing_dates: list[str] = []
        for session in sessions:
            basis = "open_to_close" if session == entry else "close_to_close"
            stock_return, stock_problem = _optional_return(stock, session, basis, calendar)
            try:
                benchmark_return = session_total_return(benchmark, session, basis, calendar)
            except MissingPriceError as exc:
                raise DataContractError(
                    f"benchmark {benchmark_security_id} is missing a {basis} return on {session.date()}"
                ) from exc
            if stock_return is None:
                abnormal = None
                cumulative_valid = False
                cumulative_value = None
                missing_dates.append(f"{session.date().isoformat()}:{stock_problem}")
            else:
                abnormal = stock_return - benchmark_return
                if cumulative_valid:
                    cumulative += abnormal
                    cumulative_value = cumulative
                else:
                    cumulative_value = None
            rows.append(
                {
                    "rebalance_id": signal["rebalance_id"],
                    "security_id": security_id,
                    "index_name": signal["index_name"],
                    "ticker_published": signal["ticker_published"],
                    "action": signal["action"],
                    "date": session,
                    "days_from_announcement": calendar.session_distance(entry, session),
                    "days_from_rebalance": calendar.session_distance(rebalance, session),
                    "stock_return": stock_return,
                    "benchmark_return": benchmark_return,
                    "abnormal_return": abnormal,
                    "cumulative_abnormal_return": cumulative_value,
                    "signal_known_at": signal["signal_known_at"],
                    "feature_asof": signal["feature_asof"],
                    "earliest_trade_at": signal["earliest_trade_at"],
                    "entry_session": entry,
                    "effective_close": rebalance,
                    "return_basis": basis,
                }
            )
        if missing_dates:
            diagnostics.append(
                {
                    **_diagnostic_key(signal),
                    "reason": "missing_price_bars",
                    "detail": ";".join(missing_dates),
                }
            )
    panel = pd.DataFrame(rows)
    if panel.empty:
        raise DataContractError("event panel is empty")
    panel = panel.sort_values(["rebalance_id", "security_id", "date"]).reset_index(drop=True)
    return panel, pd.DataFrame(diagnostics)


def window_cars(
    panel: pd.DataFrame,
    calendar: TradingCalendar,
    windows: list[WindowSpec],
) -> pd.DataFrame:
    """Sum abnormal returns over each configured trading-day window.

    A window with any missing session or missing abnormal return is NaN.
    Partial sums are not reported as if the window were complete.
    """
    if panel.empty:
        raise DataContractError("cannot compute window CARs on an empty panel")
    identity = (
        panel.groupby(["rebalance_id", "security_id"], sort=False)
        .agg(
            action=("action", "first"),
            index_name=("index_name", "first"),
            entry_session=("entry_session", "first"),
            effective_close=("effective_close", "first"),
        )
        .reset_index()
    )
    records: list[dict[str, object]] = []
    for event in identity.to_dict(orient="records"):
        event_rows = panel[
            (panel["rebalance_id"] == event["rebalance_id"]) & (panel["security_id"] == event["security_id"])
        ]
        for window in windows:
            start = calendar.shift_session(_anchor(event, window.start_anchor), window.start_offset)
            end = calendar.shift_session(_anchor(event, window.end_anchor), window.end_offset)
            car, status = _sum_window(event_rows, start, end, calendar)
            records.append(
                {
                    "rebalance_id": event["rebalance_id"],
                    "security_id": event["security_id"],
                    "action": event["action"],
                    "index_name": event["index_name"],
                    "effective_close": event["effective_close"],
                    "window": window.name,
                    "window_start": start,
                    "window_end": end,
                    "car": car,
                    "window_status": status,
                }
            )
    return pd.DataFrame(records)


def mean_abnormal_return_by_offset(panel: pd.DataFrame) -> pd.DataFrame:
    """Average abnormal returns by action and rebalance-relative session.

    This is a descriptive cross-section. It is not an independence-robust
    inference result; use the rebalance bootstrap for intervals.
    """
    connection = duckdb.connect(database=":memory:")
    try:
        connection.register("panel", panel)
        return connection.execute(
            """
            SELECT action,
                   days_from_rebalance,
                   AVG(abnormal_return) AS mean_abnormal_return,
                   COUNT(abnormal_return) AS n_names,
                   COUNT(DISTINCT rebalance_id) AS n_rebalances
            FROM panel
            GROUP BY action, days_from_rebalance
            ORDER BY action, days_from_rebalance
            """
        ).df()
    finally:
        connection.close()


def car_paths(
    panel: pd.DataFrame,
    *,
    align: str,
    start_offset: int,
    end_offset: int,
    stop_at_rebalance: bool = False,
) -> pd.DataFrame:
    """Event-equal-weighted mean CAR path.

    Each event's CAR is the sum of its own abnormal returns. Each rebalance
    then contributes one mean, and those rebalance means are averaged. Events
    with a hole in the requested offset range are omitted from the whole path
    rather than filled.
    """
    if align not in {"days_from_announcement", "days_from_rebalance"}:
        raise DataContractError(f"unsupported alignment {align}")
    per_event: list[dict[str, object]] = []
    grouped = panel.groupby(["rebalance_id", "security_id"], sort=False)
    for (rebalance_id, security_id), group in grouped:
        ordered = group.sort_values(align)
        action = str(ordered["action"].iloc[0])
        by_offset = ordered.drop_duplicates(align).set_index(align)
        running = 0.0
        points: list[tuple[int, float]] = []
        complete = True
        for offset in range(start_offset, end_offset + 1):
            if stop_at_rebalance and align == "days_from_announcement":
                if offset not in by_offset.index:
                    break
                if int(by_offset.loc[offset, "days_from_rebalance"]) > 0:
                    break
            if offset not in by_offset.index or pd.isna(by_offset.loc[offset, "abnormal_return"]):
                complete = False
                break
            running += float(by_offset.loc[offset, "abnormal_return"])
            points.append((offset, running))
        if not complete or not points:
            continue
        for offset, value in points:
            per_event.append(
                {
                    "rebalance_id": rebalance_id,
                    "security_id": security_id,
                    "action": action,
                    "offset": offset,
                    "car": value,
                }
            )
    detail = pd.DataFrame(per_event)
    if detail.empty:
        return pd.DataFrame(columns=["action", "offset", "mean_car", "n_rebalances", "n_names"])
    by_rebalance = (
        detail.groupby(["action", "rebalance_id", "offset"], sort=False)["car"].mean().reset_index(name="event_mean")
    )
    summary = (
        by_rebalance.groupby(["action", "offset"], sort=False)
        .agg(mean_car=("event_mean", "mean"), n_rebalances=("rebalance_id", "nunique"))
        .reset_index()
    )
    counts = detail.groupby(["action", "offset"], sort=False)["security_id"].nunique().reset_index(name="n_names")
    return summary.merge(counts, on=["action", "offset"], how="left").sort_values(["action", "offset"])


def _anchor(event: dict[str, object], anchor: str) -> pd.Timestamp:
    if anchor == "entry":
        return pd.Timestamp(event["entry_session"])
    if anchor == "rebalance":
        return pd.Timestamp(event["effective_close"])
    raise DataContractError(f"unknown window anchor {anchor}")


def _sum_window(
    event_rows: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    calendar: TradingCalendar,
) -> tuple[float | None, str]:
    if end < start:
        return None, "empty_window"
    expected = list(calendar.sessions_between(start, end))
    present = event_rows[event_rows["date"].isin(expected)]
    if len(present) != len(expected):
        return None, "incomplete_window"
    if present["abnormal_return"].isna().any():
        return None, "missing_return"
    return float(present["abnormal_return"].sum()), "ok"


def _optional_return(
    frame: pd.DataFrame,
    session: pd.Timestamp,
    basis: str,
    calendar: TradingCalendar,
) -> tuple[float | None, str | None]:
    try:
        return session_total_return(frame, session, basis, calendar), None
    except MissingPriceError as exc:
        return None, str(exc)


def _diagnostic_key(signal: dict[str, object]) -> dict[str, object]:
    return {
        "rebalance_id": signal["rebalance_id"],
        "security_id": signal["security_id"],
        "action": signal["action"],
    }
