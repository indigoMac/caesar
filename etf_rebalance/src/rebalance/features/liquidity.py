"""Pre-signal liquidity and return features.

ADV20 is the mean of raw close * raw volume over the 20 completed sessions
from t-25 through t-6, where t is the last session whose close is at or
before `signal_known_at`. Sessions after the signal are never read. A short
history produces a missing feature and a diagnostic, not a shorter window.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from rebalance.calendar import TradingCalendar, as_utc
from rebalance.errors import DataContractError, MissingPriceError
from rebalance.events.timeline import assert_no_lookahead
from rebalance.features.returns import close_to_close_total_return, index_prices


def compute_liquidity_features(
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    calendar: TradingCalendar,
    benchmark_security_id: str,
    fundamentals: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    indexed = index_prices(prices)
    if benchmark_security_id not in indexed:
        raise DataContractError(f"benchmark {benchmark_security_id} is missing from prices")
    benchmark = indexed[benchmark_security_id]
    rows: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    for signal in signals.to_dict(orient="records"):
        security_id = str(signal["security_id"])
        info_session = calendar.last_completed_session(signal["signal_known_at"])
        info_close = calendar.session_close(info_session)
        if info_close > as_utc(signal["signal_known_at"]):
            raise DataContractError("info session close is after the signal")
        feature: dict[str, object] = {
            "rebalance_id": signal["rebalance_id"],
            "security_id": security_id,
            "action": signal["action"],
            "signal_known_at": signal["signal_known_at"],
            "earliest_trade_at": signal["earliest_trade_at"],
            "info_session": info_session,
        }
        stock = indexed.get(security_id)
        adv_asof = None
        _fill_missing_market_features(feature)
        if stock is None:
            diagnostics.append(_diag(signal, "missing_security_prices"))
        else:
            adv_asof = _assign_window_features(feature, stock, benchmark, calendar, info_session, signal, diagnostics)
        fundamental_asof = _attach_fundamentals(feature, fundamentals, security_id, signal["signal_known_at"])
        asofs = [info_close]
        if adv_asof is not None:
            asofs.append(adv_asof)
        if fundamental_asof is not None:
            asofs.append(fundamental_asof)
        feature_asof = max(asofs)
        assert_no_lookahead(feature_asof, signal["signal_known_at"], signal["earliest_trade_at"])
        feature["feature_asof"] = feature_asof
        rows.append(feature)
    return pd.DataFrame(rows), pd.DataFrame(diagnostics)


def _assign_window_features(
    feature: dict[str, object],
    stock: pd.DataFrame,
    benchmark: pd.DataFrame,
    calendar: TradingCalendar,
    info_session: pd.Timestamp,
    signal: dict[str, object],
    diagnostics: list[dict[str, object]],
) -> pd.Timestamp | None:
    """Fill each pre-signal feature independently so a short sample can still have ADV20."""
    adv_asof = None
    try:
        adv_start = calendar.shift_session(info_session, -25)
        adv_end = calendar.shift_session(info_session, -6)
        if adv_end > info_session:
            raise DataContractError("ADV window extends past the signal session")
        previous = calendar.shift_session(adv_start, -1)
        adv_sessions = list(calendar.sessions_between(adv_start, adv_end))
        if len(adv_sessions) != 20:
            raise DataContractError("ADV window is not 20 sessions")
        adv20, returns = _dollar_volume_and_returns(stock, adv_sessions, previous)
        benchmark_returns = _benchmark_returns(benchmark, adv_sessions, previous)
        feature["adv20"] = adv20
        feature["realized_vol_20_daily"] = _sample_std(returns)
        feature["beta_pre_20"] = _beta(returns, benchmark_returns)
        adv_asof = calendar.session_close(adv_end)
        feature["adv_feature_asof"] = adv_asof
    except (DataContractError, MissingPriceError) as exc:
        diagnostics.append(_diag(signal, "adv20_unavailable", str(exc)))
    try:
        feature["adv60"] = _adv_over(stock, calendar, info_session, lookback_end=6, length=60)
    except (DataContractError, MissingPriceError) as exc:
        diagnostics.append(_diag(signal, "adv60_unavailable", str(exc)))
    for sessions, column in ((5, "pre_event_return_5"), (20, "pre_event_return_20")):
        try:
            feature[column] = _span_return(stock, calendar, info_session, sessions)
        except (DataContractError, MissingPriceError) as exc:
            diagnostics.append(_diag(signal, f"{column}_unavailable", str(exc)))
    return adv_asof


def _dollar_volume_and_returns(
    frame: pd.DataFrame,
    sessions: list[pd.Timestamp],
    previous: pd.Timestamp,
) -> tuple[float, list[float]]:
    if any(session not in frame.index for session in sessions) or previous not in frame.index:
        raise MissingPriceError("ADV window is missing raw bars")
    dollar = [float(frame.loc[session, "close_raw"]) * float(frame.loc[session, "volume_raw"]) for session in sessions]
    returns: list[float] = []
    previous_close = float(frame.loc[previous, "close_raw"])
    for session in sessions:
        row = frame.loc[session]
        returns.append(
            close_to_close_total_return(
                previous_close,
                float(row["close_raw"]),
                float(row["split_factor"]),
                float(row["dividend"]),
            )
        )
        previous_close = float(row["close_raw"])
    return float(np.mean(dollar)), returns


def _benchmark_returns(
    frame: pd.DataFrame,
    sessions: list[pd.Timestamp],
    previous: pd.Timestamp,
) -> list[float]:
    _, returns = _dollar_volume_and_returns(frame, sessions, previous)
    return returns


def _adv_over(
    frame: pd.DataFrame,
    calendar: TradingCalendar,
    info_session: pd.Timestamp,
    *,
    lookback_end: int,
    length: int,
) -> float:
    end = calendar.shift_session(info_session, -lookback_end)
    start = calendar.shift_session(end, -(length - 1))
    sessions = list(calendar.sessions_between(start, end))
    if len(sessions) != length:
        raise DataContractError(f"expected {length} sessions in the ADV window")
    if any(session not in frame.index for session in sessions):
        raise MissingPriceError(f"ADV{length} window is missing raw bars")
    dollar = [float(frame.loc[session, "close_raw"]) * float(frame.loc[session, "volume_raw"]) for session in sessions]
    return float(np.mean(dollar))


def _span_return(frame: pd.DataFrame, calendar: TradingCalendar, info_session: pd.Timestamp, sessions: int) -> float:
    """Total return from the close `sessions` before info_session through info_session's close."""
    start = calendar.shift_session(info_session, -sessions)
    end_sessions = list(calendar.sessions_between(calendar.shift_session(start, 1), info_session))
    if start not in frame.index or any(session not in frame.index for session in end_sessions):
        raise MissingPriceError(f"pre-event {sessions}-session return is missing bars")
    previous_close = float(frame.loc[start, "close_raw"])
    wealth = 1.0
    for session in end_sessions:
        row = frame.loc[session]
        wealth *= (float(row["close_raw"]) + float(row["dividend"])) / (previous_close / float(row["split_factor"]))
        previous_close = float(row["close_raw"])
    return wealth - 1.0


def _sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return float("nan")
    return float(np.std(np.asarray(values, dtype=float), ddof=1))


def _beta(stock: list[float], benchmark: list[float]) -> float:
    if len(stock) != len(benchmark) or len(stock) < 2:
        return float("nan")
    market = np.asarray(benchmark, dtype=float)
    asset = np.asarray(stock, dtype=float)
    variance = float(np.var(market, ddof=1))
    if variance == 0.0 or np.isnan(variance):
        return float("nan")
    covariance = float(np.cov(asset, market, ddof=1)[0, 1])
    return covariance / variance


def _attach_fundamentals(
    feature: dict[str, object],
    fundamentals: pd.DataFrame | None,
    security_id: str,
    signal_known_at: pd.Timestamp,
) -> pd.Timestamp | None:
    feature["market_cap"] = np.nan
    feature["delta_weight"] = np.nan
    feature["estimated_flow"] = np.nan
    feature["fundamental_asof"] = pd.NaT
    if fundamentals is None or fundamentals.empty:
        return None
    known = as_utc(signal_known_at)
    rows = fundamentals[fundamentals["security_id"] == security_id].copy()
    if rows.empty:
        return None
    rows = rows.loc[pd.to_datetime(rows["asof"], utc=True) <= known]
    if rows.empty:
        return None
    latest = rows.sort_values("asof").iloc[-1]
    feature["market_cap"] = latest.get("market_cap", np.nan)
    feature["delta_weight"] = latest.get("delta_weight", np.nan)
    feature["estimated_flow"] = latest.get("estimated_flow", np.nan)
    fundamental_asof = as_utc(latest["asof"])
    feature["fundamental_asof"] = fundamental_asof
    return fundamental_asof


def _fill_missing_market_features(feature: dict[str, object]) -> None:
    for column in (
        "adv20",
        "adv60",
        "realized_vol_20_daily",
        "beta_pre_20",
        "pre_event_return_5",
        "pre_event_return_20",
        "adv_feature_asof",
    ):
        feature[column] = np.nan


def _diag(signal: dict[str, object], reason: str, detail: str | None = None) -> dict[str, object]:
    payload: dict[str, object] = {
        "rebalance_id": signal["rebalance_id"],
        "security_id": signal["security_id"],
        "reason": reason,
    }
    if detail is not None:
        payload["detail"] = detail
    return payload
