"""Buy-and-hold portfolio accounting for one reconstitution at a time.

Official returns are sum(signed_weight * underlying total return), minus
entry and exit costs on absolute weight. This is not a daily-rebalanced book.

Short sales are marked `theoretical` unless a point-in-time borrow file says
the name was locatable. Missing borrow data is not treated as availability.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from rebalance.calendar import TradingCalendar, as_utc
from rebalance.errors import DataContractError
from rebalance.events.timeline import assert_fill_time
from rebalance.features.returns import holding_period_total_return, index_prices
from rebalance.portfolio.costs import CostContext, CostModel, net_return
from rebalance.portfolio.sizing import size_positions


@dataclass
class SimulationResult:
    trades: pd.DataFrame
    rebalance_returns: pd.DataFrame
    diagnostics: pd.DataFrame
    contains_theoretical_shorts: bool
    notes: list[str] = field(default_factory=list)


def borrow_status_for(
    security_id: str,
    signal_known_at: pd.Timestamp,
    borrow: pd.DataFrame | None,
) -> str:
    if borrow is None:
        return "theoretical"
    known = as_utc(signal_known_at)
    rows = borrow[borrow["security_id"] == security_id]
    if rows.empty:
        return "theoretical_no_record"
    visible = rows.loc[pd.to_datetime(rows["asof"], utc=True) <= known]
    if visible.empty:
        return "theoretical_no_record"
    latest = visible.sort_values("asof").iloc[-1]
    if bool(latest["borrow_available"]):
        return "borrow_located"
    return "unavailable"


def simulate_portfolio(
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    calendar: TradingCalendar,
    *,
    mode: str,
    scheme: str,
    cost_model: CostModel,
    max_name_weight: float,
    gross_exposure_cap: float,
    benchmark_security_id: str,
    borrow: pd.DataFrame | None = None,
    portfolio_nav: float | None = None,
    max_adv_participation: float | None = None,
) -> SimulationResult:
    indexed = index_prices(prices)
    if benchmark_security_id not in indexed:
        raise DataContractError(f"benchmark {benchmark_security_id} is missing from prices")
    trades: list[dict[str, object]] = []
    books: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    theoretical = False
    for rebalance_id, group in signals.groupby("rebalance_id", sort=False):
        kept, statuses, dropped = _apply_borrow_filter(group, mode, borrow)
        diagnostics.extend(dropped)
        if kept.empty:
            diagnostics.append({"rebalance_id": rebalance_id, "reason": "no_names_after_borrow_filter", "mode": mode})
            continue
        sizing = size_positions(
            kept,
            mode=mode,
            scheme=scheme,
            max_name_weight=max_name_weight,
            gross_exposure_cap=gross_exposure_cap,
            benchmark_security_id=benchmark_security_id,
            portfolio_nav=portfolio_nav,
            max_adv_participation=max_adv_participation,
        )
        if sizing.weights.empty or sizing.achieved_gross == 0.0:
            diagnostics.append(
                {
                    "rebalance_id": rebalance_id,
                    "reason": "no_positions",
                    "mode": mode,
                    "notes": ";".join(sizing.notes),
                }
            )
            continue
        by_id = kept.drop_duplicates("security_id").set_index("security_id")
        hedge_context = _hedge_context(mode, sizing.weights, by_id, benchmark_security_id)
        position_gross_total = 0.0
        position_net_total = 0.0
        for security_id, weight in sizing.weights.items():
            weight = float(weight)
            if weight == 0.0:
                continue
            if security_id == benchmark_security_id and mode == "long_additions_hedged":
                if hedge_context is None:
                    raise DataContractError("hedge context was not resolved")
                security_id = str(security_id)
                frame = indexed[security_id]
                underlying = holding_period_total_return(
                    frame,
                    hedge_context["entry_session"],
                    hedge_context["effective_close"],
                    calendar,
                )
                assert_fill_time(
                    hedge_context["earliest_trade_at"],
                    hedge_context["earliest_trade_at"],
                    hedge_context["signal_known_at"],
                    hedge_context["feature_asof"],
                )
                side = "hedge"
                borrow_status = "not_applicable"
                signal_known_at = hedge_context["signal_known_at"]
                feature_asof = hedge_context["feature_asof"]
                earliest = hedge_context["earliest_trade_at"]
                entry_session = hedge_context["entry_session"]
                exit_session = hedge_context["effective_close"]
                action = "HEDGE"
            else:
                row = by_id.loc[security_id]
                assert_fill_time(row["earliest_trade_at"], row["earliest_trade_at"], row["signal_known_at"], row["feature_asof"])
                if str(security_id) not in indexed:
                    raise DataContractError(f"missing prices for {security_id}")
                underlying = holding_period_total_return(
                    indexed[str(security_id)],
                    row["entry_session"],
                    row["effective_close"],
                    calendar,
                )
                side = "long" if weight > 0 else "short"
                borrow_status = statuses.get(str(security_id), "not_applicable")
                if side == "short":
                    if borrow_status == "not_applicable":
                        borrow_status = "theoretical"
                    if borrow_status.startswith("theoretical"):
                        theoretical = True
                signal_known_at = row["signal_known_at"]
                feature_asof = row["feature_asof"]
                earliest = row["earliest_trade_at"]
                entry_session = row["entry_session"]
                exit_session = row["effective_close"]
                action = row["action"]
            entry_cost = cost_model.one_way_cost(CostContext(security_id=str(security_id), leg="entry"))
            exit_cost = cost_model.one_way_cost(CostContext(security_id=str(security_id), leg="exit"))
            gross_contribution = weight * underlying
            scaled_entry = abs(weight) * entry_cost
            scaled_exit = abs(weight) * exit_cost
            net_contribution = net_return(gross_contribution, entry_cost=scaled_entry, exit_cost=scaled_exit)
            position_gross_total += gross_contribution
            position_net_total += net_contribution
            trades.append(
                {
                    "rebalance_id": rebalance_id,
                    "security_id": security_id,
                    "action": action,
                    "side": side,
                    "weight": weight,
                    "underlying_return": underlying,
                    "gross_contribution": gross_contribution,
                    "entry_cost": entry_cost,
                    "exit_cost": exit_cost,
                    "cost_drag": scaled_entry + scaled_exit,
                    "net_contribution": net_contribution,
                    "signal_known_at": signal_known_at,
                    "feature_asof": feature_asof,
                    "earliest_trade_at": earliest,
                    "entry_session": entry_session,
                    "exit_session": exit_session,
                    "borrow_status": borrow_status,
                    "cost_model": cost_model.name,
                    "mode": mode,
                    "scheme": scheme,
                    "accounting": "buy_and_hold",
                }
            )
            del net_contribution
        books.append(
            {
                "rebalance_id": rebalance_id,
                "mode": mode,
                "scheme": scheme,
                "cost_model": cost_model.name,
                "gross_return": position_gross_total,
                "net_return": position_net_total,
                "requested_gross": sizing.requested_gross,
                "achieved_gross": sizing.achieved_gross,
                "sizing_notes": ";".join(sizing.notes),
                "liquidity_constraint": sizing.liquidity_constraint,
                "contains_theoretical_shorts": any(
                    str(trade["borrow_status"]).startswith("theoretical")
                    for trade in trades
                    if trade["rebalance_id"] == rebalance_id
                ),
                "accounting": "buy_and_hold",
            }
        )
    notes = []
    if theoretical:
        notes.append("short positions are theoretical because borrow data was missing or not located")
    return SimulationResult(
        trades=pd.DataFrame(trades),
        rebalance_returns=pd.DataFrame(books),
        diagnostics=pd.DataFrame(diagnostics),
        contains_theoretical_shorts=theoretical,
        notes=notes,
    )


def _apply_borrow_filter(
    group: pd.DataFrame,
    mode: str,
    borrow: pd.DataFrame | None,
) -> tuple[pd.DataFrame, dict[str, str], list[dict[str, object]]]:
    statuses: dict[str, str] = {}
    dropped: list[dict[str, object]] = []
    keep_index: list[int] = []
    shorts_required = mode in {"short_deletions", "long_short"}
    for index, row in group.iterrows():
        security_id = str(row["security_id"])
        if shorts_required and row["action"] == "DELETE":
            status = borrow_status_for(security_id, row["signal_known_at"], borrow)
            statuses[security_id] = status
            if status == "unavailable":
                dropped.append(
                    {
                        "rebalance_id": row["rebalance_id"],
                        "security_id": security_id,
                        "reason": "borrow_unavailable",
                        "mode": mode,
                    }
                )
                continue
        else:
            statuses[security_id] = "not_applicable"
        keep_index.append(index)
    return group.loc[keep_index].copy(), statuses, dropped


def _hedge_context(
    mode: str,
    weights: pd.Series,
    by_id: pd.DataFrame,
    benchmark_security_id: str,
) -> dict[str, object] | None:
    if mode != "long_additions_hedged":
        return None
    long_ids = [security_id for security_id, weight in weights.items() if weight > 0 and security_id != benchmark_security_id]
    if not long_ids:
        return None
    entries = by_id.loc[long_ids, "entry_session"]
    exits = by_id.loc[long_ids, "effective_close"]
    if entries.nunique() != 1 or exits.nunique() != 1:
        raise DataContractError(
            "hedged mode requires one shared entry session and effective close inside the rebalance"
        )
    known = pd.to_datetime(by_id.loc[long_ids, "signal_known_at"], utc=True)
    feature = pd.to_datetime(by_id.loc[long_ids, "feature_asof"], utc=True)
    earliest = pd.to_datetime(by_id.loc[long_ids, "earliest_trade_at"], utc=True)
    return {
        "entry_session": entries.iloc[0],
        "effective_close": exits.iloc[0],
        "signal_known_at": known.max(),
        "feature_asof": feature.max(),
        "earliest_trade_at": earliest.max(),
    }
