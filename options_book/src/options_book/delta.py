"""Dollar delta of the option book, and the futures quantity that flattens it."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal, ROUND_HALF_UP

from options_book.errors import HaltError
from options_book.types import FutureCandidate

POINT_VALUE = {"MES": Decimal("5"), "ES": Decimal("50")}


@dataclass(frozen=True)
class GreekLeg:
    quantity: float
    delta: float
    multiplier: float
    vega: float | None = None


@dataclass(frozen=True)
class HedgeDecision:
    trade_qty: int
    target_qty: int
    net_contracts: Decimal
    option_contracts: Decimal
    halt: bool
    reason: str


def point_value(symbol: str) -> Decimal:
    try:
        return POINT_VALUE[symbol]
    except KeyError:
        raise HaltError(f"unsupported hedge symbol: {symbol}") from None


def dollar_delta(legs: list[GreekLeg] | tuple[GreekLeg, ...]) -> Decimal:
    total = Decimal("0")
    for leg in legs:
        total += Decimal(str(leg.quantity)) * Decimal(str(leg.delta)) * Decimal(str(leg.multiplier))
    return total


def net_vega(legs: list[GreekLeg] | tuple[GreekLeg, ...]) -> Decimal:
    total = Decimal("0")
    for leg in legs:
        if leg.vega is None:
            raise HaltError("held leg is missing vega")
        total += Decimal(str(leg.quantity)) * Decimal(str(leg.vega)) * Decimal(str(leg.multiplier))
    return total


def round_half_away_from_zero(value: Decimal) -> int:
    return int(value.quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def decide_hedge(
    option_dollar_delta: Decimal,
    current_future_qty: int,
    future_point_value: Decimal,
    band_contracts: float,
    min_adjustment: int,
    max_position: int,
) -> HedgeDecision:
    option_contracts = option_dollar_delta / future_point_value
    net = option_contracts + Decimal(current_future_qty)
    band = Decimal(str(band_contracts))
    if abs(net) <= band:
        return HedgeDecision(0, current_future_qty, net, option_contracts, False, "inside band")
    raw_target = -round_half_away_from_zero(option_contracts)
    clamped = max(-max_position, min(max_position, raw_target))
    residual = option_contracts + Decimal(clamped)
    if abs(raw_target) > max_position and abs(residual) > band:
        return HedgeDecision(0, clamped, net, option_contracts, True, "hedge exceeds max position")
    trade = clamped - current_future_qty
    if abs(trade) < min_adjustment:
        return HedgeDecision(0, current_future_qty, net, option_contracts, False, "adjustment below minimum")
    return HedgeDecision(trade, clamped, net, option_contracts, False, "rebalance")


def hedge_limit_price(
    trade_qty: int,
    bid: float,
    ask: float,
    tick: float,
    max_spread: float,
) -> float:
    if trade_qty == 0:
        raise HaltError("hedge limit requested for a zero trade")
    bid_size = Decimal(str(bid))
    ask_size = Decimal(str(ask))
    if bid_size <= 0 or ask_size <= 0 or ask_size < bid_size:
        raise HaltError("future quote is not two-sided")
    if ask_size - bid_size > Decimal(str(max_spread)):
        raise HaltError("future quote is wider than the cap")
    touch = ask_size if trade_qty > 0 else bid_size
    snapped = _snap_to_tick(touch, Decimal(str(tick)))
    if trade_qty > 0 and snapped - ask_size > Decimal("0.000001"):
        raise HaltError("future ask is off the tick grid")
    if trade_qty < 0 and bid_size - snapped > Decimal("0.000001"):
        raise HaltError("future bid is off the tick grid")
    return float(snapped)


def select_future(candidates: list[FutureCandidate], *, after: date, symbol: str) -> FutureCandidate:
    value = point_value(symbol)
    eligible = [
        candidate
        for candidate in candidates
        if candidate.symbol == symbol
        and candidate.expiry > after
        and Decimal(str(candidate.multiplier)) == value
    ]
    if not eligible:
        raise HaltError(f"no {symbol} future expires after {after.isoformat()}")
    return min(eligible, key=lambda candidate: (candidate.expiry, candidate.con_id))


def _snap_to_tick(price: Decimal, tick: Decimal) -> Decimal:
    return (price / tick).quantize(Decimal("1"), rounding=ROUND_HALF_UP) * tick
