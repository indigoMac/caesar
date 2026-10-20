"""Choose one defined-risk SPX put spread and price the combo."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal, ROUND_HALF_DOWN, ROUND_HALF_UP

from options_book.config import SpreadSettings
from options_book.errors import HaltError, SelectionError
from options_book.types import (
    ComboIntent,
    ComboLegSpec,
    OpenSpread,
    OptionQuote,
    is_finite,
)

SPX_MULTIPLIER = 100
STRIKE_TOLERANCE = 0.001


@dataclass(frozen=True)
class SelectedSpread:
    intent: ComboIntent
    short: OptionQuote
    long: OptionQuote


@dataclass(frozen=True)
class CloseDecision:
    should_close: bool
    reason: str
    debit_points: float


def choose_expiry(
    expiries: list[date],
    as_of: date,
    target_dte: int,
    min_dte: int,
    max_dte: int,
) -> date:
    window = [expiry for expiry in expiries if min_dte <= (expiry - as_of).days <= max_dte]
    if not window:
        raise SelectionError("no SPXW expiry in the dte window")
    return min(window, key=lambda expiry: (abs((expiry - as_of).days - target_dte), (expiry - as_of).days))


def select_put_spread(
    chain: list[OptionQuote] | tuple[OptionQuote, ...],
    spot: float,
    as_of: date,
    rules: SpreadSettings,
) -> SelectedSpread:
    if not is_finite(spot) or spot <= 0:
        raise SelectionError("spot is not a live price")
    puts = [quote for quote in chain if quote.right == "P"]
    if not puts:
        raise SelectionError("no puts in the chain snapshot")
    expiry = choose_expiry(
        sorted({quote.expiry for quote in puts}),
        as_of,
        rules.target_dte,
        rules.min_dte,
        rules.max_dte,
    )
    on_expiry = [quote for quote in puts if quote.expiry == expiry]
    floor = min(quote.strike for quote in on_expiry)
    candidates = [quote for quote in on_expiry if _short_candidate(quote, spot)]
    if not candidates:
        raise SelectionError("no otm put with a delta and a two-sided market")
    short = min(candidates, key=lambda quote: (abs(_delta(quote) - rules.short_delta), -quote.strike))
    if _same_strike(short.strike, floor):
        raise SelectionError("short strike is at the bottom of the scanned chain")
    long_strike = short.strike - rules.width_points
    longs = [quote for quote in on_expiry if _same_strike(quote.strike, long_strike)]
    if len(longs) != 1:
        raise SelectionError("long strike is not on the chain")
    long = longs[0]
    if not _two_sided(long) or long.con_id <= 0:
        raise SelectionError("long strike has no two-sided market")
    credit = round_credit_points(_mid(short) - _mid(long), rules.option_tick)
    if credit <= 0:
        raise SelectionError("combo mid is not a credit")
    if credit < Decimal(str(rules.min_credit_points)):
        raise SelectionError("credit is below the minimum")
    if credit >= Decimal(str(rules.width_points)):
        raise SelectionError("credit is not below the width")
    loss = (Decimal(str(rules.width_points)) - credit) * SPX_MULTIPLIER * rules.quantity
    if loss > Decimal(str(rules.max_loss_dollars)):
        raise SelectionError("max loss exceeds the cap")
    intent = ComboIntent(
        role="entry",
        order_action="BUY",
        quantity=rules.quantity,
        limit_price=float(-credit),
        legs=(
            ComboLegSpec(short.con_id, "SELL", 1),
            ComboLegSpec(long.con_id, "BUY", 1),
        ),
        expiry=expiry,
        short_con_id=short.con_id,
        long_con_id=long.con_id,
        short_strike=short.strike,
        long_strike=long.strike,
        premium_points=float(credit),
        width_points=rules.width_points,
        max_loss_dollars=float(loss),
        price_meaning="credit",
    )
    return SelectedSpread(intent=intent, short=short, long=long)


def evaluate_close(
    spread: OpenSpread,
    short: OptionQuote,
    long: OptionQuote,
    as_of: date,
    rules: SpreadSettings,
) -> CloseDecision:
    if not _two_sided(short) or not _two_sided(long):
        raise HaltError("close quote is not two-sided")
    debit = _mid(short) - _mid(long)
    dte = (spread.expiry - as_of).days
    entry = Decimal(str(spread.entry_credit_points))
    if dte <= rules.close_dte:
        return CloseDecision(True, "dte", float(debit))
    profit_line = entry * (Decimal("1") - Decimal(str(rules.profit_take_fraction)))
    if debit <= profit_line:
        return CloseDecision(True, "profit", float(debit))
    loss = debit - entry
    if loss >= Decimal(str(rules.loss_multiple_of_credit)) * entry:
        return CloseDecision(True, "loss", float(debit))
    return CloseDecision(False, "hold", float(debit))


def build_close_intent(spread: OpenSpread, short: OptionQuote, long: OptionQuote, tick: float) -> ComboIntent:
    debit = round_debit_points(_mid(short) - _mid(long), tick)
    return ComboIntent(
        role="close",
        order_action="BUY",
        quantity=spread.quantity,
        limit_price=float(debit),
        legs=(
            ComboLegSpec(spread.short_con_id, "BUY", 1),
            ComboLegSpec(spread.long_con_id, "SELL", 1),
        ),
        expiry=spread.expiry,
        short_con_id=spread.short_con_id,
        long_con_id=spread.long_con_id,
        short_strike=spread.short_strike,
        long_strike=spread.long_strike,
        premium_points=float(debit),
        width_points=spread.width_points,
        max_loss_dollars=0,
        price_meaning="debit",
    )


def credit_from_fill(limit_price: float, avg_fill: float | None, width_points: float) -> float:
    """A credit combo uses a negative IB limit. A positive fill on that order is the same credit."""
    if avg_fill is None:
        raise HaltError("combo fill price is missing")
    if limit_price >= 0:
        raise HaltError("combo fill price sign is ambiguous")
    credit = -Decimal(str(avg_fill)) if avg_fill < 0 else Decimal(str(avg_fill))
    if credit <= 0 or credit >= Decimal(str(width_points)):
        raise HaltError("combo fill price is outside the spread width")
    return float(credit)


def round_credit_points(credit: Decimal, tick: float) -> Decimal:
    tick_size = Decimal(str(tick))
    units = (credit / tick_size).quantize(Decimal("1"), rounding=ROUND_HALF_DOWN)
    return units * tick_size


def round_debit_points(debit: Decimal, tick: float) -> Decimal:
    tick_size = Decimal(str(tick))
    units = (debit / tick_size).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    return units * tick_size


def _mid(quote: OptionQuote) -> Decimal:
    if quote.bid is None or quote.ask is None:
        raise SelectionError("quote is missing a bid or ask")
    return (Decimal(str(quote.bid)) + Decimal(str(quote.ask))) / Decimal("2")


def _delta(quote: OptionQuote) -> float:
    if quote.delta is None:
        raise SelectionError("put is missing a delta")
    return quote.delta


def _same_strike(left: float, right: float) -> bool:
    return abs(left - right) <= STRIKE_TOLERANCE


def _short_candidate(quote: OptionQuote, spot: float) -> bool:
    if quote.con_id <= 0 or quote.delta is None or quote.delta >= 0 or quote.strike >= spot:
        return False
    return _two_sided(quote) and quote.bid is not None and quote.bid > 0


def _two_sided(quote: OptionQuote) -> bool:
    if quote.bid is None or quote.ask is None:
        return False
    if not is_finite(quote.bid) or not is_finite(quote.ask):
        return False
    return quote.bid >= 0 and quote.ask > 0 and quote.ask >= quote.bid
