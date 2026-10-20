"""Shared values passed between the broker, the book, and the cycle."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime


def is_finite(value: float | None) -> bool:
    return value is not None and math.isfinite(value)


@dataclass(frozen=True)
class OptionQuote:
    con_id: int
    expiry: date
    strike: float
    right: str
    delta: float | None
    bid: float | None
    ask: float | None
    vega: float | None
    quoted_at: datetime | None


@dataclass(frozen=True)
class FutureCandidate:
    con_id: int
    symbol: str
    expiry: date
    local_symbol: str
    exchange: str
    multiplier: float


@dataclass(frozen=True)
class FutureQuote:
    con_id: int
    bid: float
    ask: float
    quoted_at: datetime | None


@dataclass(frozen=True)
class BrokerPosition:
    con_id: int
    symbol: str
    sec_type: str
    quantity: float
    trading_class: str = ""


@dataclass(frozen=True)
class ComboLegSpec:
    con_id: int
    action: str
    ratio: int


@dataclass(frozen=True)
class ComboIntent:
    """A BAG order. A credit entry is a negative limit on a buy combo."""

    role: str
    order_action: str
    quantity: int
    limit_price: float
    legs: tuple[ComboLegSpec, ...]
    expiry: date
    short_con_id: int
    long_con_id: int
    short_strike: float
    long_strike: float
    premium_points: float
    width_points: float
    max_loss_dollars: float
    price_meaning: str


@dataclass(frozen=True)
class HedgeIntent:
    role: str
    symbol: str
    con_id: int
    exchange: str
    action: str
    quantity: int
    limit_price: float
    trade_qty: int


@dataclass(frozen=True)
class OrderStatus:
    broker_order_id: int
    status: str
    filled: int
    remaining: int
    avg_price: float | None


@dataclass(frozen=True)
class TrackedOpenOrder:
    broker_order_id: int
    symbol: str
    sec_type: str


@dataclass(frozen=True)
class MarginCheck:
    init_margin: float | None
    error: str | None


@dataclass(frozen=True)
class EntrySnapshot:
    spot: float
    options: tuple[OptionQuote, ...]


@dataclass(frozen=True)
class OpenSpread:
    short_con_id: int
    long_con_id: int
    short_strike: float
    long_strike: float
    expiry: date
    quantity: int
    entry_credit_points: float
    width_points: float


@dataclass(frozen=True)
class BookState:
    spread: OpenSpread | None
    hedge_qty: int
    hedge_con_id: int | None
    hedge_symbol: str


@dataclass(frozen=True)
class CycleResult:
    halted: bool
    reason: str
