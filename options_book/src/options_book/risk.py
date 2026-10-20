"""Gates that have to pass before any order, and the session clock."""

from __future__ import annotations

from datetime import date, datetime, time
from pathlib import Path
from typing import Mapping, Protocol
from zoneinfo import ZoneInfo

from options_book.config import Settings
from options_book.errors import HaltError, OrdersDisabled
from options_book.types import BookState, BrokerPosition, is_finite

NY = ZoneInfo("America/New_York")
PAPER_HOSTS = frozenset({"127.0.0.1", "localhost"})
PAPER_PORT = 4002
LIVE_ENV = "OPTIONS_BOOK_LIVE"
OPTION_SYMBOLS = frozenset({"SPX", "SPXW"})
FUTURE_SYMBOLS = frozenset({"MES", "ES"})


class TradingCalendar(Protocol):
    def is_session(self, day: date) -> bool: ...


class XnysCalendar:
    def __init__(self) -> None:
        import exchange_calendars as calendars

        self._calendar = calendars.get_calendar("XNYS")

    def is_session(self, day: date) -> bool:
        return bool(self._calendar.is_session(day.isoformat()))


def ensure_endpoint(settings: Settings, env: Mapping[str, str]) -> None:
    paper = settings.broker.host in PAPER_HOSTS and settings.broker.port == PAPER_PORT
    live = settings.broker.live_enabled and env.get(LIVE_ENV) == "1"
    if paper or live:
        return
    raise HaltError("refusing endpoint that is not the paper gateway")


def assert_can_submit(
    settings: Settings,
    env: Mapping[str, str],
    halted: bool,
    halt_file: Path,
) -> None:
    ensure_endpoint(settings, env)
    if not settings.broker.submit_orders:
        raise OrdersDisabled()
    if halted:
        raise HaltError("halt latch is set")
    if halt_file.exists():
        raise HaltError("halt file present")


def in_session(now: datetime, settings: Settings, calendar: TradingCalendar) -> bool:
    if now.tzinfo is None:
        raise HaltError("clock is naive")
    local = now.astimezone(NY)
    if not calendar.is_session(local.date()):
        return False
    start = time.fromisoformat(settings.session.entry_start)
    end = time.fromisoformat(settings.session.entry_end)
    return start <= local.time() <= end


def session_date(now: datetime) -> date:
    if now.tzinfo is None:
        raise HaltError("clock is naive")
    return now.astimezone(NY).date()


def assert_fresh(quoted_at: datetime | None, now: datetime, max_age_seconds: float, label: str) -> None:
    if quoted_at is None:
        raise HaltError(f"{label} quote has no timestamp")
    if quoted_at.tzinfo is None or now.tzinfo is None:
        raise HaltError(f"{label} quote timestamp is naive")
    age = (now - quoted_at).total_seconds()
    if age < -1:
        raise HaltError(f"{label} quote timestamp is in the future")
    if age > max_age_seconds:
        raise HaltError(f"{label} quote is stale")


def assert_daily_loss(pnl: float | None, limit: float, book_open: bool) -> None:
    if pnl is None or not is_finite(pnl):
        if book_open:
            raise HaltError("daily pnl unavailable")
        return
    if pnl <= -abs(limit):
        raise HaltError("daily loss breach")


def reconcile(book: BookState, positions: list[BrokerPosition] | tuple[BrokerPosition, ...]) -> None:
    options, futures = _tracked(positions)
    if book.spread is None:
        if options:
            raise HaltError("unexplained SPX or futures position")
    else:
        expected = {
            book.spread.short_con_id: -book.spread.quantity,
            book.spread.long_con_id: book.spread.quantity,
        }
        if options != expected:
            raise HaltError("spread legs do not match the audit log")
    if book.hedge_qty == 0:
        if futures:
            raise HaltError("unexplained SPX or futures position")
        return
    if book.hedge_con_id is None:
        raise HaltError("hedge quantity does not match the audit log")
    expected_future = {book.hedge_con_id: (book.hedge_symbol, book.hedge_qty)}
    if futures != expected_future:
        raise HaltError("hedge quantity does not match the audit log")


def option_quantities(positions: list[BrokerPosition] | tuple[BrokerPosition, ...]) -> dict[int, int]:
    options, _futures = _tracked(positions)
    return options


def future_quantities(
    positions: list[BrokerPosition] | tuple[BrokerPosition, ...],
) -> dict[int, tuple[str, int]]:
    _options, futures = _tracked(positions)
    return futures


def _tracked(
    positions: list[BrokerPosition] | tuple[BrokerPosition, ...],
) -> tuple[dict[int, int], dict[int, tuple[str, int]]]:
    options: dict[int, int] = {}
    futures: dict[int, tuple[str, int]] = {}
    for position in positions:
        if position.quantity == 0 or not _is_tracked(position):
            continue
        if position.con_id <= 0:
            raise HaltError("tracked position is missing a contract id")
        quantity = _whole(position.quantity)
        if position.sec_type == "OPT":
            options[position.con_id] = options.get(position.con_id, 0) + quantity
        else:
            current = futures.get(position.con_id)
            if current is not None and current[0] != position.symbol:
                raise HaltError("hedge quantity does not match the audit log")
            total = quantity if current is None else current[1] + quantity
            futures[position.con_id] = (position.symbol, total)
    options = {con_id: quantity for con_id, quantity in options.items() if quantity != 0}
    futures = {con_id: row for con_id, row in futures.items() if row[1] != 0}
    return options, futures


def _is_tracked(position: BrokerPosition) -> bool:
    if position.sec_type == "OPT" and position.symbol in OPTION_SYMBOLS:
        return True
    return position.sec_type == "FUT" and position.symbol in FUTURE_SYMBOLS


def _whole(quantity: float) -> int:
    rounded = round(quantity)
    if abs(quantity - rounded) > 1e-6:
        raise HaltError("fractional position")
    return int(rounded)
