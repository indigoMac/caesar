"""One cycle: reconcile, maybe sell the spread, maybe close it, maybe hedge delta."""

from __future__ import annotations

import argparse
import logging
import os
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict
from datetime import date, datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from options_book.broker import IbBroker
from options_book.config import LoadedConfig, load_settings
from options_book.delta import (
    decide_hedge,
    dollar_delta,
    GreekLeg,
    hedge_limit_price,
    net_vega,
    point_value,
    select_future,
)
from options_book.errors import HaltError, SelectionError
from options_book.risk import (
    TradingCalendar,
    XnysCalendar,
    assert_daily_loss,
    assert_fresh,
    future_quantities,
    in_session,
    option_quantities,
    reconcile,
    session_date,
)
from options_book.spreads import (
    SPX_MULTIPLIER,
    build_close_intent,
    credit_from_fill,
    evaluate_close,
    select_put_spread,
)
from options_book.store import Store, StoredOrder
from options_book.types import (
    BookState,
    BrokerPosition,
    ComboIntent,
    CycleResult,
    HedgeIntent,
    OpenSpread,
    OrderStatus,
)

logger = logging.getLogger(__name__)
NY = ZoneInfo("America/New_York")


class CycleContext:
    def __init__(
        self,
        *,
        now: datetime,
        loaded: LoadedConfig,
        broker,
        store: Store,
        calendar: TradingCalendar,
        env: Mapping[str, str],
    ) -> None:
        self.now = now
        self.loaded = loaded
        self.broker = broker
        self.store = store
        self.calendar = calendar
        self.env = env

    @property
    def settings(self):
        return self.loaded.settings

    def halt_file(self) -> Path:
        return self.loaded.resolve(self.settings.risk.halt_file)


def cycle(
    *,
    now: datetime,
    loaded: LoadedConfig,
    broker,
    store: Store,
    calendar: TradingCalendar,
    env: Mapping[str, str],
) -> CycleResult:
    ctx = CycleContext(now=now, loaded=loaded, broker=broker, store=store, calendar=calendar, env=env)
    try:
        _run(ctx)
    except HaltError as exc:
        ctx.store.halt(exc.reason, ctx.now)
        _safe_cancel(ctx)
        logger.error("%s", exc.reason)
        return CycleResult(True, exc.reason)
    except Exception as exc:
        reason = f"exception: {type(exc).__name__}: {exc}"[:500]
        ctx.store.halt(reason, ctx.now)
        _safe_cancel(ctx)
        logger.exception("cycle failed")
        return CycleResult(True, reason)
    return CycleResult(False, "")


def run_forever(
    *,
    clock: Callable[[], datetime],
    sleep: Callable[[float], None],
    max_cycles: int | None = None,
    **kwargs,
) -> CycleResult:
    last = CycleResult(False, "")
    count = 0
    while max_cycles is None or count < max_cycles:
        last = cycle(now=clock(), **kwargs)
        count += 1
        if last.halted:
            return last
        if max_cycles is not None and count >= max_cycles:
            return last
        sleep(kwargs["loaded"].settings.loop.poll_seconds)
    return last


def _run(ctx: CycleContext) -> None:
    blocked = _existing_halt(ctx)
    if blocked is not None:
        raise HaltError(blocked.reason)
    _unknown_orders(ctx)
    positions = ctx.broker.positions()
    reconcile(ctx.store.book(), positions)
    book = ctx.store.book()
    assert_daily_loss(
        ctx.broker.daily_pnl(),
        ctx.settings.risk.max_daily_loss_dollars,
        book.spread is not None or book.hedge_qty != 0,
    )
    if _refresh_working(ctx):
        return
    if not in_session(ctx.now, ctx.settings, ctx.calendar):
        ctx.store.record_event("outside_session", {}, ctx.now)
        return
    book = ctx.store.book()
    if book.spread is None and book.hedge_qty == 0:
        _maybe_enter(ctx)
        book = ctx.store.book()
    if book.spread is not None:
        _manage(ctx)
    elif book.hedge_qty != 0:
        _flatten_hedge(ctx)


def _existing_halt(ctx: CycleContext) -> CycleResult | None:
    if ctx.halt_file().exists() and not ctx.store.is_halted():
        ctx.store.halt("halt file present", ctx.now)
    if ctx.store.is_halted():
        return CycleResult(True, ctx.store.halt_reason() or "halted")
    return None


def _unknown_orders(ctx: CycleContext) -> None:
    live = {order.broker_order_id for order in ctx.broker.open_tracked_orders()}
    unknown = sorted(live - ctx.store.broker_order_ids())
    if unknown:
        raise HaltError("untracked open order: " + ",".join(str(order_id) for order_id in unknown))


def _maybe_enter(ctx: CycleContext) -> None:
    as_of = session_date(ctx.now)
    try:
        snapshot = ctx.broker.entry_snapshot(as_of)
        selected = select_put_spread(snapshot.options, snapshot.spot, as_of, ctx.settings.spread)
    except SelectionError as exc:
        ctx.store.record_event("no_trade", {"reason": str(exc)}, ctx.now)
        logger.info("no trade: %s", exc)
        return
    assert_fresh(selected.short.quoted_at, ctx.now, ctx.settings.hedge.greek_max_age_seconds, "short leg")
    assert_fresh(selected.long.quoted_at, ctx.now, ctx.settings.hedge.greek_max_age_seconds, "long leg")
    _submit(ctx, selected.intent)


def _manage(ctx: CycleContext) -> None:
    book = ctx.store.book()
    spread = book.spread
    if spread is None:
        return
    short, long = _leg_quotes(ctx, spread)
    legs = (
        GreekLeg(-spread.quantity, _required(short.delta, "short delta"), SPX_MULTIPLIER, short.vega),
        GreekLeg(spread.quantity, _required(long.delta, "long delta"), SPX_MULTIPLIER, long.vega),
    )
    option_dd = dollar_delta(legs)
    vega = net_vega(legs)
    net_dd = option_dd + _future_dollar_delta(book)
    ctx.store.record_event(
        "risk",
        {
            "dollar_delta": str(net_dd),
            "option_dollar_delta": str(option_dd),
            "vega": str(vega),
            "hedge_qty": book.hedge_qty,
        },
        ctx.now,
    )
    decision = evaluate_close(spread, short, long, session_date(ctx.now), ctx.settings.spread)
    if decision.should_close:
        logger.info("closing spread: %s", decision.reason)
        _submit(ctx, build_close_intent(spread, short, long, ctx.settings.spread.option_tick))
        return
    _hedge(ctx, option_dd, spread.expiry)


def _flatten_hedge(ctx: CycleContext) -> None:
    _hedge(ctx, point_value(ctx.store.book().hedge_symbol) * 0, session_date(ctx.now))


def _hedge(ctx: CycleContext, option_dollar_delta, after: date) -> None:
    from decimal import Decimal

    book = ctx.store.book()
    rules = ctx.settings.hedge
    if not isinstance(option_dollar_delta, Decimal):
        option_dollar_delta = Decimal(option_dollar_delta)
    decision = decide_hedge(
        option_dollar_delta,
        book.hedge_qty,
        point_value(book.hedge_symbol),
        rules.band_contracts,
        rules.min_adjustment,
        rules.max_mes_position,
    )
    if decision.halt:
        raise HaltError(decision.reason)
    if decision.trade_qty == 0:
        ctx.store.record_event(
            "hedge_hold",
            {"reason": decision.reason, "net_contracts": str(decision.net_contracts)},
            ctx.now,
        )
        return
    chosen = _hedge_contract(ctx, book, after)
    quote = ctx.broker.future_quote(chosen.con_id)
    assert_fresh(quote.quoted_at, ctx.now, rules.greek_max_age_seconds, "future")
    limit = hedge_limit_price(
        decision.trade_qty,
        quote.bid,
        quote.ask,
        rules.tick_size,
        rules.max_spread_points,
    )
    action = "BUY" if decision.trade_qty > 0 else "SELL"
    intent = HedgeIntent(
        role="hedge",
        symbol=chosen.symbol,
        con_id=chosen.con_id,
        exchange=chosen.exchange,
        action=action,
        quantity=abs(decision.trade_qty),
        limit_price=limit,
        trade_qty=decision.trade_qty,
    )
    _submit(ctx, intent)


def _hedge_contract(ctx: CycleContext, book: BookState, after: date):
    candidates = ctx.broker.list_futures(book.hedge_symbol)
    if book.hedge_con_id is None:
        return select_future(candidates, after=after, symbol=book.hedge_symbol)
    matches = [candidate for candidate in candidates if candidate.con_id == book.hedge_con_id]
    if len(matches) != 1:
        raise HaltError("hedge contract is no longer available")
    return matches[0]


def _leg_quotes(ctx: CycleContext, spread: OpenSpread):
    quotes = {quote.con_id: quote for quote in ctx.broker.quote_legs([spread.short_con_id, spread.long_con_id])}
    short = quotes.get(spread.short_con_id)
    long = quotes.get(spread.long_con_id)
    if short is None or long is None:
        raise HaltError("held leg quote is missing")
    age = ctx.settings.hedge.greek_max_age_seconds
    assert_fresh(short.quoted_at, ctx.now, age, "short leg")
    assert_fresh(long.quoted_at, ctx.now, age, "long leg")
    return short, long


def _submit(ctx: CycleContext, intent: ComboIntent | HedgeIntent) -> None:
    payload = _payload(intent)
    if not ctx.settings.broker.submit_orders:
        ctx.store.insert_order(
            ts=ctx.now,
            role=intent.role,
            status="suppressed",
            broker_order_id=None,
            payload=payload,
        )
        logger.info("suppressed %s order", intent.role)
        return
    if isinstance(intent, ComboIntent):
        margin = ctx.broker.what_if(
            intent,
            halted=ctx.store.is_halted(),
            halt_file=ctx.halt_file(),
            env=ctx.env,
        )
        if intent.role == "entry" and not _entry_margin_ok(ctx, margin):
            ctx.store.insert_order(
                ts=ctx.now,
                role=intent.role,
                status="suppressed",
                broker_order_id=None,
                payload=payload,
            )
            ctx.store.record_event(
                "margin_skip",
                {"error": margin.error, "init_margin": margin.init_margin},
                ctx.now,
            )
            return
        if intent.role == "close" and (margin.error or margin.init_margin is None):
            raise HaltError("close what-if margin unavailable")
        ctx.store.record_event(
            "margin",
            {"role": intent.role, "init_margin": margin.init_margin},
            ctx.now,
        )
    broker_order_id = ctx.broker.place_order(
        intent,
        halted=ctx.store.is_halted(),
        halt_file=ctx.halt_file(),
        env=ctx.env,
    )
    order_id = ctx.store.insert_order(
        ts=ctx.now,
        role=intent.role,
        status="submitted",
        broker_order_id=broker_order_id,
        payload=payload,
    )
    _advance_order(ctx, ctx.store.get_order(order_id))


def _entry_margin_ok(ctx: CycleContext, margin) -> bool:
    if margin.error or margin.init_margin is None:
        return False
    funds = ctx.broker.available_funds()
    return funds is not None and margin.init_margin <= funds


def _refresh_working(ctx: CycleContext) -> bool:
    working = ctx.store.working_orders()
    if len(working) > 1:
        raise HaltError("more than one working order")
    if not working:
        return False
    return _advance_order(ctx, working[0])


def _advance_order(ctx: CycleContext, order: StoredOrder) -> bool:
    if order.broker_order_id is None:
        raise HaltError("working order has no broker id")
    status = ctx.broker.order_status(order.broker_order_id)
    age = (ctx.now - order.ts).total_seconds()
    timeout = ctx.settings.spread.order_timeout_seconds
    if status.status == "rejected":
        ctx.store.update_status(order.id, "rejected", ctx.now)
        raise HaltError(f"{order.role} order was rejected")
    if status.status == "cancelled" and status.filled == 0:
        if order.status != "cancelled":
            ctx.store.update_status(order.id, "cancelled", ctx.now)
        return True
    if status.status == "submitted" and status.filled == 0 and order.status == "submitted" and age > timeout:
        ctx.broker.cancel_order(order.broker_order_id)
        ctx.store.update_status(order.id, "cancel_requested", ctx.now)
        return True
    if status.status == "submitted" and status.filled == 0:
        return True
    if status.filled != int(order.payload["quantity"]):
        raise HaltError("unexpected partial fill")
    before = ctx.store.book()
    verdict = _verdict(ctx, order, status)
    if verdict == "pending":
        if age > timeout:
            raise HaltError("fill is not visible in positions")
        return True
    if verdict == "mismatch":
        raise HaltError("fill does not match the order")
    _commit(ctx, order, status)
    if order.status != "filled":
        ctx.store.update_status(order.id, "filled", ctx.now)
    after = ctx.store.book()
    return not (before.spread is None and after.spread is not None)


def _verdict(ctx: CycleContext, order: StoredOrder, status: OrderStatus) -> str:
    positions = ctx.broker.positions()
    options = option_quantities(positions)
    futures = future_quantities(positions)
    book = ctx.store.book()
    payload = order.payload
    if order.role == "hedge":
        return _hedge_verdict(book, futures, options, payload)
    if not _futures_match(book, futures):
        return "mismatch"
    short_id = int(payload["short_con_id"])
    long_id = int(payload["long_con_id"])
    quantity = int(payload["quantity"])
    if order.role == "entry":
        expected = {short_id: -quantity, long_id: quantity}
        if options == expected:
            return "match"
        if options == {}:
            return "pending"
        return "mismatch"
    if order.role == "close":
        previous = {short_id: -quantity, long_id: quantity}
        if options == {}:
            return "match"
        if options == previous:
            return "pending"
        return "mismatch"
    raise HaltError(f"unknown order role: {order.role}")


def _hedge_verdict(book: BookState, futures: dict, options: dict, payload: dict) -> str:
    if book.spread is None:
        expected_options: dict[int, int] = {}
    else:
        expected_options = {
            book.spread.short_con_id: -book.spread.quantity,
            book.spread.long_con_id: book.spread.quantity,
        }
    if options != expected_options:
        return "mismatch"
    new_qty = book.hedge_qty + int(payload["trade_qty"])
    con_id = int(payload["con_id"])
    symbol = str(payload["symbol"])
    target = {} if new_qty == 0 else {con_id: (symbol, new_qty)}
    current = {} if book.hedge_qty == 0 or book.hedge_con_id is None else {book.hedge_con_id: (book.hedge_symbol, book.hedge_qty)}
    if futures == target:
        return "match"
    if futures == current:
        return "pending"
    return "mismatch"


def _futures_match(book: BookState, futures: dict) -> bool:
    if book.hedge_qty == 0:
        return futures == {}
    if book.hedge_con_id is None:
        return False
    return futures == {book.hedge_con_id: (book.hedge_symbol, book.hedge_qty)}


def _commit(ctx: CycleContext, order: StoredOrder, status: OrderStatus) -> None:
    book = ctx.store.book()
    payload = order.payload
    if order.role == "entry":
        credit = credit_from_fill(
            float(payload["limit_price"]),
            status.avg_price,
            float(payload["width_points"]),
        )
        spread = OpenSpread(
            short_con_id=int(payload["short_con_id"]),
            long_con_id=int(payload["long_con_id"]),
            short_strike=float(payload["short_strike"]),
            long_strike=float(payload["long_strike"]),
            expiry=date.fromisoformat(payload["expiry"]),
            quantity=status.filled,
            entry_credit_points=credit,
            width_points=float(payload["width_points"]),
        )
        ctx.store.save_book(
            BookState(spread, book.hedge_qty, book.hedge_con_id, book.hedge_symbol),
            ctx.now,
        )
        return
    if order.role == "close":
        ctx.store.save_book(BookState(None, book.hedge_qty, book.hedge_con_id, book.hedge_symbol), ctx.now)
        return
    if order.role == "hedge":
        ctx.store.save_book(
            BookState(
                book.spread,
                book.hedge_qty + int(payload["trade_qty"]),
                int(payload["con_id"]),
                str(payload["symbol"]),
            ),
            ctx.now,
        )
        return
    raise HaltError(f"unknown order role: {order.role}")


def _safe_cancel(ctx: CycleContext) -> None:
    try:
        for order in ctx.store.working_orders():
            if order.broker_order_id is None:
                continue
            status = ctx.broker.order_status(order.broker_order_id)
            if status.filled == 0 and order.status == "submitted":
                ctx.broker.cancel_order(order.broker_order_id)
                ctx.store.update_status(order.id, "cancel_requested", ctx.now)
            elif status.filled > 0:
                ctx.store.record_event(
                    "left_partial_order",
                    {"id": order.id, "filled": status.filled},
                    ctx.now,
                )
    except Exception:
        logger.exception("could not cancel working orders after halt")


def _payload(intent: ComboIntent | HedgeIntent) -> dict:
    payload = asdict(intent)
    expiry = payload.get("expiry")
    if isinstance(expiry, date):
        payload["expiry"] = expiry.isoformat()
    return payload


def _required(value: float | None, label: str) -> float:
    if value is None:
        raise HaltError(f"held leg is missing {label}")
    return value


def _future_dollar_delta(book: BookState):
    from decimal import Decimal

    return Decimal(book.hedge_qty) * point_value(book.hedge_symbol)


def run_command(
    command: str,
    loaded: LoadedConfig,
    store: Store,
    *,
    broker=None,
    calendar: TradingCalendar | None = None,
    env: Mapping[str, str] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], datetime] | None = None,
    max_cycles: int | None = None,
) -> int:
    env = os.environ if env is None else env
    clock = clock or (lambda: datetime.now(NY))
    if command == "clear-halt":
        halt_file = loaded.resolve(loaded.settings.risk.halt_file)
        if halt_file.exists():
            logger.error("remove the halt file before clearing the latch")
            return 1
        store.clear_halt(datetime.now(timezone.utc))
        logger.info("halt latch cleared")
        return 0
    owns_broker = broker is None
    broker = broker if broker is not None else IbBroker(loaded.settings)
    calendar = calendar if calendar is not None else XnysCalendar()
    try:
        if owns_broker:
            try:
                broker.connect()
            except Exception as exc:
                reason = exc.reason if isinstance(exc, HaltError) else f"exception: {type(exc).__name__}: {exc}"
                store.halt(reason[:500], datetime.now(timezone.utc))
                logger.error("%s", reason)
                return 2
        kwargs = {
            "loaded": loaded,
            "broker": broker,
            "store": store,
            "calendar": calendar,
            "env": env,
        }
        if command == "once":
            result = cycle(now=clock(), **kwargs)
        elif command == "run":
            result = run_forever(clock=clock, sleep=sleep, max_cycles=max_cycles, **kwargs)
        else:
            raise HaltError(f"unknown command: {command}")
    except KeyboardInterrupt:
        logger.info("stopped")
        return 0
    finally:
        if owns_broker:
            broker.disconnect()
    if result.halted:
        logger.error("%s", result.reason)
        return 2
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="options-book")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "once", "clear-halt"):
        command = sub.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    loaded = load_settings(args.config)
    try:
        store = Store(loaded.resolve(loaded.settings.store.path), loaded.settings.hedge.symbol)
    except HaltError as exc:
        logger.error("%s", exc.reason)
        return 2
    try:
        return run_command(args.command, loaded, store)
    finally:
        store.close()


def cli() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    cli()
