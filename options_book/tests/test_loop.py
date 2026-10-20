"""Cycle tests. None of these open a socket."""

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from zoneinfo import ZoneInfo

import yaml

from options_book.config import Settings
from options_book.loop import cycle, main, run_forever
from options_book.types import BookState, BrokerPosition, FutureCandidate, FutureQuote, OpenSpread, TrackedOpenOrder
from tests.helpers import EXPIRY, NOW, Closed, Weekdays, loaded, settings_with, snapshot, store_for
from tests.fakes import FakeBroker

NY = ZoneInfo("America/New_York")


def _cycle(tmp_path, broker, settings=None, now=NOW, calendar=None, book=None):
    config = loaded(tmp_path, settings)
    store = store_for(config)
    if book is not None:
        store.save_book(book, now)
    result = cycle(
        now=now,
        loaded=config,
        broker=broker,
        store=store,
        calendar=calendar or Weekdays(),
        env={},
    )
    return result, store


def _broker(settings, chain=None):
    broker = FakeBroker(settings)
    if chain is not None:
        broker.snapshot = chain
    return broker


def _spread_book() -> BookState:
    return BookState(OpenSpread(3, 2, 5650, 5625, EXPIRY, 1, 4, 25), 0, None, "MES")


def test_dry_run_logs_the_combo_and_does_not_place(tmp_path):
    settings = Settings_submit(False)
    broker = _broker(settings, snapshot())
    result, store = _cycle(tmp_path, broker, settings)
    assert result.halted is False
    assert broker.placed == []
    assert store.orders()[0].status == "suppressed"
    assert store.orders()[0].role == "entry"
    assert store.orders()[0].payload["limit_price"] == -4
    store.close()


def test_submit_enters_once_and_does_not_hedge_inside_the_band(tmp_path):
    settings = Settings_submit(True)
    broker = _broker(settings, snapshot())
    result, store = _cycle(tmp_path, broker, settings)
    assert result.halted is False
    assert len(broker.placed) == 1
    assert broker.placed[0].role == "entry"
    assert store.book().spread is not None
    assert store.book().spread.entry_credit_points == 4
    assert store.book().hedge_qty == 0
    kinds = [kind for kind, _payload in store.events()]
    assert "risk" in kinds
    risk = next(payload for kind, payload in store.events() if kind == "risk")
    assert Decimal(risk["vega"]) == Decimal("-15")
    result2, _store2 = _cycle(tmp_path, broker, settings)
    assert result2.halted is False
    assert len(broker.placed) == 1
    store.close()


def test_no_trade_does_not_latch(tmp_path):
    settings = Settings_submit(True, spread={"max_loss_dollars": 100})
    broker = _broker(settings, snapshot())
    result, store = _cycle(tmp_path, broker, settings)
    assert result.halted is False
    assert broker.placed == []
    assert store.events()[-1][0] == "no_trade"
    store.close()


def test_stale_entry_quotes_latch(tmp_path):
    settings = Settings_submit(True)
    old = NOW - timedelta(seconds=30)
    broker = _broker(settings, snapshot(old))
    result, store = _cycle(tmp_path, broker, settings)
    assert result.halted is True
    assert "stale" in result.reason
    assert broker.placed == []
    store.close()
    assert store_for(loaded(tmp_path, settings)).is_halted() is True


def test_margin_skip_does_not_place(tmp_path):
    settings = Settings_submit(True)
    broker = _broker(settings, snapshot())
    broker.margin = broker.margin.__class__(1_000_000, None)
    broker.funds = 1_000
    result, store = _cycle(tmp_path, broker, settings)
    assert result.halted is False
    assert broker.placed == []
    assert any(kind == "margin_skip" for kind, _payload in store.events())
    store.close()


def test_rejected_order_latches(tmp_path):
    settings = Settings_submit(True)
    broker = _broker(settings, snapshot())
    broker.fill_mode = "rejected"
    result, store = _cycle(tmp_path, broker, settings)
    assert result.halted is True
    assert "rejected" in result.reason
    assert store.orders()[0].status == "rejected"
    store.close()


def test_working_order_is_not_duplicated_and_times_out(tmp_path):
    settings = Settings_submit(True)
    broker = _broker(settings, snapshot())
    broker.fill_mode = "submitted"
    first, store = _cycle(tmp_path, broker, settings)
    assert first.halted is False
    assert len(broker.placed) == 1
    second, _store = _cycle(tmp_path, broker, settings, now=NOW + timedelta(seconds=30))
    assert second.halted is False
    assert len(broker.placed) == 1
    third, store = _cycle(tmp_path, broker, settings, now=NOW + timedelta(seconds=181))
    assert third.halted is False
    assert broker.cancelled == [1]
    assert store.working_orders()[0].status == "cancel_requested"
    store.close()


def test_filled_order_without_positions_halts_after_the_timeout(tmp_path):
    settings = Settings_submit(True)
    broker = _broker(settings, snapshot())
    broker.fill_mode = "filled_unconfirmed"
    first, _store = _cycle(tmp_path, broker, settings)
    assert first.halted is False
    result, store = _cycle(tmp_path, broker, settings, now=NOW + timedelta(seconds=181))
    assert result.halted is True
    assert "not visible" in result.reason
    assert store.book().spread is None
    store.close()


def test_unexplained_position_and_one_leg_latch(tmp_path):
    settings = Settings_submit(True)
    broker = _broker(settings, snapshot())
    broker.positions_list = [BrokerPosition(3, "SPX", "OPT", -1, "SPXW")]
    result, store = _cycle(tmp_path / "flat", broker, settings)
    assert result.halted is True
    assert "unexplained" in result.reason
    store.close()

    broker = _broker(settings, snapshot())
    broker.positions_list = [BrokerPosition(3, "SPX", "OPT", -1, "SPXW")]
    result, store = _cycle(tmp_path / "one-leg", broker, settings, book=_spread_book())
    assert result.halted is True
    assert "spread legs" in result.reason
    store.close()


def test_untracked_open_order_latches(tmp_path):
    settings = Settings_submit(True)
    broker = _broker(settings, snapshot())
    broker.extra_open = [TrackedOpenOrder(99, "SPX", "BAG")]
    result, _store = _cycle(tmp_path, broker, settings)
    assert result.halted is True
    assert "untracked open order" in result.reason


def test_exception_latches_and_persists(tmp_path):
    settings = Settings_submit(True)
    broker = _broker(settings, snapshot())
    broker.fail_positions = RuntimeError("feed down")
    result, store = _cycle(tmp_path, broker, settings)
    assert result.halted is True
    assert "feed down" in result.reason
    store.close()
    assert store_for(loaded(tmp_path, settings)).is_halted() is True


def test_outside_session_and_holiday_do_not_enter(tmp_path):
    settings = Settings_submit(True)
    broker = _broker(settings, snapshot())
    morning = datetime(2026, 10, 21, 8, 0, tzinfo=NY)
    result, store = _cycle(tmp_path / "morning", broker, settings, now=morning)
    assert result.halted is False
    assert broker.snapshot_calls == 0
    store.close()
    holiday = _broker(settings, snapshot())
    result, store = _cycle(tmp_path / "holiday", holiday, settings, calendar=Closed())
    assert result.halted is False
    assert holiday.snapshot_calls == 0
    store.close()


def test_daily_loss_latches_and_missing_pnl_does_not_when_flat(tmp_path):
    settings = Settings_submit(True)
    morning = datetime(2026, 10, 21, 8, 0, tzinfo=NY)
    broker = _broker(settings, snapshot())
    broker.pnl = -2500
    result, store = _cycle(tmp_path / "loss", broker, settings, now=morning)
    assert result.halted is True
    assert "daily loss" in result.reason
    store.close()

    quiet = _broker(settings, snapshot())
    quiet.pnl = None
    result, store = _cycle(tmp_path / "flat", quiet, settings, now=morning)
    assert result.halted is False
    store.close()

    held = _broker(settings)
    held.pnl = None
    held.positions_list = [
        BrokerPosition(3, "SPX", "OPT", -1, "SPXW"),
        BrokerPosition(2, "SPX", "OPT", 1, "SPXW"),
    ]
    result, store = _cycle(tmp_path / "open", held, settings, book=_spread_book())
    assert result.halted is True
    assert "pnl unavailable" in result.reason
    store.close()


def test_open_book_hedges_with_mes_and_logs_vega(tmp_path):
    settings = Settings_submit(True)
    broker = _broker(settings)
    _hold_quotes(broker, NOW, short_delta=-0.20, long_delta=-0.02)
    _future(broker)
    broker.positions_list = [
        BrokerPosition(3, "SPX", "OPT", -1, "SPXW"),
        BrokerPosition(2, "SPX", "OPT", 1, "SPXW"),
    ]
    result, store = _cycle(tmp_path, broker, settings, book=_spread_book())
    assert result.halted is False
    assert len(broker.placed) == 1
    hedge = broker.placed[0]
    assert hedge.role == "hedge"
    assert hedge.action == "SELL"
    assert hedge.quantity == 4
    assert hedge.symbol == "MES"
    assert hedge.limit_price == 5800
    assert store.book().hedge_qty == -4
    assert store.book().hedge_con_id == 77
    assert not any(order.role == "vega" for order in store.orders())
    store.close()


def test_hedge_above_the_position_cap_latches_without_an_order(tmp_path):
    settings = Settings_submit(True, hedge={"max_mes_position": 1})
    broker = _broker(settings)
    _hold_quotes(broker, NOW, short_delta=-0.20, long_delta=-0.02)
    broker.positions_list = [
        BrokerPosition(3, "SPX", "OPT", -1, "SPXW"),
        BrokerPosition(2, "SPX", "OPT", 1, "SPXW"),
    ]
    result, _store = _cycle(tmp_path, broker, settings, book=_spread_book())
    assert result.halted is True
    assert "max position" in result.reason
    assert broker.placed == []


def test_profit_target_closes_with_the_opposite_combo(tmp_path):
    settings = Settings_submit(True)
    broker = _broker(settings)
    broker.quotes[3] = _quote(3, 5650, -0.1, 2.1, 2.1, 0.2)
    broker.quotes[2] = _quote(2, 5625, -0.05, 0.1, 0.1, 0.1)
    broker.positions_list = [
        BrokerPosition(3, "SPX", "OPT", -1, "SPXW"),
        BrokerPosition(2, "SPX", "OPT", 1, "SPXW"),
    ]
    result, store = _cycle(tmp_path, broker, settings, book=_spread_book())
    assert result.halted is False
    assert len(broker.placed) == 1
    close = broker.placed[0]
    assert close.role == "close"
    assert close.limit_price == 2
    assert [leg.action for leg in close.legs] == ["BUY", "SELL"]
    assert store.book().spread is None
    assert any(kind == "risk" for kind, _payload in store.events())
    store.close()


def test_stale_greeks_on_an_open_book_latch(tmp_path):
    settings = Settings_submit(True)
    broker = _broker(settings)
    _hold_quotes(broker, NOW - timedelta(seconds=30), short_delta=-0.16, long_delta=-0.12)
    broker.positions_list = [
        BrokerPosition(3, "SPX", "OPT", -1, "SPXW"),
        BrokerPosition(2, "SPX", "OPT", 1, "SPXW"),
    ]
    result, _store = _cycle(tmp_path, broker, settings, book=_spread_book())
    assert result.halted is True
    assert "stale" in result.reason
    assert broker.placed == []


def test_halt_file_latches_and_clear_halt_refuses_while_it_exists(tmp_path):
    settings = Settings_submit(False)
    config = loaded(tmp_path, settings)
    config.resolve(settings.risk.halt_file).parent.mkdir(parents=True, exist_ok=True)
    config.resolve(settings.risk.halt_file).write_text("stop")
    broker = _broker(settings, snapshot())
    result, store = _cycle(tmp_path, broker, settings)
    assert result.halted is True
    assert "halt file" in result.reason
    store.close()
    code = main(["clear-halt", "--config", str(_write_config(tmp_path, settings))])
    assert code == 1


def test_clear_halt_command(tmp_path):
    settings = Settings()
    path = _write_config(tmp_path, settings)
    config = loaded(tmp_path, settings)
    store = store_for(config)
    store.halt("feed down", datetime.now(timezone.utc))
    store.close()
    assert main(["clear-halt", "--config", str(path)]) == 0
    assert store_for(config).is_halted() is False


def test_run_forever_stops_on_halt_without_sleeping(tmp_path):
    settings = Settings_submit(True)
    broker = _broker(settings, snapshot())
    broker.fail_positions = RuntimeError("feed down")
    config = loaded(tmp_path, settings)
    store = store_for(config)

    def fail_sleep(_seconds):
        raise AssertionError("slept after halt")

    result = run_forever(
        clock=lambda: NOW,
        sleep=fail_sleep,
        max_cycles=3,
        loaded=config,
        broker=broker,
        store=store,
        calendar=Weekdays(),
        env={},
    )
    assert result.halted is True
    store.close()


def Settings_submit(submit: bool, **sections):
    return settings_with(broker={"submit_orders": submit}, **sections)


def _write_config(tmp_path, settings):
    path = tmp_path / "config" / "paper.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(settings.model_dump()))
    return path


def _quote(con_id, strike, delta, bid, ask, vega, quoted_at=NOW):
    from tests.helpers import put_quote

    return put_quote(con_id, strike, delta, bid, ask, quoted_at, vega=vega)


def _hold_quotes(broker, quoted_at, short_delta, long_delta):
    broker.quotes[3] = _quote(3, 5650, short_delta, 8, 8.4, 0.40, quoted_at)
    broker.quotes[2] = _quote(2, 5625, long_delta, 4, 4.4, 0.25, quoted_at)


def _future(broker):
    broker.futures = [FutureCandidate(77, "MES", EXPIRY + timedelta(days=20), "MESZ6", "CME", 5)]
    broker.future_quotes[77] = FutureQuote(77, 5800, 5800.25, NOW)
