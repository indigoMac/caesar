from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
import yaml
from pydantic import ValidationError

from options_book.config import Settings, load_settings
from options_book.errors import HaltError, OrdersDisabled
from options_book.risk import (
    XnysCalendar,
    assert_can_submit,
    assert_daily_loss,
    assert_fresh,
    ensure_endpoint,
    in_session,
    reconcile,
)
from options_book.types import BookState, BrokerPosition, OpenSpread
from tests.helpers import EXPIRY, NOW, Weekdays, settings_with

ROOT = Path(__file__).resolve().parents[1]
NY = ZoneInfo("America/New_York")


def _book(spread: OpenSpread | None = None, hedge_qty: int = 0, hedge_con_id: int | None = None) -> BookState:
    return BookState(spread, hedge_qty, hedge_con_id, "MES")


def _spread() -> OpenSpread:
    return OpenSpread(3, 2, 5650, 5625, EXPIRY, 1, 4, 25)


def test_paper_yaml_ships_with_orders_off():
    loaded = load_settings(ROOT / "config" / "paper.yaml")
    assert loaded.settings.broker.submit_orders is False
    assert loaded.settings.broker.live_enabled is False
    assert loaded.settings.broker.host == "127.0.0.1"
    assert loaded.settings.broker.port == 4002
    assert loaded.settings.spread.quantity == 1
    assert loaded.settings.spread.width_points == 25
    assert loaded.settings.spread.short_delta == -0.16
    assert loaded.settings.spread.max_loss_dollars == 2500
    assert loaded.settings.hedge.symbol == "MES"
    assert loaded.settings.hedge.max_mes_position == 10
    assert loaded.settings.loop.poll_seconds == 60


def test_invalid_rules_are_rejected():
    raw = Settings().model_dump()
    raw["spread"]["quantity"] = 2
    with pytest.raises(ValidationError):
        Settings.model_validate(raw)
    raw = Settings().model_dump()
    raw["spread"]["close_dte"] = 30
    with pytest.raises(ValidationError):
        Settings.model_validate(raw)
    dumped = yaml.safe_dump(Settings().model_dump())
    assert "submit_orders: false" in dumped


def test_only_the_paper_gateway_is_open_by_default():
    ensure_endpoint(Settings(), {})
    ensure_endpoint(Settings().model_copy(update={"broker": Settings().broker.model_copy(update={"host": "localhost"})}), {})
    with pytest.raises(HaltError, match="paper gateway"):
        ensure_endpoint(settings_with(broker={"port": 7496}), {"OPTIONS_BOOK_LIVE": "1"})
    with pytest.raises(HaltError, match="paper gateway"):
        ensure_endpoint(settings_with(broker={"port": 7496, "live_enabled": True}), {})
    ensure_endpoint(settings_with(broker={"port": 7496, "live_enabled": True}), {"OPTIONS_BOOK_LIVE": "1"})


def test_submit_gate(tmp_path):
    halt_file = tmp_path / "HALT"
    with pytest.raises(OrdersDisabled):
        assert_can_submit(Settings(), {}, False, halt_file)
    live = settings_with(broker={"submit_orders": True})
    assert_can_submit(live, {}, False, halt_file)
    with pytest.raises(HaltError, match="latch"):
        assert_can_submit(live, {}, True, halt_file)
    halt_file.write_text("stop")
    with pytest.raises(HaltError, match="halt file"):
        assert_can_submit(live, {}, False, halt_file)


def test_session_window_and_holiday():
    settings = Settings()
    calendar = Weekdays()
    assert in_session(NOW, settings, calendar) is True
    assert in_session(datetime(2026, 10, 21, 13, 45, tzinfo=ZoneInfo("UTC")), settings, calendar) is True
    assert in_session(datetime(2026, 10, 21, 13, 44, tzinfo=ZoneInfo("UTC")), settings, calendar) is False
    assert in_session(NOW.replace(hour=9, minute=44), settings, calendar) is False
    assert in_session(NOW.replace(hour=9, minute=45), settings, calendar) is True
    assert in_session(NOW.replace(hour=15, minute=45), settings, calendar) is True
    assert in_session(NOW.replace(hour=15, minute=46), settings, calendar) is False
    assert in_session(datetime(2026, 10, 24, 14, 0, tzinfo=NY), settings, calendar) is False
    with pytest.raises(HaltError, match="naive"):
        in_session(datetime(2026, 10, 21, 14, 0), settings, calendar)
    xnsy = XnysCalendar()
    assert xnsy.is_session(datetime(2024, 5, 27).date()) is False
    assert xnsy.is_session(datetime(2024, 5, 28).date()) is True


def test_stale_and_daily_loss_gates():
    from datetime import timedelta

    assert_fresh(NOW, NOW, 5, "short leg")
    assert_fresh(NOW - timedelta(seconds=5), NOW, 5, "short leg")
    with pytest.raises(HaltError, match="stale"):
        assert_fresh(NOW - timedelta(seconds=6), NOW, 5, "short leg")
    with pytest.raises(HaltError, match="no timestamp"):
        assert_fresh(None, NOW, 5, "short leg")
    assert_daily_loss(None, 2500, False)
    with pytest.raises(HaltError, match="unavailable"):
        assert_daily_loss(None, 2500, True)
    with pytest.raises(HaltError, match="daily loss"):
        assert_daily_loss(-2500, 2500, False)


def test_reconcile_matches_a_spread_and_halts_on_strays():
    spread = _spread()
    positions = [
        BrokerPosition(3, "SPX", "OPT", -1, "SPXW"),
        BrokerPosition(2, "SPX", "OPT", 1, "SPXW"),
    ]
    reconcile(_book(spread), positions)
    reconcile(_book(None, -2, 77), [BrokerPosition(77, "MES", "FUT", -2)])
    with pytest.raises(HaltError, match="unexplained"):
        reconcile(_book(), [BrokerPosition(3, "SPX", "OPT", -1)])
    with pytest.raises(HaltError, match="unexplained"):
        reconcile(_book(), [BrokerPosition(77, "MES", "FUT", -1)])
    with pytest.raises(HaltError, match="spread legs"):
        reconcile(_book(spread), [BrokerPosition(3, "SPX", "OPT", -1)])
    with pytest.raises(HaltError, match="hedge quantity"):
        reconcile(_book(spread, -2, 77), positions + [BrokerPosition(77, "MES", "FUT", -1)])
    with pytest.raises(HaltError, match="fractional"):
        reconcile(_book(), [BrokerPosition(3, "SPX", "OPT", 0.5)])
    reconcile(_book(), [BrokerPosition(99, "AAPL", "STK", 10)])
