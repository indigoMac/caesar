import sys
import types

import pytest

from options_book.broker import IbBroker, map_order_status, parse_contract_date, parse_multiplier
from options_book.config import Settings
from options_book.errors import HaltError, OrdersDisabled
from options_book.types import ComboIntent, ComboLegSpec
from tests.helpers import EXPIRY, settings_with


def test_order_status_map():
    assert map_order_status("Submitted", 0, 1, 1) == "submitted"
    assert map_order_status("PendingSubmit", 0, 1, 1) == "submitted"
    assert map_order_status("Filled", 1, 0, 1) == "filled"
    assert map_order_status("Cancelled", 0, 1, 1) == "cancelled"
    assert map_order_status("Cancelled", 1, 0, 1) == "filled"
    assert map_order_status("Inactive", 0, 1, 1) == "rejected"
    assert map_order_status("Submitted", 1, 1, 2) == "partial"
    with pytest.raises(HaltError, match="unknown order status"):
        map_order_status("Weird", 0, 1, 1)


def test_contract_dates_and_multipliers():
    assert parse_contract_date("20261218", "202612") == EXPIRY.replace(year=2026, month=12, day=18)
    assert parse_contract_date("", "20261218") == EXPIRY.replace(year=2026, month=12, day=18)
    with pytest.raises(HaltError, match="last trade date"):
        parse_contract_date("", "202612")
    assert parse_multiplier("5") == 5
    with pytest.raises(HaltError, match="multiplier"):
        parse_multiplier("")


def test_connect_refuses_a_live_port_before_opening_a_socket(monkeypatch):
    class Boom:
        def __init__(self, *_args, **_kwargs):
            raise AssertionError("socket")

    module = types.ModuleType("ib_async")
    module.IB = Boom
    monkeypatch.setitem(sys.modules, "ib_async", module)
    settings = settings_with(broker={"port": 7496})
    with pytest.raises(HaltError, match="paper gateway"):
        IbBroker(settings).connect()


def test_place_order_refuses_a_dry_run_without_connecting(tmp_path):
    broker = IbBroker(Settings())
    with pytest.raises(OrdersDisabled):
        broker.place_order(_intent(), halted=False, halt_file=tmp_path / "HALT", env={})


def test_place_order_requires_a_connection_when_submit_is_on(tmp_path):
    settings = settings_with(broker={"submit_orders": True})
    broker = IbBroker(settings)
    with pytest.raises(HaltError, match="not connected"):
        broker.place_order(_intent(), halted=False, halt_file=tmp_path / "HALT", env={})


def _intent() -> ComboIntent:
    return ComboIntent(
        role="entry",
        order_action="BUY",
        quantity=1,
        limit_price=-4,
        legs=(ComboLegSpec(3, "SELL", 1), ComboLegSpec(2, "BUY", 1)),
        expiry=EXPIRY,
        short_con_id=3,
        long_con_id=2,
        short_strike=5650,
        long_strike=5625,
        premium_points=4,
        width_points=25,
        max_loss_dollars=2100,
        price_meaning="credit",
    )
