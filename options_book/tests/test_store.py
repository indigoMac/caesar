from datetime import datetime, timedelta, timezone

import pytest

from options_book.errors import HaltError
from options_book.store import Store
from options_book.types import BookState, OpenSpread
from tests.helpers import EXPIRY, NOW


def _store(tmp_path) -> Store:
    return Store(tmp_path / "state.sqlite", "MES")


def test_halt_survives_reopen_until_cleared(tmp_path):
    store = _store(tmp_path)
    store.halt("feed down", NOW)
    store.halt("later", NOW + timedelta(seconds=1))
    store.close()
    reopened = _store(tmp_path)
    assert reopened.is_halted() is True
    assert reopened.halt_reason() == "feed down"
    assert any(kind == "halt_repeat" for kind, _payload in reopened.events())
    reopened.clear_halt(NOW)
    assert reopened.is_halted() is False
    reopened.close()


def test_book_round_trip_and_symbol_guard(tmp_path):
    store = _store(tmp_path)
    spread = OpenSpread(3, 2, 5650, 5625, EXPIRY, 1, 4.0, 25)
    store.save_book(BookState(spread, -2, 77, "MES"), NOW)
    loaded = store.book()
    assert loaded.spread == spread
    assert loaded.hedge_qty == -2
    assert loaded.hedge_con_id == 77
    with pytest.raises(HaltError, match="hedge symbol"):
        store.save_book(BookState(spread, 0, None, "ES"), NOW)
    store.close()


def test_order_transitions(tmp_path):
    store = _store(tmp_path)
    order_id = store.insert_order(
        ts=NOW,
        role="entry",
        status="submitted",
        broker_order_id=10,
        payload={"quantity": 1},
    )
    assert store.broker_order_ids() == {10}
    assert len(store.working_orders()) == 1
    store.update_status(order_id, "cancel_requested", NOW)
    store.update_status(order_id, "cancelled", NOW + timedelta(seconds=1))
    assert store.working_orders() == []
    with pytest.raises(HaltError, match="cannot move"):
        store.update_status(order_id, "filled", NOW)
    suppressed = store.insert_order(
        ts=NOW,
        role="hedge",
        status="suppressed",
        broker_order_id=None,
        payload={"quantity": 1},
    )
    assert store.get_order(suppressed).status == "suppressed"
    store.close()


def test_flat_book_can_change_hedge_symbol(tmp_path):
    path = tmp_path / "state.sqlite"
    Store(path, "MES").close()
    switched = Store(path, "ES")
    assert switched.book().hedge_symbol == "ES"
    switched.save_book(BookState(None, -1, 5, "ES"), datetime.now(timezone.utc))
    switched.close()
    with pytest.raises(HaltError, match="hedge symbol"):
        Store(path, "MES")
