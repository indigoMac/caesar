from datetime import date, timedelta

import pytest

from options_book.config import Settings
from options_book.errors import SelectionError
from options_book.spreads import (
    build_close_intent,
    credit_from_fill,
    evaluate_close,
    select_put_spread,
)
from options_book.types import OpenSpread
from tests.helpers import AS_OF, EXPIRY, NOW, put_quote, tradable_chain


def test_entry_combo_is_a_buy_at_a_negative_credit():
    selected = select_put_spread(tradable_chain(), 5800, AS_OF, Settings().spread)
    intent = selected.intent
    assert intent.order_action == "BUY"
    assert intent.price_meaning == "credit"
    assert intent.limit_price == -4
    assert intent.premium_points == 4
    assert intent.legs[0].action == "SELL"
    assert intent.legs[0].con_id == 3
    assert intent.legs[1].action == "BUY"
    assert intent.legs[1].con_id == 2
    assert intent.max_loss_dollars == 2100
    assert selected.short.strike == 5650
    assert selected.long.strike == 5625


def test_credit_tie_rounds_toward_a_smaller_credit():
    chain = (
        put_quote(1, 5600, -0.05, 1, 1.2),
        put_quote(2, 5625, -0.10, 2, 2),
        put_quote(3, 5650, -0.16, 6, 6.05),
    )
    selected = select_put_spread(chain, 5800, AS_OF, Settings().spread)
    assert selected.intent.premium_points == 4
    assert selected.intent.limit_price == -4


def test_max_loss_cap_refuses_the_spread():
    rules = Settings().spread.model_copy(update={"max_loss_dollars": 100})
    with pytest.raises(SelectionError, match="max loss"):
        select_put_spread(tradable_chain(), 5800, AS_OF, rules)


def test_minimum_credit_refuses_a_dust_spread():
    rules = Settings().spread.model_copy(update={"min_credit_points": 10})
    with pytest.raises(SelectionError, match="minimum"):
        select_put_spread(tradable_chain(), 5800, AS_OF, rules)


def test_missing_long_strike_refuses():
    chain = (
        put_quote(1, 5600, -0.05, 1, 1.2),
        put_quote(3, 5650, -0.16, 8, 8.4),
    )
    with pytest.raises(SelectionError, match="long strike"):
        select_put_spread(chain, 5800, AS_OF, Settings().spread)


def test_short_strike_on_the_scan_floor_is_refused():
    chain = (
        put_quote(1, 5600, -0.16, 8, 8.4),
        put_quote(2, 5700, -0.40, 14, 14.4),
    )
    with pytest.raises(SelectionError, match="bottom of the scanned chain"):
        select_put_spread(chain, 5800, AS_OF, Settings().spread)


def test_itm_put_is_not_the_short_leg():
    chain = (
        put_quote(1, 5600, -0.05, 1, 1.2),
        put_quote(2, 5625, -0.10, 4, 4.4),
        put_quote(3, 5650, -0.20, 8, 8.4),
        put_quote(9, 5900, -0.16, 30, 30.4),
    )
    selected = select_put_spread(chain, 5800, AS_OF, Settings().spread)
    assert selected.short.strike == 5650


def test_calls_are_ignored():
    chain = tradable_chain() + (put_quote(8, 5650, -0.16, 8, 8.4, right="C"),)
    selected = select_put_spread(chain, 5800, AS_OF, Settings().spread)
    assert selected.short.right == "P"


def test_expiry_tie_picks_the_sooner_date():
    sooner = AS_OF + timedelta(days=38)
    later = AS_OF + timedelta(days=42)
    chain = (
        put_quote(1, 5600, -0.05, 1, 1.2, expiry=sooner),
        put_quote(2, 5625, -0.10, 4, 4.4, expiry=sooner),
        put_quote(3, 5650, -0.16, 8, 8.4, expiry=sooner),
        put_quote(4, 5600, -0.05, 1, 1.2, expiry=later),
        put_quote(5, 5625, -0.10, 4, 4.4, expiry=later),
        put_quote(6, 5700, -0.16, 8, 8.4, expiry=later),
    )
    selected = select_put_spread(chain, 5800, AS_OF, Settings().spread)
    assert selected.intent.expiry == sooner


def test_expiry_outside_the_window_is_ignored():
    near = AS_OF + timedelta(days=10)
    chain = tuple(put_quote(quote.con_id, quote.strike, quote.delta, quote.bid, quote.ask, expiry=near) for quote in tradable_chain())
    with pytest.raises(SelectionError, match="dte window"):
        select_put_spread(chain, 5800, AS_OF, Settings().spread)


def test_close_reasons_and_opposite_combo():
    spread = OpenSpread(3, 2, 5650, 5625, EXPIRY, 1, 4, 25)
    rules = Settings().spread
    hold = evaluate_close(spread, put_quote(3, 5650, -0.16, 8, 8.4), put_quote(2, 5625, -0.1, 4, 4.4), AS_OF, rules)
    assert hold.should_close is False
    profit = evaluate_close(spread, put_quote(3, 5650, -0.1, 2.1, 2.1), put_quote(2, 5625, -0.05, 0.1, 0.1), AS_OF, rules)
    assert profit.reason == "profit"
    loss = evaluate_close(spread, put_quote(3, 5650, -0.4, 16, 16), put_quote(2, 5625, -0.2, 4, 4), AS_OF, rules)
    assert loss.reason == "loss"
    soon = OpenSpread(3, 2, 5650, 5625, AS_OF + timedelta(days=21), 1, 4, 25)
    timed = evaluate_close(soon, put_quote(3, 5650, -0.16, 8, 8.4), put_quote(2, 5625, -0.1, 4, 4.4), AS_OF, rules)
    assert timed.reason == "dte"
    closing = build_close_intent(spread, put_quote(3, 5650, -0.1, 2.1, 2.1), put_quote(2, 5625, -0.05, 0.1, 0.1), 0.05)
    assert closing.order_action == "BUY"
    assert closing.price_meaning == "debit"
    assert closing.limit_price == 2
    assert closing.legs[0].action == "BUY"
    assert closing.legs[1].action == "SELL"


def test_fill_credit_accepts_a_negative_or_positive_report():
    assert credit_from_fill(-4, -4.1, 25) == pytest.approx(4.1)
    assert credit_from_fill(-4, 4.5, 25) == pytest.approx(4.5)
    with pytest.raises(Exception, match="missing"):
        credit_from_fill(-4, None, 25)
    with pytest.raises(Exception, match="outside"):
        credit_from_fill(-4, 30, 25)
    with pytest.raises(Exception, match="ambiguous"):
        credit_from_fill(4, -4, 25)


def test_as_of_is_a_date():
    assert AS_OF == date(2026, 10, 21)
    assert NOW.tzinfo is not None
