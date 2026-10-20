from datetime import date, timedelta
from decimal import Decimal

import pytest

from options_book.delta import (
    GreekLeg,
    decide_hedge,
    dollar_delta,
    hedge_limit_price,
    net_vega,
    point_value,
    round_half_away_from_zero,
    select_future,
)
from options_book.errors import HaltError
from options_book.types import FutureCandidate


def _future(con_id: int, expiry: date, multiplier: float = 5, symbol: str = "MES") -> FutureCandidate:
    return FutureCandidate(con_id, symbol, expiry, f"{symbol}{con_id}", "CME", multiplier)


def test_credit_spread_delta_sells_two_mes():
    legs = (
        GreekLeg(-1, -0.16, 100, 0.4),
        GreekLeg(1, -0.08, 100, 0.2),
    )
    assert dollar_delta(legs) == Decimal("8")
    assert net_vega(legs) == Decimal("-20")
    decision = decide_hedge(dollar_delta(legs), 0, point_value("MES"), 1, 1, 10)
    assert decision.trade_qty == -2
    assert decision.halt is False
    assert decision.reason == "rebalance"


def test_existing_hedge_inside_the_band_is_left_alone():
    decision = decide_hedge(Decimal("8"), -2, point_value("MES"), 1, 1, 10)
    assert decision.trade_qty == 0
    assert decision.reason == "inside band"


def test_one_lot_is_inside_a_one_es_band():
    decision = decide_hedge(Decimal("8"), 0, point_value("ES"), 1, 1, 10)
    assert decision.trade_qty == 0
    assert decision.reason == "inside band"


def test_band_edge_and_rounding():
    assert round_half_away_from_zero(Decimal("1.5")) == 2
    assert round_half_away_from_zero(Decimal("-1.5")) == -2
    assert round_half_away_from_zero(Decimal("1.49")) == 1
    at_band = decide_hedge(Decimal("5"), 0, point_value("MES"), 1, 1, 10)
    assert at_band.trade_qty == 0
    just_outside = decide_hedge(Decimal("5.05"), 0, point_value("MES"), 1, 1, 10)
    assert just_outside.trade_qty == -1
    one_and_a_half = decide_hedge(Decimal("7.5"), 0, point_value("MES"), 1, 1, 10)
    assert one_and_a_half.trade_qty == -2


def test_position_cap_halts_when_the_book_would_remain_outside_the_band():
    decision = decide_hedge(Decimal("18"), 0, point_value("MES"), 1, 1, 1)
    assert decision.halt is True
    assert decision.trade_qty == 0
    assert decision.reason == "hedge exceeds max position"


def test_small_adjustment_is_not_sent():
    decision = decide_hedge(Decimal("6"), 0, point_value("MES"), 1, 2, 10)
    assert decision.trade_qty == 0
    assert decision.halt is False
    assert decision.reason == "adjustment below minimum"


def test_hedge_limit_is_the_inside_quote():
    assert hedge_limit_price(-1, 5800, 5800.25, 0.25, 1) == 5800
    assert hedge_limit_price(1, 5800, 5800.25, 0.25, 1) == 5800.25
    with pytest.raises(HaltError, match="wider than the cap"):
        hedge_limit_price(-1, 5800, 5802, 0.25, 1)
    with pytest.raises(HaltError, match="not two-sided"):
        hedge_limit_price(1, 5800.25, 5800, 0.25, 1)


def test_future_selection_uses_the_nearest_contract_after_the_option():
    option_expiry = date(2026, 11, 30)
    chosen = select_future(
        [
            _future(1, option_expiry - timedelta(days=5)),
            _future(2, date(2026, 12, 18)),
            _future(3, date(2027, 3, 19)),
            _future(4, date(2026, 12, 18), multiplier=50, symbol="ES"),
        ],
        after=option_expiry,
        symbol="MES",
    )
    assert chosen.con_id == 2


def test_no_future_after_the_option_halts():
    with pytest.raises(HaltError, match="no MES future"):
        select_future([_future(1, date(2026, 11, 1))], after=date(2026, 11, 30), symbol="MES")
