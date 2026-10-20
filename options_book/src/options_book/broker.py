"""IB Gateway adapter. The process stores no username and no password."""

from __future__ import annotations

import os
import time
from datetime import date, datetime
from pathlib import Path
from typing import Mapping

from options_book.config import Settings
from options_book.errors import HaltError, SelectionError
from options_book.risk import assert_can_submit, ensure_endpoint
from options_book.spreads import choose_expiry
from options_book.types import (
    BrokerPosition,
    ComboIntent,
    EntrySnapshot,
    FutureCandidate,
    FutureQuote,
    HedgeIntent,
    MarginCheck,
    OptionQuote,
    OrderStatus,
    TrackedOpenOrder,
    is_finite,
)

_WORKING = {"PendingSubmit", "PreSubmitted", "Submitted", "PendingCancel", "ApiPending"}
_CANCELLED = {"Cancelled", "ApiCancelled"}
_TRACKED_OPTIONS = {"SPX", "SPXW"}
_TRACKED_FUTURES = {"MES", "ES"}


def map_order_status(ib_status: str, filled: int, remaining: int, total: int) -> str:
    if ib_status == "Filled" or (total > 0 and filled >= total):
        return "filled"
    if ib_status in _CANCELLED:
        return "cancelled"
    if ib_status == "Inactive":
        return "partial" if filled > 0 else "rejected"
    if filled > 0:
        return "partial"
    if ib_status in _WORKING:
        return "submitted"
    raise HaltError(f"unknown order status: {ib_status}")


def parse_contract_date(real_expiration: str, last_trade: str) -> date:
    for raw in (real_expiration, last_trade):
        text = (raw or "").strip()
        if len(text) >= 8 and text[:8].isdigit():
            return datetime.strptime(text[:8], "%Y%m%d").date()
    raise HaltError("future contract is missing a last trade date")


def parse_multiplier(raw: str) -> float:
    text = (raw or "").strip()
    if text == "":
        raise HaltError("contract is missing a multiplier")
    try:
        value = float(text)
    except ValueError as exc:
        raise HaltError("contract multiplier is not a number") from exc
    if not is_finite(value) or value <= 0:
        raise HaltError("contract multiplier is not a number")
    return value


class IbBroker:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._ib = None
        self._account = ""
        self._pnl = None
        self._trades: dict[int, object] = {}

    def connect(self) -> None:
        ensure_endpoint(self.settings, os.environ)
        ib_mod = _load_ib()
        ib = ib_mod.IB()
        self._ib = ib
        try:
            ib.connect(
                self.settings.broker.host,
                self.settings.broker.port,
                clientId=self.settings.broker.client_id,
                timeout=15,
                readonly=not self.settings.broker.submit_orders,
            )
            if not ib.isConnected():
                raise HaltError("broker connection failed")
            self._account = self._select_account(ib)
        except Exception:
            self.disconnect()
            raise

    def _select_account(self, ib) -> str:
        accounts = [account for account in ib.managedAccounts() if account]
        configured = self.settings.broker.account
        if configured:
            if configured not in accounts:
                raise HaltError("configured account is not on this login")
            return configured
        if len(accounts) != 1:
            raise HaltError("login manages more than one account; set broker.account")
        return accounts[0]

    def disconnect(self) -> None:
        ib = self._ib
        self._ib = None
        if ib is not None and ib.isConnected():
            ib.disconnect()

    def positions(self) -> list[BrokerPosition]:
        ib = self._require_ib()
        found: list[BrokerPosition] = []
        for row in ib.positions():
            if row.account != self._account:
                continue
            contract = row.contract
            found.append(
                BrokerPosition(
                    con_id=int(contract.conId),
                    symbol=contract.symbol,
                    sec_type=contract.secType,
                    quantity=float(row.position),
                    trading_class=contract.tradingClass or "",
                )
            )
        return found

    def daily_pnl(self) -> float | None:
        ib = self._require_ib()
        if self._pnl is None:
            self._pnl = ib.reqPnL(self._account)
        if not self._wait(lambda: is_finite(self._pnl.dailyPnL), 2):
            return None
        return float(self._pnl.dailyPnL)

    def available_funds(self) -> float | None:
        ib = self._require_ib()

        def ready() -> bool:
            return any(
                row.tag == "AvailableFunds" and row.currency == "USD"
                for row in ib.accountSummary(self._account)
            )

        self._wait(ready, 3)
        for row in ib.accountSummary(self._account):
            if row.tag == "AvailableFunds" and row.currency == "USD":
                return _money(row.value)
        return None

    def entry_snapshot(self, as_of: date) -> EntrySnapshot:
        index = self._qualify_spx()
        spot = self._live_index_price(index)
        expiry, strikes = self._spxw_strikes(index, spot, as_of)
        return EntrySnapshot(spot=spot, options=tuple(self._stream_puts(expiry, strikes)))

    def quote_legs(self, con_ids: list[int]) -> list[OptionQuote]:
        from ib_async import Contract

        ib = self._require_ib()
        contracts = []
        for con_id in con_ids:
            qualified = ib.qualifyContracts(Contract(conId=con_id, secType="OPT", exchange="CBOE", currency="USD"))
            if qualified and qualified[0].conId:
                contracts.append(qualified[0])
        return self._stream_contracts(contracts)

    def list_futures(self, symbol: str) -> list[FutureCandidate]:
        from ib_async import Future

        details = self._require_ib().reqContractDetails(Future(symbol, exchange="CME", currency="USD"))
        found: list[FutureCandidate] = []
        for item in details:
            contract = item.contract
            if contract is None or not contract.conId or contract.symbol != symbol:
                continue
            found.append(
                FutureCandidate(
                    con_id=int(contract.conId),
                    symbol=contract.symbol,
                    expiry=parse_contract_date(item.realExpirationDate, contract.lastTradeDateOrContractMonth),
                    local_symbol=contract.localSymbol,
                    exchange=contract.exchange or "CME",
                    multiplier=parse_multiplier(contract.multiplier),
                )
            )
        if not found:
            raise HaltError(f"no {symbol} futures were returned")
        return found

    def future_quote(self, con_id: int) -> FutureQuote:
        from ib_async import Contract

        ib = self._require_ib()
        qualified = ib.qualifyContracts(Contract(conId=con_id, secType="FUT", exchange="CME", currency="USD"))
        if not qualified or not qualified[0].conId:
            raise HaltError("could not qualify the hedge future")
        contract = qualified[0]
        ticker = ib.reqMktData(contract, "", False, False)
        try:
            self._wait(lambda: _two_sided(ticker), 5)
            bid = _px(ticker.bid)
            ask = _px(ticker.ask)
            if bid is None or ask is None:
                raise HaltError("future quote is not two-sided")
            return FutureQuote(con_id=int(contract.conId), bid=bid, ask=ask, quoted_at=_stamp(ticker))
        finally:
            ib.cancelMktData(contract)

    def what_if(
        self,
        intent: ComboIntent,
        *,
        halted: bool,
        halt_file: Path,
        env: Mapping[str, str],
    ) -> MarginCheck:
        assert_can_submit(self.settings, env, halted, halt_file)
        trade = self._transmit(intent, what_if=True)
        try:
            self._wait(lambda: _margin_ready(trade), 5)
            margin = _money(getattr(trade.orderState, "initMarginChange", None))
            if margin is None:
                return MarginCheck(None, "what-if margin unavailable")
            return MarginCheck(margin, None)
        finally:
            status = trade.orderStatus.status
            if status not in {"Filled", "Cancelled", "ApiCancelled", "Inactive"}:
                self._require_ib().cancelOrder(trade.order)

    def place_order(
        self,
        intent: ComboIntent | HedgeIntent,
        *,
        halted: bool,
        halt_file: Path,
        env: Mapping[str, str],
    ) -> int:
        assert_can_submit(self.settings, env, halted, halt_file)
        trade = self._transmit(intent, what_if=False)
        order_id = int(trade.order.orderId)
        if order_id <= 0:
            raise HaltError("broker did not assign an order id")
        self._trades[order_id] = trade
        return order_id

    def cancel_order(self, broker_order_id: int) -> None:
        trade = self._find_trade(broker_order_id)
        if trade is None:
            raise HaltError(f"order {broker_order_id} is not known to the broker session")
        self._require_ib().cancelOrder(trade.order)

    def order_status(self, broker_order_id: int) -> OrderStatus:
        trade = self._find_trade(broker_order_id)
        if trade is None:
            raise HaltError(f"order {broker_order_id} is not known to the broker session")
        filled = int(round(trade.orderStatus.filled))
        remaining = int(round(trade.orderStatus.remaining))
        total = filled + remaining
        if total <= 0:
            total = int(round(trade.order.totalQuantity))
        status = map_order_status(trade.orderStatus.status, filled, remaining, total)
        return OrderStatus(
            broker_order_id=broker_order_id,
            status=status,
            filled=filled,
            remaining=remaining,
            avg_price=_avg_price(trade.orderStatus.avgFillPrice),
        )

    def open_tracked_orders(self) -> list[TrackedOpenOrder]:
        ib = self._require_ib()
        ib.reqAllOpenOrders()
        ib.sleep(1)
        found: list[TrackedOpenOrder] = []
        for trade in ib.openTrades():
            contract = trade.contract
            if _tracked_contract(contract):
                found.append(
                    TrackedOpenOrder(
                        broker_order_id=int(trade.order.orderId),
                        symbol=contract.symbol,
                        sec_type=contract.secType,
                    )
                )
        return found

    def _transmit(self, intent: ComboIntent | HedgeIntent, *, what_if: bool):
        contract = _bag(intent) if isinstance(intent, ComboIntent) else _future_contract(intent)
        action = intent.order_action if isinstance(intent, ComboIntent) else intent.action
        order = self._limit(action, intent.quantity, intent.limit_price, what_if)
        trade = self._require_ib().placeOrder(contract, order)
        return trade

    def _limit(self, action: str, quantity: int, limit: float, what_if: bool):
        from ib_async import LimitOrder

        order = LimitOrder(action, quantity, limit)
        order.tif = "DAY"
        order.transmit = True
        order.outsideRth = False
        order.orderRef = "options_book"
        order.whatIf = what_if
        order.eTradeOnly = False
        order.firmQuoteOnly = False
        if self._account:
            order.account = self._account
        return order

    def _qualify_spx(self):
        from ib_async import Index

        qualified = self._require_ib().qualifyContracts(Index("SPX", "CBOE", "USD"))
        if not qualified or not qualified[0].conId:
            raise HaltError("could not qualify SPX")
        return qualified[0]

    def _live_index_price(self, index) -> float:
        ib = self._require_ib()
        ticker = ib.reqMktData(index, "", False, False)
        try:
            self._wait(lambda: _positive_price(ticker), 5)
            price = ticker.marketPrice()
            if not _positive_value(price):
                price = ticker.last
            if not _positive_value(price) or ticker.time is None:
                raise SelectionError("SPX price is not live")
            return float(price)
        finally:
            ib.cancelMktData(index)

    def _spxw_strikes(self, index, spot: float, as_of: date) -> tuple[date, list[float]]:
        params = self._require_ib().reqSecDefOptParams(index.symbol, "", index.secType, index.conId)
        rows = [row for row in params if row.tradingClass == "SPXW" and row.exchange == "CBOE"]
        if not rows:
            raise HaltError("SPXW chain is not available")
        chain = max(rows, key=lambda row: len(row.strikes))
        expiries: list[date] = []
        for raw in chain.expirations:
            if len(raw) >= 8 and raw[:8].isdigit():
                expiries.append(datetime.strptime(raw[:8], "%Y%m%d").date())
        rules = self.settings.spread
        expiry = choose_expiry(expiries, as_of, rules.target_dte, rules.min_dte, rules.max_dte)
        low = spot - rules.scan_points_below_spot
        strikes = sorted(float(strike) for strike in chain.strikes if low <= float(strike) < spot)
        if not strikes:
            raise SelectionError("no OTM put strikes in the scan window")
        return expiry, strikes

    def _stream_puts(self, expiry: date, strikes: list[float]) -> list[OptionQuote]:
        from ib_async import Option

        contracts = [
            Option(
                "SPX",
                expiry.strftime("%Y%m%d"),
                strike,
                "P",
                "CBOE",
                "100",
                "USD",
                tradingClass="SPXW",
            )
            for strike in strikes
        ]
        qualified = []
        ib = self._require_ib()
        for batch in _chunks(contracts, 40):
            qualified.extend(ib.qualifyContracts(*batch))
        by_key = {
            _strike_key(contract.strike): contract
            for contract in qualified
            if contract.conId
        }
        live = self._stream_contracts(list(by_key.values()))
        by_con = {quote.con_id: quote for quote in live}
        quotes: list[OptionQuote] = []
        for strike in strikes:
            contract = by_key.get(_strike_key(strike))
            if contract is None or contract.conId not in by_con:
                quotes.append(_empty_put(expiry, strike))
                continue
            quotes.append(by_con[contract.conId])
        return quotes

    def _stream_contracts(self, contracts: list) -> list[OptionQuote]:
        if not contracts:
            return []
        ib = self._require_ib()
        pairs = [(contract, ib.reqMktData(contract, "106", False, False)) for contract in contracts]
        try:
            self._wait(lambda: all(ticker.modelGreeks is not None for _contract, ticker in pairs), 8)
            return [_quote_from_ticker(contract, ticker) for contract, ticker in pairs]
        finally:
            for contract, _ticker in pairs:
                ib.cancelMktData(contract)

    def _find_trade(self, order_id: int):
        known = self._trades.get(order_id)
        if known is not None:
            return known
        ib = self._require_ib()
        for trade in ib.trades():
            if trade.order.orderId == order_id:
                self._trades[order_id] = trade
                return trade
        ib.reqAllOpenOrders()
        ib.sleep(1)
        for trade in ib.openTrades():
            if trade.order.orderId == order_id:
                self._trades[order_id] = trade
                return trade
        return None

    def _require_ib(self):
        if self._ib is None or not self._ib.isConnected():
            raise HaltError("broker is not connected")
        return self._ib

    def _wait(self, predicate, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while True:
            if predicate():
                return True
            if time.monotonic() >= deadline:
                return False
            self._require_ib().sleep(0.2)


def _load_ib():
    try:
        import ib_async
    except ImportError as exc:
        raise HaltError("ib_async is not installed") from exc
    return ib_async


def _bag(intent: ComboIntent):
    from ib_async import ComboLeg, Contract

    bag = Contract(secType="BAG", symbol="SPX", exchange="CBOE", currency="USD")
    bag.comboLegs = [
        ComboLeg(conId=leg.con_id, ratio=leg.ratio, action=leg.action, exchange="CBOE")
        for leg in intent.legs
    ]
    return bag


def _future_contract(intent: HedgeIntent):
    from ib_async import Future

    return Future(symbol=intent.symbol, exchange=intent.exchange, currency="USD", conId=intent.con_id)


def _quote_from_ticker(contract, ticker) -> OptionQuote:
    greeks = ticker.modelGreeks
    expiry = _expiry_of(contract)
    return OptionQuote(
        con_id=int(contract.conId),
        expiry=expiry,
        strike=float(contract.strike),
        right=contract.right or "P",
        delta=_greek(greeks, "delta"),
        bid=_px(ticker.bid),
        ask=_px(ticker.ask),
        vega=_greek(greeks, "vega"),
        quoted_at=_stamp(ticker),
    )


def _empty_put(expiry: date, strike: float) -> OptionQuote:
    return OptionQuote(0, expiry, strike, "P", None, None, None, None, None)


def _expiry_of(contract) -> date:
    raw = contract.lastTradeDateOrContractMonth or ""
    if len(raw) >= 8 and raw[:8].isdigit():
        return datetime.strptime(raw[:8], "%Y%m%d").date()
    raise HaltError("option contract is missing an expiry")


def _greek(greeks, name: str) -> float | None:
    if greeks is None:
        return None
    value = getattr(greeks, name, None)
    if not is_finite(value):
        return None
    return float(value)


def _px(value) -> float | None:
    if not is_finite(value) or value < 0:
        return None
    return float(value)


def _stamp(ticker) -> datetime | None:
    stamp = getattr(ticker, "time", None)
    if isinstance(stamp, datetime):
        return stamp
    return None


def _positive_price(ticker) -> bool:
    return _positive_value(ticker.marketPrice()) or _positive_value(getattr(ticker, "last", None))


def _positive_value(value) -> bool:
    return is_finite(value) and value > 0


def _two_sided(ticker) -> bool:
    return _px(ticker.bid) is not None and _px(ticker.ask) is not None


def _money(value) -> float | None:
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    if text == "":
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if not is_finite(number):
        return None
    return number


def _avg_price(value) -> float | None:
    if not is_finite(value) or value == 0 or abs(value) > 1e100:
        return None
    return float(value)


def _margin_ready(trade) -> bool:
    if _money(getattr(trade.orderState, "initMarginChange", None)) is not None:
        return True
    return trade.orderStatus.status in {"Filled", "Cancelled", "ApiCancelled", "Inactive"}


def _tracked_contract(contract) -> bool:
    if contract.secType == "BAG" and contract.symbol in _TRACKED_OPTIONS:
        return True
    if contract.secType == "OPT" and contract.symbol in _TRACKED_OPTIONS:
        return True
    return contract.secType == "FUT" and contract.symbol in _TRACKED_FUTURES


def _strike_key(value: float) -> int:
    return int(round(float(value) * 1000))


def _chunks(items: list, size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]
