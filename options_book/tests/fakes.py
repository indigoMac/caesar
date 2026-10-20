"""In-memory broker for the cycle tests. No socket."""

from __future__ import annotations

from options_book.errors import HaltError, SelectionError
from options_book.risk import assert_can_submit
from options_book.types import (
    BrokerPosition,
    ComboIntent,
    EntrySnapshot,
    FutureQuote,
    HedgeIntent,
    MarginCheck,
    OptionQuote,
    OrderStatus,
    TrackedOpenOrder,
)


class FakeBroker:
    def __init__(self, settings) -> None:
        self.settings = settings
        self.positions_list: list[BrokerPosition] = []
        self.snapshot: EntrySnapshot | None = None
        self.snapshot_error: Exception | None = None
        self.quotes: dict[int, OptionQuote] = {}
        self.futures: list = []
        self.future_quotes: dict[int, FutureQuote] = {}
        self.pnl: float | None = 0.0
        self.funds: float | None = 100_000.0
        self.margin = MarginCheck(1000.0, None)
        self.fill_mode = "filled"
        self.placed: list = []
        self.cancelled: list[int] = []
        self.orders: dict[int, OrderStatus] = {}
        self.extra_open: list[TrackedOpenOrder] = []
        self.snapshot_calls = 0
        self.fail_positions: Exception | None = None
        self._next = 1

    def positions(self) -> list[BrokerPosition]:
        if self.fail_positions is not None:
            raise self.fail_positions
        return list(self.positions_list)

    def daily_pnl(self) -> float | None:
        return self.pnl

    def available_funds(self) -> float | None:
        return self.funds

    def entry_snapshot(self, as_of) -> EntrySnapshot:
        self.snapshot_calls += 1
        if self.snapshot_error is not None:
            raise self.snapshot_error
        if self.snapshot is None:
            raise SelectionError("no snapshot")
        for quote in self.snapshot.options:
            self.quotes[quote.con_id] = quote
        return self.snapshot

    def quote_legs(self, con_ids: list[int]) -> list[OptionQuote]:
        return [self.quotes[con_id] for con_id in con_ids if con_id in self.quotes]

    def list_futures(self, symbol: str) -> list:
        return [candidate for candidate in self.futures if candidate.symbol == symbol]

    def future_quote(self, con_id: int) -> FutureQuote:
        try:
            return self.future_quotes[con_id]
        except KeyError:
            raise HaltError("future quote is not two-sided") from None

    def what_if(self, intent, *, halted: bool, halt_file, env) -> MarginCheck:
        assert_can_submit(self.settings, env, halted, halt_file)
        return self.margin

    def place_order(self, intent, *, halted: bool, halt_file, env) -> int:
        assert_can_submit(self.settings, env, halted, halt_file)
        order_id = self._next
        self._next += 1
        self.placed.append(intent)
        if self.fill_mode == "rejected":
            self.orders[order_id] = OrderStatus(order_id, "rejected", 0, intent.quantity, None)
            return order_id
        if self.fill_mode == "submitted":
            self.orders[order_id] = OrderStatus(order_id, "submitted", 0, intent.quantity, None)
            return order_id
        self.orders[order_id] = OrderStatus(order_id, "filled", intent.quantity, 0, intent.limit_price)
        if self.fill_mode == "filled":
            self._apply_positions(intent)
        return order_id

    def cancel_order(self, broker_order_id: int) -> None:
        self.cancelled.append(broker_order_id)
        current = self.orders[broker_order_id]
        self.orders[broker_order_id] = OrderStatus(
            broker_order_id,
            "cancelled",
            current.filled,
            current.remaining,
            current.avg_price,
        )

    def order_status(self, broker_order_id: int) -> OrderStatus:
        try:
            return self.orders[broker_order_id]
        except KeyError:
            raise HaltError(f"order {broker_order_id} is not known to the broker session") from None

    def open_tracked_orders(self) -> list[TrackedOpenOrder]:
        live = [
            TrackedOpenOrder(order_id, "SPX", "BAG")
            for order_id, status in self.orders.items()
            if status.status == "submitted"
        ]
        return live + list(self.extra_open)

    def connect(self) -> None:
        return None

    def disconnect(self) -> None:
        return None

    def _apply_positions(self, intent) -> None:
        if isinstance(intent, ComboIntent):
            for leg in intent.legs:
                signed = intent.quantity if leg.action == "BUY" else -intent.quantity
                self._net(leg.con_id, "SPX", "OPT", signed)
            return
        if isinstance(intent, HedgeIntent):
            signed = intent.quantity if intent.action == "BUY" else -intent.quantity
            self._net(intent.con_id, intent.symbol, "FUT", signed)

    def _net(self, con_id: int, symbol: str, sec_type: str, quantity: float) -> None:
        for index, position in enumerate(self.positions_list):
            if position.con_id != con_id:
                continue
            updated = position.quantity + quantity
            if updated == 0:
                self.positions_list.pop(index)
            else:
                self.positions_list[index] = BrokerPosition(con_id, symbol, sec_type, updated, position.trading_class)
            return
        if quantity != 0:
            self.positions_list.append(BrokerPosition(con_id, symbol, sec_type, quantity))
