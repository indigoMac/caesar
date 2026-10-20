"""SQLite audit of orders, fills, and the halt latch. Restart does not clear a halt."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from options_book.errors import HaltError
from options_book.types import BookState, OpenSpread

_TRANSITIONS = {
    "submitted": {"filled", "cancelled", "rejected", "cancel_requested"},
    "cancel_requested": {"filled", "cancelled", "rejected"},
}


@dataclass(frozen=True)
class StoredOrder:
    id: int
    ts: datetime
    role: str
    status: str
    broker_order_id: int | None
    payload: dict


class Store:
    def __init__(self, path: Path, hedge_symbol: str) -> None:
        if hedge_symbol not in {"MES", "ES"}:
            raise HaltError(f"unsupported hedge symbol: {hedge_symbol}")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._hedge_symbol = hedge_symbol
        self._conn = sqlite3.connect(path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._migrate()

    def close(self) -> None:
        self._conn.close()

    def is_halted(self) -> bool:
        row = self._conn.execute("SELECT halted FROM halt WHERE id = 1").fetchone()
        return bool(row["halted"])

    def halt_reason(self) -> str:
        row = self._conn.execute("SELECT reason FROM halt WHERE id = 1").fetchone()
        return str(row["reason"])

    def halt(self, reason: str, ts: datetime) -> None:
        with self._conn:
            row = self._conn.execute("SELECT halted FROM halt WHERE id = 1").fetchone()
            if row["halted"]:
                self._write_event("halt_repeat", {"reason": reason}, ts)
                return
            self._conn.execute(
                "UPDATE halt SET halted = 1, reason = ?, updated_at = ? WHERE id = 1",
                (reason, ts.isoformat()),
            )
            self._write_event("halt", {"reason": reason}, ts)

    def clear_halt(self, ts: datetime) -> None:
        with self._conn:
            if not self.is_halted():
                return
            self._conn.execute(
                "UPDATE halt SET halted = 0, reason = '', updated_at = ? WHERE id = 1",
                (ts.isoformat(),),
            )
            self._write_event("halt_cleared", {}, ts)

    def record_event(self, kind: str, payload: dict, ts: datetime) -> None:
        with self._conn:
            self._write_event(kind, payload, ts)

    def events(self) -> list[tuple[str, dict]]:
        rows = self._conn.execute("SELECT kind, payload FROM events ORDER BY id").fetchall()
        return [(str(row["kind"]), json.loads(row["payload"])) for row in rows]

    def book(self) -> BookState:
        row = self._conn.execute("SELECT * FROM book WHERE id = 1").fetchone()
        return BookState(
            spread=_load_spread(row["spread_json"]),
            hedge_qty=int(row["hedge_qty"]),
            hedge_con_id=None if row["hedge_con_id"] is None else int(row["hedge_con_id"]),
            hedge_symbol=str(row["hedge_symbol"]),
        )

    def save_book(self, book: BookState, ts: datetime) -> None:
        if book.hedge_symbol != self._hedge_symbol:
            raise HaltError("stored hedge symbol does not match config")
        with self._conn:
            self._conn.execute(
                """
                UPDATE book
                SET spread_json = ?, hedge_qty = ?, hedge_con_id = ?, hedge_symbol = ?
                WHERE id = 1
                """,
                (
                    _dump_spread(book.spread),
                    book.hedge_qty,
                    book.hedge_con_id,
                    book.hedge_symbol,
                ),
            )
            self._write_event("book", {"hedge_qty": book.hedge_qty, "open": book.spread is not None}, ts)

    def insert_order(
        self,
        *,
        ts: datetime,
        role: str,
        status: str,
        broker_order_id: int | None,
        payload: dict,
    ) -> int:
        if role not in {"entry", "close", "hedge"}:
            raise HaltError("unknown order role")
        if status not in {"suppressed", "submitted"}:
            raise HaltError("new orders start as suppressed or submitted")
        if status == "submitted" and broker_order_id is None:
            raise HaltError("submitted order is missing a broker id")
        with self._conn:
            cursor = self._conn.execute(
                """
                INSERT INTO orders (ts, role, status, broker_order_id, payload)
                VALUES (?, ?, ?, ?, ?)
                """,
                (ts.isoformat(), role, status, broker_order_id, _dump(payload)),
            )
            return int(cursor.lastrowid)

    def get_order(self, order_id: int) -> StoredOrder:
        row = self._conn.execute("SELECT * FROM orders WHERE id = ?", (order_id,)).fetchone()
        if row is None:
            raise HaltError(f"order {order_id} is not in the audit log")
        return _stored(row)

    def orders(self) -> list[StoredOrder]:
        rows = self._conn.execute("SELECT * FROM orders ORDER BY id").fetchall()
        return [_stored(row) for row in rows]

    def working_orders(self) -> list[StoredOrder]:
        rows = self._conn.execute(
            """
            SELECT * FROM orders
            WHERE status IN ('submitted', 'cancel_requested')
            ORDER BY id
            """
        ).fetchall()
        return [_stored(row) for row in rows]

    def broker_order_ids(self) -> set[int]:
        rows = self._conn.execute(
            "SELECT broker_order_id FROM orders WHERE broker_order_id IS NOT NULL"
        ).fetchall()
        return {int(row["broker_order_id"]) for row in rows}

    def update_status(self, order_id: int, status: str, ts: datetime) -> None:
        order = self.get_order(order_id)
        allowed = _TRANSITIONS.get(order.status, set())
        if status not in allowed:
            raise HaltError(f"cannot move order {order_id} from {order.status} to {status}")
        with self._conn:
            self._conn.execute(
                "UPDATE orders SET status = ? WHERE id = ?",
                (status, order_id),
            )
            self._write_event("order_status", {"id": order_id, "status": status}, ts)

    def _migrate(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS halt (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                halted INTEGER NOT NULL,
                reason TEXT NOT NULL DEFAULT '',
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS book (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                spread_json TEXT,
                hedge_qty INTEGER NOT NULL,
                hedge_con_id INTEGER,
                hedge_symbol TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS orders (
                id INTEGER PRIMARY KEY,
                ts TEXT NOT NULL,
                role TEXT NOT NULL,
                status TEXT NOT NULL,
                broker_order_id INTEGER,
                payload TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY,
                ts TEXT NOT NULL,
                kind TEXT NOT NULL,
                payload TEXT NOT NULL
            );
            """
        )
        now = datetime.now().astimezone().isoformat()
        with self._conn:
            self._conn.execute(
                "INSERT OR IGNORE INTO halt (id, halted, reason, updated_at) VALUES (1, 0, '', ?)",
                (now,),
            )
            self._conn.execute(
                """
                INSERT OR IGNORE INTO book (id, spread_json, hedge_qty, hedge_con_id, hedge_symbol)
                VALUES (1, NULL, 0, NULL, ?)
                """,
                (self._hedge_symbol,),
            )
        row = self._conn.execute("SELECT hedge_symbol, spread_json, hedge_qty FROM book WHERE id = 1").fetchone()
        if row["hedge_symbol"] != self._hedge_symbol:
            if row["spread_json"] is not None or int(row["hedge_qty"]) != 0:
                self.halt(
                    "stored hedge symbol does not match config",
                    datetime.now().astimezone(),
                )
                raise HaltError("stored hedge symbol does not match config")
            with self._conn:
                self._conn.execute(
                    "UPDATE book SET hedge_symbol = ? WHERE id = 1",
                    (self._hedge_symbol,),
                )

    def _write_event(self, kind: str, payload: dict, ts: datetime) -> None:
        self._conn.execute(
            "INSERT INTO events (ts, kind, payload) VALUES (?, ?, ?)",
            (ts.isoformat(), kind, _dump(payload)),
        )


def _dump(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _dump_spread(spread: OpenSpread | None) -> str | None:
    if spread is None:
        return None
    return _dump(
        {
            "short_con_id": spread.short_con_id,
            "long_con_id": spread.long_con_id,
            "short_strike": spread.short_strike,
            "long_strike": spread.long_strike,
            "expiry": spread.expiry.isoformat(),
            "quantity": spread.quantity,
            "entry_credit_points": spread.entry_credit_points,
            "width_points": spread.width_points,
        }
    )


def _load_spread(raw: str | None) -> OpenSpread | None:
    if raw is None:
        return None
    payload = json.loads(raw)
    payload["expiry"] = date.fromisoformat(payload["expiry"])
    return OpenSpread(**payload)


def _stored(row: sqlite3.Row) -> StoredOrder:
    return StoredOrder(
        id=int(row["id"]),
        ts=datetime.fromisoformat(row["ts"]),
        role=str(row["role"]),
        status=str(row["status"]),
        broker_order_id=None if row["broker_order_id"] is None else int(row["broker_order_id"]),
        payload=json.loads(row["payload"]),
    )
