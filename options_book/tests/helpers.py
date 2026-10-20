"""Chain and clock fixtures shared by the book tests."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from options_book.config import LoadedConfig, Settings
from options_book.store import Store
from options_book.types import EntrySnapshot, OptionQuote

NY = ZoneInfo("America/New_York")
AS_OF = date(2026, 10, 21)
EXPIRY = AS_OF + timedelta(days=40)
NOW = datetime(2026, 10, 21, 14, 0, tzinfo=NY)


class Weekdays:
    def is_session(self, day: date) -> bool:
        return day.weekday() < 5


class Closed:
    def is_session(self, day: date) -> bool:
        return False


def settings_with(base: Settings | None = None, **sections) -> Settings:
    current = base or Settings()
    updates = {}
    for name, values in sections.items():
        updates[name] = getattr(current, name).model_copy(update=values)
    return current.model_copy(update=updates)


def loaded(tmp_path, settings: Settings | None = None) -> LoadedConfig:
    config = tmp_path / "config" / "paper.yaml"
    config.parent.mkdir(parents=True, exist_ok=True)
    return LoadedConfig(settings or Settings(), config)


def store_for(config: LoadedConfig) -> Store:
    return Store(config.resolve(config.settings.store.path), config.settings.hedge.symbol)


def put_quote(
    con_id: int,
    strike: float,
    delta: float | None,
    bid: float | None,
    ask: float | None,
    quoted_at: datetime | None = NOW,
    *,
    vega: float | None = 0.2,
    expiry: date = EXPIRY,
    right: str = "P",
) -> OptionQuote:
    return OptionQuote(con_id, expiry, strike, right, delta, bid, ask, vega, quoted_at)


def tradable_chain(quoted_at: datetime = NOW, *, long_delta: float = -0.12) -> tuple[OptionQuote, ...]:
    return (
        put_quote(1, 5600, -0.05, 1.0, 1.2, quoted_at, vega=0.1),
        put_quote(2, 5625, long_delta, 4.0, 4.4, quoted_at, vega=0.25),
        put_quote(3, 5650, -0.16, 8.0, 8.4, quoted_at, vega=0.40),
        put_quote(4, 5700, -0.30, 12.0, 12.4, quoted_at, vega=0.5),
        put_quote(5, 5900, -0.55, 40.0, 40.4, quoted_at, vega=0.6),
    )


def snapshot(quoted_at: datetime = NOW, **kwargs) -> EntrySnapshot:
    return EntrySnapshot(spot=5800, options=tradable_chain(quoted_at, **kwargs))
