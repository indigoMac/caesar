"""US regular-session calendar.

Weekdays are not treated as trading days. Session labels are naive calendar
dates. Instants (opens, closes, announcements) are timezone-aware UTC.
"""

from __future__ import annotations

from functools import lru_cache

import exchange_calendars as xcals
import pandas as pd

from rebalance.errors import DataContractError

_CALENDARS: dict[str, xcals.ExchangeCalendar] = {}


def get_exchange_calendar(name: str) -> xcals.ExchangeCalendar:
    calendar = _CALENDARS.get(name)
    if calendar is None:
        calendar = xcals.get_calendar(name)
        _CALENDARS[name] = calendar
    return calendar


def as_utc(timestamp: pd.Timestamp | str) -> pd.Timestamp:
    """Convert an aware timestamp to UTC. Naive timestamps are rejected."""
    parsed = pd.Timestamp(timestamp)
    if parsed.tzinfo is None:
        raise DataContractError(
            f"timestamp {timestamp!r} is naive; supply a timezone offset "
            "or localize it before calling the research code"
        )
    return parsed.tz_convert("UTC")


def session_label(day: pd.Timestamp | str) -> pd.Timestamp:
    """Return a naive midnight session label. Aware instants are rejected."""
    parsed = pd.Timestamp(day)
    if parsed.tzinfo is not None:
        raise DataContractError(
            f"{day!r} is timezone-aware. Pass a calendar date for a session label, "
            "not an announcement instant."
        )
    normalized = parsed.normalize()
    if parsed != normalized:
        raise DataContractError(f"{day!r} has a time component; session dates must be dates")
    # Exchange session indexes are datetime64[ns]. Pandas 3 defaults to
    # microseconds, which would miss exact session lookups.
    if hasattr(normalized, "as_unit"):
        normalized = normalized.as_unit("ns")
    return normalized


class TradingCalendar:
    """Regular-session calendar for a named exchange, defaulting to NYSE."""

    def __init__(self, name: str = "XNYS", timezone: str = "America/New_York") -> None:
        self.name = name
        self.timezone = timezone
        self._calendar = get_exchange_calendar(name)

    def is_session(self, day: pd.Timestamp | str) -> bool:
        return bool(self._calendar.is_session(session_label(day)))

    def require_session(self, day: pd.Timestamp | str) -> pd.Timestamp:
        label = session_label(day)
        if not self._calendar.is_session(label):
            raise DataContractError(f"{label.date()} is not a regular {self.name} session")
        return label

    def session_open(self, day: pd.Timestamp | str) -> pd.Timestamp:
        label = self.require_session(day)
        return pd.Timestamp(self._calendar.session_open(label)).tz_convert("UTC")

    def session_close(self, day: pd.Timestamp | str) -> pd.Timestamp:
        label = self.require_session(day)
        return pd.Timestamp(self._calendar.session_close(label)).tz_convert("UTC")

    def shift_session(self, day: pd.Timestamp | str, offset: int) -> pd.Timestamp:
        label = self.require_session(day)
        if offset == 0:
            return label
        location = int(self._calendar.sessions.get_loc(label))
        target = location + offset
        sessions = self._calendar.sessions
        if target < 0 or target >= len(sessions):
            raise DataContractError(
                f"session offset {offset} from {label.date()} is outside "
                f"{self.name} ({pd.Timestamp(sessions[0]).date()} .. "
                f"{pd.Timestamp(sessions[-1]).date()})"
            )
        return session_label(pd.Timestamp(sessions[target]))

    def session_distance(self, origin: pd.Timestamp | str, target: pd.Timestamp | str) -> int:
        start = self.require_session(origin)
        end = self.require_session(target)
        sessions = self._calendar.sessions
        return int(sessions.get_loc(end) - sessions.get_loc(start))

    def sessions_between(self, start: pd.Timestamp | str, end: pd.Timestamp | str) -> pd.DatetimeIndex:
        start_label = self.require_session(start)
        end_label = self.require_session(end)
        if end_label < start_label:
            raise DataContractError(
                f"session range ends on {end_label.date()} before it starts on {start_label.date()}"
            )
        index = self._calendar.sessions_in_range(start_label, end_label)
        return pd.DatetimeIndex(session_label(pd.Timestamp(value)) for value in index)

    def earliest_regular_open_after(self, signal_known_at: pd.Timestamp | str) -> pd.Timestamp:
        """First regular-session open strictly after the signal is public.

        An announcement during or after a session cannot execute at that
        session's open or close. The next regular open is the first legal fill.
        """
        known = as_utc(signal_known_at)
        local = known.tz_convert(self.timezone)
        search_start = (local - pd.Timedelta(days=10)).normalize().tz_localize(None)
        search_end = (local + pd.Timedelta(days=20)).normalize().tz_localize(None)
        # sessions_in_range requires session endpoints when the dates themselves
        # might fall on weekends, so widen to the surrounding sessions.
        first = self._calendar.date_to_session(search_start, direction="next")
        last = self._calendar.date_to_session(search_end, direction="previous")
        for session in self._calendar.sessions_in_range(first, last):
            open_utc = pd.Timestamp(self._calendar.session_open(session)).tz_convert("UTC")
            if open_utc > known:
                return open_utc
        raise DataContractError(f"no regular {self.name} open after {known.isoformat()}")

    def last_completed_session(self, asof: pd.Timestamp | str) -> pd.Timestamp:
        """Last session whose official close is at or before `asof`."""
        instant = as_utc(asof)
        local_date = instant.tz_convert(self.timezone).normalize().tz_localize(None)
        search_start = local_date - pd.Timedelta(days=14)
        first = self._calendar.date_to_session(search_start, direction="next")
        last = self._calendar.date_to_session(local_date, direction="previous")
        completed: list[pd.Timestamp] = []
        for session in self._calendar.sessions_in_range(first, last):
            close_utc = pd.Timestamp(self._calendar.session_close(session)).tz_convert("UTC")
            label = session_label(pd.Timestamp(session))
            if close_utc <= instant:
                completed.append(label)
        if not completed:
            raise DataContractError(f"no completed {self.name} session at or before {instant.isoformat()}")
        return completed[-1]

    def entry_session(self, earliest_trade_at: pd.Timestamp | str) -> pd.Timestamp:
        """Session label of an open instant, evaluated in the exchange timezone."""
        instant = as_utc(earliest_trade_at)
        local_date = instant.tz_convert(self.timezone).normalize().tz_localize(None)
        return self.require_session(local_date)


@lru_cache(maxsize=8)
def cached_calendar(name: str = "XNYS", timezone: str = "America/New_York") -> TradingCalendar:
    return TradingCalendar(name, timezone)
