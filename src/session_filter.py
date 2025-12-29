from __future__ import annotations

from datetime import datetime, time, timezone


_LONDON_START = time(7, 0)
_LONDON_END = time(16, 0)
_NY_CORE_START = time(12, 0)
_NY_CORE_END = time(21, 0)


def _is_in_window(current: time, start: time, end: time) -> bool:
    return start <= current < end


def is_entry_session(now_utc: datetime, *, mode: str | None = None) -> bool:
    """Return True if new entries are allowed for the given UTC timestamp."""

    aware = now_utc.astimezone(timezone.utc)
    current_time = aware.time()

    in_london = _is_in_window(current_time, _LONDON_START, _LONDON_END)
    in_ny = _is_in_window(current_time, _NY_CORE_START, _NY_CORE_END)
    return in_london or in_ny


__all__ = ["is_entry_session"]
