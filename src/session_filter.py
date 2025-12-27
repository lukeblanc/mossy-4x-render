from __future__ import annotations

from datetime import datetime, time, timezone


_LONDON_START = time(7, 0)
_LONDON_END = time(16, 0)
_NY_EARLY_START = time(12, 0)
_NY_EARLY_END = time(17, 0)


def _is_in_window(current: time, start: time, end: time) -> bool:
    return start <= current < end


def is_entry_session(now_utc: datetime, *, mode: str | None = None) -> bool:
    """Return True if new entries are allowed for the given timestamp.

    The filter only applies in demo mode; for all other modes entries are always
    permitted.
    """

    if (mode or "").lower() != "demo":
        return True

    aware = now_utc.astimezone(timezone.utc)
    current_time = aware.time()

    in_london = _is_in_window(current_time, _LONDON_START, _LONDON_END)
    in_ny = _is_in_window(current_time, _NY_EARLY_START, _NY_EARLY_END)
    return in_london or in_ny


__all__ = ["is_entry_session"]
