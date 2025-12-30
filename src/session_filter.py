from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
import os
from typing import Iterable, List, Optional


AWST = timezone(timedelta(hours=8))


@dataclass(frozen=True)
class SessionSnapshot:
    name: str
    start_awst: datetime
    end_awst: datetime
    start_utc: datetime
    end_utc: datetime

    @property
    def session_id(self) -> str:
        start_date = self.start_awst.date().isoformat()
        start_ts = self.start_awst.strftime("%H%M")
        return f"{self.name}-{start_date}-{start_ts}"


def _parse_awst_time(value: Optional[str], fallback: time) -> time:
    if not value:
        return fallback
    parts = str(value).split(":")
    if len(parts) < 2:
        return fallback
    try:
        hour = int(parts[0])
        minute = int(parts[1])
        return time(hour, minute)
    except (TypeError, ValueError):
        return fallback


def _window_bounds(day: date, start: time, end: time) -> tuple[datetime, datetime]:
    start_dt = datetime.combine(day, start, tzinfo=AWST)
    end_dt = datetime.combine(day, end, tzinfo=AWST)
    if end_dt <= start_dt:
        end_dt += timedelta(days=1)
    return start_dt, end_dt


def _windows_from_env() -> List[tuple[str, time, time]]:
    london_start = _parse_awst_time(os.getenv("LONDON_SESSION_START_AWST"), time(15, 0))
    london_end = _parse_awst_time(os.getenv("LONDON_SESSION_END_AWST"), time(0, 0))
    ny_start = _parse_awst_time(os.getenv("NY_SESSION_START_AWST"), time(20, 0))
    ny_end = _parse_awst_time(os.getenv("NY_SESSION_END_AWST"), time(5, 0))
    return [
        ("london", london_start, london_end),
        ("newyork", ny_start, ny_end),
    ]


def _session_for_window(now_awst: datetime, *, name: str, start: time, end: time) -> SessionSnapshot | None:
    for day_offset in (0, -1):
        anchor_day = (now_awst.date() + timedelta(days=day_offset))
        start_dt, end_dt = _window_bounds(anchor_day, start, end)
        if start_dt <= now_awst < end_dt:
            return SessionSnapshot(
                name=name,
                start_awst=start_dt,
                end_awst=end_dt,
                start_utc=start_dt.astimezone(timezone.utc),
                end_utc=end_dt.astimezone(timezone.utc),
            )
    return None


def current_session(now_utc: datetime, *, mode: str | None = None) -> SessionSnapshot | None:
    awst_now = now_utc.astimezone(AWST)
    for name, start, end in _windows_from_env():
        session = _session_for_window(awst_now, name=name, start=start, end=end)
        if session:
            return session
    return None


_last_session: SessionSnapshot | None = None


def is_entry_session(now_utc: datetime, *, mode: str | None = None) -> bool:
    """Return True if new entries are allowed for the given UTC timestamp."""

    global _last_session
    session = current_session(now_utc, mode=mode)
    if session and (_last_session is None or session.session_id != _last_session.session_id):
        print(
            f"[SESSION] Start {session.name} awst={session.start_awst.strftime('%H:%M')}..{session.end_awst.strftime('%H:%M')}",
            flush=True,
        )
    if _last_session and (session is None or session.session_id != _last_session.session_id):
        print(
            f"[SESSION] End {_last_session.name} awst={_last_session.end_awst.strftime('%H:%M')}",
            flush=True,
        )
    _last_session = session
    return session is not None


__all__ = ["is_entry_session", "current_session", "SessionSnapshot", "AWST"]
