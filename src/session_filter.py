from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
import os
from typing import Iterable, List, Optional


AWST = timezone(timedelta(hours=8))
STRICT = "STRICT"
EXTENDED = "EXTENDED"
ALWAYS = "ALWAYS"
SOFT = "SOFT"


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


@dataclass(frozen=True)
class SessionDecision:
    allowed: bool
    in_session: bool
    session: SessionSnapshot | None
    mode: str
    risk_scale: float = 1.0
    reason: str | None = None

    @property
    def off_session(self) -> bool:
        return not self.in_session


def _mode_from_env(mode: str | None) -> str:
    env_mode = os.getenv("SESSION_MODE")
    label = (env_mode or mode or STRICT).strip().upper()
    if label in {STRICT, EXTENDED, ALWAYS, SOFT}:
        return label
    return STRICT


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


def _near_session_window(now_utc: datetime, *, buffer_minutes: float) -> bool:
    """Return True when now_utc is within the buffer around any configured session start/end."""

    awst_now = now_utc.astimezone(AWST)
    buffer = timedelta(minutes=max(buffer_minutes, 0.0))
    for name, start, end in _windows_from_env():
        for day_offset in (0, -1, 1):
            anchor_day = (awst_now.date() + timedelta(days=day_offset))
            start_dt, end_dt = _window_bounds(anchor_day, start, end)
            if abs(start_dt - awst_now) <= buffer or abs(end_dt - awst_now) <= buffer:
                return True
    return False


_last_session: SessionSnapshot | None = None


def _low_volatility(atr: Optional[float], atr_baseline: Optional[float], *, max_ratio: float) -> bool:
    try:
        if atr is None or atr_baseline is None:
            return False
        if atr_baseline <= 0:
            return False
        return float(atr) <= float(atr_baseline) * max_ratio
    except (TypeError, ValueError):
        return False


def session_decision(
    now_utc: datetime,
    *,
    mode: str | None = None,
    atr: Optional[float] = None,
    atr_baseline: Optional[float] = None,
    trend_aligned: Optional[bool] = None,
    max_off_session_vol_ratio: float = 1.25,
    off_session_risk_scale: float = 0.5,
) -> SessionDecision:
    """Compute session gating with flexible modes."""

    global _last_session
    session = current_session(now_utc, mode=mode)
    normalized_mode = _mode_from_env(mode)
    in_session = session is not None
    risk_scale = 1.0
    allowed = in_session
    reason: str | None = None

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

    if normalized_mode == ALWAYS:
        allowed = True
        if not in_session:
            risk_scale = max(0.1, float(off_session_risk_scale))
            reason = "off-session-risk-reduced"
        return SessionDecision(
            allowed=allowed,
            in_session=in_session,
            session=session,
            mode=normalized_mode,
            risk_scale=risk_scale,
            reason=reason,
        )

    if normalized_mode in {EXTENDED, SOFT}:
        if in_session:
            allowed = True
        else:
            low_vol = _low_volatility(atr, atr_baseline, max_ratio=max_off_session_vol_ratio)
            trend_ok = bool(trend_aligned)
            allowed = low_vol and trend_ok
            near_window = True
            if normalized_mode == SOFT:
                near_window = _near_session_window(
                    now_utc,
                    buffer_minutes=float(os.getenv("SESSION_SOFT_BUFFER_MINUTES", 60.0)),
                )
                allowed = allowed and near_window
            reason = "off-session-conditional" if not allowed else None
            if normalized_mode == SOFT and not near_window:
                reason = "soft-boundary-only"
        return SessionDecision(
            allowed=allowed,
            in_session=in_session,
            session=session,
            mode=normalized_mode,
            risk_scale=risk_scale,
            reason=reason,
        )

    # STRICT default
    return SessionDecision(
        allowed=allowed,
        in_session=in_session,
        session=session,
        mode=normalized_mode,
        risk_scale=risk_scale,
        reason="strict-off-session" if not allowed else None,
    )


def is_entry_session(now_utc: datetime, *, mode: str | None = None) -> bool:
    """Return True if new entries are allowed for the given UTC timestamp."""

    decision = session_decision(now_utc, mode=mode)
    return decision.allowed


__all__ = [
    "is_entry_session",
    "current_session",
    "SessionSnapshot",
    "SessionDecision",
    "AWST",
    "STRICT",
    "EXTENDED",
    "ALWAYS",
    "SOFT",
]
