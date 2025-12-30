from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

from src.session_filter import SessionSnapshot


@dataclass
class OpeningRange:
    high: float
    low: float
    start_utc: datetime
    end_utc: datetime
    finalized: bool

    def as_tuple(self) -> Tuple[float, float]:
        return self.high, self.low


_ranges: Dict[str, OpeningRange] = {}


def _range_key(instrument: str, session: SessionSnapshot) -> str:
    return f"{session.session_id}:{instrument.upper()}"


def reset_for_session(session: SessionSnapshot) -> None:
    keys_to_delete = [key for key in _ranges if not key.startswith(f"{session.session_id}:")]
    for key in keys_to_delete:
        _ranges.pop(key, None)


def update_opening_range(
    instrument: str,
    session: SessionSnapshot,
    *,
    candle_high: float,
    candle_low: float,
    now_utc: datetime,
    range_minutes: int = 15,
) -> OpeningRange:
    key = _range_key(instrument, session)
    existing = _ranges.get(key)
    range_end = session.start_utc + timedelta(minutes=range_minutes)

    if existing is None:
        existing = OpeningRange(
            high=candle_high,
            low=candle_low,
            start_utc=session.start_utc,
            end_utc=range_end,
            finalized=False,
        )
        _ranges[key] = existing

    if now_utc <= range_end:
        existing.high = max(existing.high, candle_high)
        existing.low = min(existing.low, candle_low)
    else:
        existing.finalized = True

    if now_utc >= range_end:
        existing.finalized = True

    return existing


def opening_range_for(instrument: str, session: SessionSnapshot) -> Optional[OpeningRange]:
    return _ranges.get(_range_key(instrument, session))


def breakout_direction(close_price: float, opening_range: OpeningRange) -> Optional[str]:
    if close_price > opening_range.high:
        return "BUY"
    if close_price < opening_range.low:
        return "SELL"
    return None


__all__ = ["OpeningRange", "opening_range_for", "update_opening_range", "reset_for_session", "breakout_direction"]
