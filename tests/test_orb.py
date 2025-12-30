from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import orb
from src.session_filter import SessionSnapshot, AWST


def _session(start_awst: datetime) -> SessionSnapshot:
    end_awst = start_awst + timedelta(hours=1)
    return SessionSnapshot(
        name="london",
        start_awst=start_awst,
        end_awst=end_awst,
        start_utc=start_awst.astimezone(timezone.utc),
        end_utc=end_awst.astimezone(timezone.utc),
    )


def test_opening_range_updates_and_finalizes():
    start_awst = datetime(2024, 1, 1, 15, 0, tzinfo=AWST)
    session = _session(start_awst)
    now = session.start_utc + timedelta(minutes=5)
    rng = orb.update_opening_range("EUR_USD", session, candle_high=1.1, candle_low=1.0, now_utc=now)
    assert rng.high == 1.1
    assert rng.low == 1.0
    assert rng.finalized is False

    later = session.start_utc + timedelta(minutes=16)
    rng = orb.update_opening_range("EUR_USD", session, candle_high=1.05, candle_low=0.99, now_utc=later)
    assert rng.finalized is True
    assert rng.high == 1.1
    assert rng.low == 1.0


def test_breakout_direction_buy_and_sell():
    start_awst = datetime(2024, 1, 1, 15, 0, tzinfo=AWST)
    session = _session(start_awst)
    now = session.start_utc + timedelta(minutes=16)
    rng = orb.update_opening_range("GBP_USD", session, candle_high=1.2, candle_low=1.1, now_utc=now)
    rng.finalized = True

    assert orb.breakout_direction(1.2001, rng) == "BUY"
    assert orb.breakout_direction(1.0999, rng) == "SELL"
    assert orb.breakout_direction(1.15, rng) is None
