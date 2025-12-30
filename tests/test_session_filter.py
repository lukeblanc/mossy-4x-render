from __future__ import annotations

from datetime import datetime, timezone

import os

from src.session_filter import (
    AWST,
    SessionSnapshot,
    ALWAYS,
    EXTENDED,
    current_session,
    is_entry_session,
    session_decision,
)


def _utc(hour: int, minute: int = 0) -> datetime:
    return datetime(2024, 1, 1, hour, minute, tzinfo=timezone.utc)


def test_entry_allowed_london_session():
    assert is_entry_session(_utc(7, 30)) is True


def test_entry_allowed_overlap_session():
    assert is_entry_session(_utc(12, 30)) is True


def test_entry_blocked_outside_session():
    assert is_entry_session(_utc(22, 0)) is False


def test_session_snapshot_respects_env(monkeypatch):
    monkeypatch.setenv("LONDON_SESSION_START_AWST", "10:00")
    monkeypatch.setenv("LONDON_SESSION_END_AWST", "12:00")
    now = datetime(2024, 1, 1, 3, 30, tzinfo=timezone.utc)  # 11:30 AWST
    session = current_session(now)
    assert session is not None
    assert session.name == "london"
    assert session.start_awst.tzinfo == AWST


def test_session_snapshot_none_when_outside(monkeypatch):
    monkeypatch.delenv("LONDON_SESSION_START_AWST", raising=False)
    monkeypatch.delenv("LONDON_SESSION_END_AWST", raising=False)
    now = datetime(2024, 1, 1, 5, 10, tzinfo=timezone.utc)  # 13:10 AWST, outside defaults
    assert current_session(now) is None


def test_session_decision_extended_allows_when_trend_and_low_vol():
    decision = session_decision(
        _utc(22, 0),
        mode=EXTENDED,
        atr=1.0,
        atr_baseline=1.0,
        trend_aligned=True,
        max_off_session_vol_ratio=1.2,
    )
    assert decision.allowed is True
    assert decision.off_session is True


def test_session_decision_extended_blocks_on_high_vol():
    decision = session_decision(
        _utc(22, 0),
        mode=EXTENDED,
        atr=2.0,
        atr_baseline=1.0,
        trend_aligned=True,
        max_off_session_vol_ratio=1.2,
    )
    assert decision.allowed is False
    assert decision.off_session is True


def test_session_decision_always_reduces_risk_off_session():
    decision = session_decision(_utc(22, 0), mode=ALWAYS, atr=1.0, atr_baseline=1.0, trend_aligned=True)
    assert decision.allowed is True
    assert decision.risk_scale < 1.0
    assert decision.off_session is True
