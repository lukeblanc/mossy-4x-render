from __future__ import annotations

from datetime import datetime, timezone

from src.session_filter import is_entry_session


def _utc(hour: int, minute: int = 0) -> datetime:
    return datetime(2024, 1, 1, hour, minute, tzinfo=timezone.utc)


def test_entry_allowed_london_session_demo_mode():
    assert is_entry_session(_utc(7, 30), mode="demo") is True


def test_entry_allowed_overlap_session_demo_mode():
    assert is_entry_session(_utc(12, 30), mode="demo") is True


def test_entry_blocked_outside_session_demo_mode():
    assert is_entry_session(_utc(20, 0), mode="demo") is False


def test_session_filter_bypassed_outside_demo_mode():
    assert is_entry_session(_utc(3, 0), mode="paper") is True
    assert is_entry_session(_utc(3, 0), mode="live") is True
