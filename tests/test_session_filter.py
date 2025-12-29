from __future__ import annotations

from datetime import datetime, timezone

from src.session_filter import is_entry_session


def _utc(hour: int, minute: int = 0) -> datetime:
    return datetime(2024, 1, 1, hour, minute, tzinfo=timezone.utc)


def test_entry_allowed_london_session():
    assert is_entry_session(_utc(7, 30)) is True


def test_entry_allowed_overlap_session():
    assert is_entry_session(_utc(12, 30)) is True


def test_entry_blocked_outside_session():
    assert is_entry_session(_utc(22, 0)) is False
