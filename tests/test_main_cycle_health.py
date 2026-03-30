from __future__ import annotations

from datetime import datetime, timedelta, timezone

import src.main as main_mod


def test_cycle_stale_detection_with_controlled_time():
    tracker = main_mod.CycleHealthTracker(gap_warn_seconds=90, summary_interval_seconds=900)
    t0 = datetime(2026, 3, 30, 0, 0, 0, tzinfo=timezone.utc)
    tracker.record_cycle_complete(0.42, now_utc=t0)

    assert tracker.is_cycle_stale(t0 + timedelta(seconds=89)) is False
    assert tracker.is_cycle_stale(t0 + timedelta(seconds=91)) is True


def test_cycle_percentiles_and_summary_interval_with_controlled_time():
    tracker = main_mod.CycleHealthTracker(gap_warn_seconds=90, summary_interval_seconds=900)
    t0 = datetime(2026, 3, 30, 0, 0, 0, tzinfo=timezone.utc)

    tracker.record_cycle_complete(0.20, now_utc=t0)
    tracker.record_cycle_complete(0.50, now_utc=t0 + timedelta(minutes=1))
    tracker.record_cycle_complete(1.50, now_utc=t0 + timedelta(minutes=2))

    p50, p95 = tracker.duration_percentiles()
    assert p50 == 0.50
    assert p95 > 1.0

    assert tracker.should_emit_summary(t0) is True
    assert tracker.should_emit_summary(t0 + timedelta(minutes=10)) is False
    assert tracker.should_emit_summary(t0 + timedelta(minutes=16)) is True
