from __future__ import annotations

from datetime import datetime, timezone

from src.scheduler_policy import RuntimeSchedule, install_runtime_jobs


class FakeScheduler:
    def __init__(self):
        self.calls = []

    def add_job(self, fn, trigger, **kwargs):
        self.calls.append((fn, trigger, kwargs))


async def heartbeat():
    return None


async def decision():
    return None


def test_render_defaults_are_respected_and_jobs_are_staggered(monkeypatch):
    monkeypatch.setenv("HEARTBEAT_SECONDS", "30")
    monkeypatch.setenv("DECISION_SECONDS", "60")
    monkeypatch.delenv("SCHEDULER_MISFIRE_GRACE_SECONDS", raising=False)
    now = datetime(2026, 9, 28, 3, 0, tzinfo=timezone.utc)
    scheduler = FakeScheduler()

    policy = install_runtime_jobs(
        scheduler, heartbeat_job=heartbeat, decision_job=decision, now_utc=now
    )

    assert policy == RuntimeSchedule(
        decision_seconds=60,
        heartbeat_seconds=30,
        misfire_grace_seconds=10,
        heartbeat_phase_seconds=15,
    )
    heartbeat_call, decision_call = scheduler.calls
    assert heartbeat_call[2]["next_run_time"].second == 15
    assert decision_call[2]["next_run_time"].minute == 1
    assert decision_call[2]["next_run_time"].second == 0


def test_jobs_coalesce_never_overlap_and_tolerate_small_delay(monkeypatch):
    monkeypatch.setenv("HEARTBEAT_SECONDS", "30")
    monkeypatch.setenv("DECISION_SECONDS", "60")
    monkeypatch.setenv("SCHEDULER_MISFIRE_GRACE_SECONDS", "12")
    scheduler = FakeScheduler()

    install_runtime_jobs(
        scheduler,
        heartbeat_job=heartbeat,
        decision_job=decision,
        now_utc=datetime(2026, 9, 28, tzinfo=timezone.utc),
    )

    for _, trigger, kwargs in scheduler.calls:
        assert trigger == "interval"
        assert kwargs["coalesce"] is True
        assert kwargs["max_instances"] == 1
        assert kwargs["misfire_grace_time"] == 12
        assert kwargs["replace_existing"] is True


def test_bad_environment_values_are_bounded(monkeypatch):
    monkeypatch.setenv("HEARTBEAT_SECONDS", "1")
    monkeypatch.setenv("DECISION_SECONDS", "9999")
    monkeypatch.setenv("SCHEDULER_MISFIRE_GRACE_SECONDS", "999")
    policy = RuntimeSchedule.from_env()

    assert policy.heartbeat_seconds == 15
    assert policy.decision_seconds == 300
    assert policy.misfire_grace_seconds == 20
    assert 1 <= policy.heartbeat_phase_seconds <= policy.heartbeat_seconds


def test_invalid_environment_values_fall_back(monkeypatch):
    monkeypatch.setenv("HEARTBEAT_SECONDS", "banana")
    monkeypatch.setenv("DECISION_SECONDS", "banana")
    monkeypatch.setenv("SCHEDULER_MISFIRE_GRACE_SECONDS", "banana")
    policy = RuntimeSchedule.from_env()

    assert policy.heartbeat_seconds == 30
    assert policy.decision_seconds == 60
    assert policy.misfire_grace_seconds == 10
    assert policy.heartbeat_phase_seconds == 15
