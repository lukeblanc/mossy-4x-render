from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable


def _bounded_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    try:
        value = int(str(os.getenv(name, default)).strip())
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


@dataclass(frozen=True)
class RuntimeSchedule:
    decision_seconds: int
    heartbeat_seconds: int
    misfire_grace_seconds: int
    heartbeat_phase_seconds: int

    @classmethod
    def from_env(cls) -> "RuntimeSchedule":
        # The decision cadence remains deliberately bounded. This is a trading
        # safety/operability policy, not a throughput control.
        decision = _bounded_int(
            "DECISION_SECONDS", 60, minimum=30, maximum=300
        )
        heartbeat = _bounded_int(
            "HEARTBEAT_SECONDS", 30, minimum=15, maximum=300
        )
        # A short grace prevents harmless 1-2 second event-loop delays from
        # discarding a cycle, while still refusing very stale scheduled work.
        max_grace = max(2, min(20, decision // 2))
        grace = _bounded_int(
            "SCHEDULER_MISFIRE_GRACE_SECONDS",
            min(10, max_grace),
            minimum=2,
            maximum=max_grace,
        )
        # Phase the heartbeat away from the decision beat. With the Render
        # defaults this produces heartbeat at :15/:45 and decisions at :00.
        phase = max(1, min(heartbeat // 2, max(1, decision // 2)))
        return cls(
            decision_seconds=decision,
            heartbeat_seconds=heartbeat,
            misfire_grace_seconds=grace,
            heartbeat_phase_seconds=phase,
        )


def install_runtime_jobs(
    scheduler: Any,
    *,
    heartbeat_job: Callable[..., Any],
    decision_job: Callable[..., Any],
    now_utc: datetime | None = None,
) -> RuntimeSchedule:
    """Install non-overlapping, coalescing periodic jobs.

    max_instances=1 prevents duplicate execution. coalesce=True collapses any
    backlog to one invocation. A bounded misfire grace tolerates minor loop
    blocking without replaying stale trading decisions.
    """

    schedule = RuntimeSchedule.from_env()
    now = (now_utc or datetime.now(timezone.utc)).astimezone(timezone.utc)

    common = {
        "coalesce": True,
        "max_instances": 1,
        "misfire_grace_time": schedule.misfire_grace_seconds,
        "replace_existing": True,
    }
    scheduler.add_job(
        heartbeat_job,
        "interval",
        seconds=schedule.heartbeat_seconds,
        next_run_time=now + timedelta(seconds=schedule.heartbeat_phase_seconds),
        id="heartbeat",
        **common,
    )
    scheduler.add_job(
        decision_job,
        "interval",
        seconds=schedule.decision_seconds,
        next_run_time=now + timedelta(seconds=schedule.decision_seconds),
        id="decision-cycle",
        **common,
    )
    return schedule
