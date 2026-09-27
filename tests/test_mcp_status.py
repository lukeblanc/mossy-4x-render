from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

import src.mcp_status as mcp_status
from src.mcp_status import (
    build_runtime_heartbeat,
    classify_verified_entry_age,
    publish_runtime_heartbeat,
)


@pytest.fixture
def anyio_backend():
    return "asyncio"


def test_runtime_heartbeat_is_sanitised_and_applies_supervisor_floor():
    heartbeat = build_runtime_heartbeat(
        service_status="running",
        mode="DEMO",
        oanda_environment="PRACTICE",
        scheduler_alive=True,
        last_cycle_age_sec=10.0,
        last_broker_sync_age_sec=20.0,
        open_trades_count=1,
        equity=1_328.80,
        revision="abc123",
        entry_window_state="weekend_locked",
        last_verified_entry_age_bucket="one_to_three_days",
        observed_at=datetime(2026, 9, 25, tzinfo=timezone.utc),
    )

    assert heartbeat["mode"] == "demo"
    assert heartbeat["oanda_environment"] == "practice"
    assert heartbeat["decision_cycle_fresh"] is True
    assert heartbeat["broker_sync_fresh"] is True
    assert heartbeat["has_open_trades"] is True
    assert heartbeat["supervisor_floor_breached"] is True
    assert heartbeat["entry_window_state"] == "weekend_locked"
    assert heartbeat["last_verified_entry_age_bucket"] == "one_to_three_days"
    assert "equity" not in heartbeat
    assert "account" not in heartbeat
    assert "last_verified_entry_at" not in heartbeat


def test_runtime_heartbeat_fails_closed_when_telemetry_is_unknown_or_stale():
    heartbeat = build_runtime_heartbeat(
        service_status="broker-unavailable",
        mode="demo",
        oanda_environment="practice",
        scheduler_alive=False,
        last_cycle_age_sec=None,
        last_broker_sync_age_sec=181.0,
        open_trades_count=None,
        equity=None,
        revision="abc123",
        entry_window_state="off_session",
        last_verified_entry_age_bucket="unknown",
    )

    assert heartbeat["decision_cycle_fresh"] is False
    assert heartbeat["broker_sync_fresh"] is False
    assert heartbeat["has_open_trades"] is None
    assert heartbeat["supervisor_floor_breached"] is True


@pytest.mark.parametrize("equity", [float("nan"), float("inf"), -float("inf"), "bad"])
def test_runtime_heartbeat_fails_closed_for_invalid_equity(equity):
    heartbeat = build_runtime_heartbeat(
        service_status="running",
        mode="demo",
        oanda_environment="practice",
        scheduler_alive=True,
        last_cycle_age_sec=10.0,
        last_broker_sync_age_sec=20.0,
        open_trades_count=0,
        equity=equity,
        revision="abc123",
        entry_window_state="in_configured_session",
        last_verified_entry_age_bucket="never",
    )

    assert heartbeat["supervisor_floor_breached"] is True


@pytest.mark.parametrize(
    ("latest_entry_at", "expected"),
    [
        (None, "never"),
        (datetime(2026, 9, 25, 11, 30, tzinfo=timezone.utc), "under_1h"),
        (datetime(2026, 9, 25, 2, 0, tzinfo=timezone.utc), "under_24h"),
        (datetime(2026, 9, 23, 12, 0, tzinfo=timezone.utc), "one_to_three_days"),
        (datetime(2026, 9, 21, 12, 0, tzinfo=timezone.utc), "over_three_days"),
        (datetime(2026, 9, 25, 12, 2, tzinfo=timezone.utc), "unknown"),
    ],
)
def test_verified_entry_timestamp_is_reduced_to_safe_age_bucket(
    latest_entry_at, expected
):
    observed_at = datetime(2026, 9, 25, 12, 0, tzinfo=timezone.utc)

    assert (
        classify_verified_entry_age(latest_entry_at, observed_at=observed_at)
        == expected
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("entry_window_state", "maybe-open"),
        ("last_verified_entry_age_bucket", "exactly-37-minutes"),
    ],
)
def test_runtime_heartbeat_rejects_non_enumerated_observability_values(field, value):
    kwargs = {
        "service_status": "running",
        "mode": "demo",
        "oanda_environment": "practice",
        "scheduler_alive": True,
        "last_cycle_age_sec": 10.0,
        "last_broker_sync_age_sec": 20.0,
        "open_trades_count": 0,
        "equity": 10_000.0,
        "revision": "abc123",
        "entry_window_state": "off_session",
        "last_verified_entry_age_bucket": "never",
    }
    kwargs[field] = value

    with pytest.raises(ValueError):
        build_runtime_heartbeat(**kwargs)


def test_observability_state_is_part_of_safety_fingerprint():
    baseline = {
        "observed_at": "first",
        "entry_window_state": "off_session",
        "last_verified_entry_age_bucket": "under_24h",
    }

    newer_timestamp = {**baseline, "observed_at": "newer"}
    changed_window = {**baseline, "entry_window_state": "weekend_locked"}
    changed_recency = {
        **baseline,
        "last_verified_entry_age_bucket": "one_to_three_days",
    }

    assert mcp_status._safety_fingerprint(newer_timestamp) == mcp_status._safety_fingerprint(
        baseline
    )
    assert mcp_status._safety_fingerprint(changed_window) != mcp_status._safety_fingerprint(
        baseline
    )
    assert mcp_status._safety_fingerprint(changed_recency) != mcp_status._safety_fingerprint(
        baseline
    )


@pytest.mark.anyio
async def test_publish_is_disabled_without_both_environment_values(monkeypatch):
    monkeypatch.delenv("MOSSY_MCP_STATUS_URL", raising=False)
    monkeypatch.delenv("MOSSY_MCP_STATUS_KEY", raising=False)

    sent, status = await publish_runtime_heartbeat({"safe": True})

    assert sent is False
    assert status == "disabled"


@pytest.mark.anyio
async def test_publish_blocks_plain_http_by_default(monkeypatch):
    monkeypatch.setenv(
        "MOSSY_MCP_STATUS_URL", "http://example.test/internal/runtime-heartbeat"
    )
    monkeypatch.setenv("MOSSY_MCP_STATUS_KEY", "test-secret")
    monkeypatch.delenv("MOSSY_MCP_ALLOW_INSECURE_HTTP", raising=False)

    sent, status = await publish_runtime_heartbeat({"safe": True})

    assert sent is False
    assert status == "insecure-url-blocked"


@pytest.mark.anyio
async def test_unchanged_active_publish_is_throttled_for_ten_minutes(monkeypatch):
    monkeypatch.setenv(
        "MOSSY_MCP_STATUS_URL", "https://example.test/internal/runtime-heartbeat"
    )
    monkeypatch.setenv("MOSSY_MCP_STATUS_KEY", "test-secret")
    payload = {"observed_at": "first", "safe": True}
    monkeypatch.setattr(mcp_status, "_last_success_monotonic", 100.0)
    monkeypatch.setattr(
        mcp_status, "_last_success_fingerprint", mcp_status._safety_fingerprint(payload)
    )
    monkeypatch.setattr(mcp_status.time, "monotonic", lambda: 699.0)

    sent, status = await publish_runtime_heartbeat(
        {"observed_at": "newer", "safe": True}, monitoring_active=True
    )

    assert sent is False
    assert status == "throttled"


@pytest.mark.anyio
async def test_safety_state_change_bypasses_publish_throttle(monkeypatch):
    monkeypatch.setenv(
        "MOSSY_MCP_STATUS_URL", "https://example.test/internal/runtime-heartbeat"
    )
    monkeypatch.setenv("MOSSY_MCP_STATUS_KEY", "test-secret")
    previous = {"observed_at": "first", "supervisor_floor_breached": False}
    monkeypatch.setattr(mcp_status, "_last_success_monotonic", 100.0)
    monkeypatch.setattr(
        mcp_status,
        "_last_success_fingerprint",
        mcp_status._safety_fingerprint(previous),
    )
    monkeypatch.setattr(mcp_status.time, "monotonic", lambda: 101.0)

    class Response:
        def raise_for_status(self):
            return None

    class Client:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, *args, **kwargs):
            return Response()

    monkeypatch.setattr(mcp_status.httpx, "AsyncClient", Client)

    sent, status = await publish_runtime_heartbeat(
        {"observed_at": "newer", "supervisor_floor_breached": True},
        monitoring_active=True,
    )

    assert sent is True
    assert status == "sent"
