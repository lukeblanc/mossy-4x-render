from __future__ import annotations

from datetime import datetime, timezone

import pytest

import src.mcp_status as mcp_status
from src.mcp_status import build_runtime_heartbeat, publish_runtime_heartbeat


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
        observed_at=datetime(2026, 9, 25, tzinfo=timezone.utc),
    )

    assert heartbeat["mode"] == "demo"
    assert heartbeat["oanda_environment"] == "practice"
    assert heartbeat["decision_cycle_fresh"] is True
    assert heartbeat["broker_sync_fresh"] is True
    assert heartbeat["has_open_trades"] is True
    assert heartbeat["supervisor_floor_breached"] is True
    assert "equity" not in heartbeat
    assert "account" not in heartbeat


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
    )

    assert heartbeat["supervisor_floor_breached"] is True


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
