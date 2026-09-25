from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


pytest.importorskip("mcp.server")

from mcp import Client  # noqa: E402
from starlette.testclient import TestClient  # noqa: E402


MCP_DIR = Path(__file__).resolve().parent


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def bridge(monkeypatch, tmp_path):
    monkeypatch.setenv("MOSSY_MCP_DATABASE_PATH", str(tmp_path / "runtime.db"))
    monkeypatch.setenv("MOSSY_MCP_STATUS_KEY", "unit-test-secret")
    monkeypatch.setenv("RENDER_EXTERNAL_HOSTNAME", "mcp-web.example.test")
    sys.path.insert(0, str(MCP_DIR))
    try:
        sys.modules.pop("bridge_state", None)
        spec = importlib.util.spec_from_file_location(
            "mossy_read_only_mcp_app", MCP_DIR / "app.py"
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.path.remove(str(MCP_DIR))
        sys.modules.pop("mossy_read_only_mcp_app", None)
        sys.modules.pop("bridge_state", None)


def _heartbeat() -> dict:
    return {
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "service_status": "running",
        "mode": "demo",
        "oanda_environment": "practice",
        "scheduler_alive": True,
        "decision_cycle_fresh": True,
        "broker_sync_fresh": True,
        "has_open_trades": False,
        "supervisor_floor_breached": True,
        "revision": "unit-test-revision",
    }


def test_internal_heartbeat_is_authenticated_and_old_write_routes_are_gone(bridge):
    with TestClient(bridge.app, base_url="http://localhost") as client:
        assert client.post("/internal/runtime-heartbeat", json=_heartbeat()).status_code == 401
        assert (
            client.post(
                "/internal/runtime-heartbeat",
                headers={"Authorization": "Bearer wrong"},
                json=_heartbeat(),
            ).status_code
            == 401
        )
        accepted = client.post(
            "/internal/runtime-heartbeat",
            headers={"Authorization": "Bearer unit-test-secret"},
            json=_heartbeat(),
        )
        assert accepted.status_code == 202
        assert client.post("/optimise").status_code == 404
        assert client.post("/ingest-logs").status_code == 404

        health = client.get("/health")
        assert health.status_code == 200
        assert health.json() == {
            "status": "ok",
            "read_only": True,
            "status_key_configured": True,
            "telemetry_received": True,
            "telemetry_fresh": True,
        }


@pytest.mark.anyio
async def test_mcp_lists_only_read_only_tools_and_reports_floor_block(bridge):
    bridge.save_runtime_heartbeat(_heartbeat())

    async with Client(bridge.mcp, raise_exceptions=True) as client:
        listed = await client.list_tools()
        assert {tool.name for tool in listed.tools} == {
            "get_runtime_health",
            "get_algo_guardrails",
            "get_latest_algo_report",
        }
        for tool in listed.tools:
            assert tool.annotations is not None
            assert tool.annotations.read_only_hint is True
            assert tool.annotations.destructive_hint is False

        result = await client.call_tool("get_runtime_health", {})
        assert result.is_error is not True
        assert result.structured_content["supervisor_status"] == "BLOCKED"
        assert any(
            "equity floor" in blocker
            for blocker in result.structured_content["blockers"]
        )


def test_broker_unavailable_status_is_always_blocked(bridge):
    heartbeat = _heartbeat()
    heartbeat["service_status"] = "broker-unavailable"
    heartbeat["supervisor_floor_breached"] = False
    bridge.save_runtime_heartbeat(heartbeat)

    result = bridge.get_runtime_health()

    assert result["supervisor_status"] == "BLOCKED"
    assert "Worker status is broker-unavailable." in result["blockers"]


def test_health_fails_when_internal_status_key_is_missing(bridge, monkeypatch):
    monkeypatch.delenv("MOSSY_MCP_STATUS_KEY")

    with TestClient(bridge.app, base_url="http://localhost") as client:
        response = client.get("/health")

    assert response.status_code == 503
    assert response.json()["status_key_configured"] is False


def test_old_observation_is_blocked_even_if_recently_received(bridge):
    heartbeat = _heartbeat()
    heartbeat["observed_at"] = (
        datetime.now(timezone.utc) - timedelta(minutes=16)
    ).isoformat()
    bridge.save_runtime_heartbeat(
        heartbeat,
        received_at=datetime.now(timezone.utc),
    )

    result = bridge.get_runtime_health()

    assert result["telemetry_fresh"] is False
    assert result["supervisor_status"] == "BLOCKED"
    assert "Worker telemetry is stale." in result["blockers"]


def test_out_of_order_heartbeat_cannot_replace_newer_state(bridge):
    newer = _heartbeat()
    newer["observed_at"] = datetime.now(timezone.utc).isoformat()
    newer["supervisor_floor_breached"] = True
    older = _heartbeat()
    older["observed_at"] = (
        datetime.now(timezone.utc) - timedelta(minutes=1)
    ).isoformat()
    older["supervisor_floor_breached"] = False

    assert bridge.save_runtime_heartbeat(newer) is True
    assert bridge.save_runtime_heartbeat(older) is False

    result = bridge.get_runtime_health()
    assert result["supervisor_status"] == "BLOCKED"
    assert any("equity floor" in blocker for blocker in result["blockers"])


def test_latest_algo_report_marks_old_report_stale(bridge):
    result = bridge.get_latest_algo_report()

    assert result["available"] is True
    assert result["stale"] is True
    assert result["freshness_warning"]
