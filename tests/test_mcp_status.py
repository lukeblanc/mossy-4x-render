from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import httpx
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


def _halt_heartbeat(**kwargs):
    return build_runtime_heartbeat(
        service_status="running", mode="demo", oanda_environment="practice",
        scheduler_alive=True, last_cycle_age_sec=1.0,
        last_broker_sync_age_sec=1.0, open_trades_count=0, equity=10000.0,
        revision="test", entry_window_state="off_session",
        last_verified_entry_age_bucket="never", **kwargs,
    )


@pytest.mark.parametrize("reason", sorted(mcp_status.BROKER_ENTRY_HALT_REASONS))
def test_halt_reason_allowlist_is_preserved(reason):
    payload = _halt_heartbeat(broker_entry_halted=True, broker_entry_halt_reason=reason)
    assert payload["broker_entry_halted"] is True
    assert payload["broker_entry_halt_reason"] == reason


@pytest.mark.parametrize("reason", [None, "", "private account detail", {"secret": "value"}])
def test_arbitrary_halt_reason_is_reduced_to_other(reason):
    payload = _halt_heartbeat(broker_entry_halted=True, broker_entry_halt_reason=reason)
    assert payload["broker_entry_halt_reason"] == "other"
    assert "private account detail" not in json.dumps(payload)
    assert "secret" not in json.dumps(payload)


@pytest.mark.parametrize("halted", [None, False])
def test_missing_and_clear_halt_observations_remain_distinct(halted):
    payload = _halt_heartbeat(broker_entry_halted=halted)
    assert payload["broker_entry_halted"] is halted
    assert payload["broker_entry_halt_reason"] is None
    assert _halt_heartbeat()["broker_entry_halted"] is None


@pytest.mark.parametrize("halted", [0, 1, "false", "true"])
def test_halt_flag_requires_a_real_boolean(halted):
    with pytest.raises(ValueError, match="halt flag"):
        _halt_heartbeat(broker_entry_halted=halted)


@pytest.mark.parametrize("halted", [None, False])
def test_contradictory_halt_reason_is_rejected(halted):
    with pytest.raises(ValueError, match="confirmed halt"):
        _halt_heartbeat(broker_entry_halted=halted, broker_entry_halt_reason="other")


@pytest.fixture
def heartbeat_transport(monkeypatch):
    monkeypatch.setenv("MOSSY_MCP_STATUS_URL", "https://example.test/internal/runtime-heartbeat")
    monkeypatch.setenv("MOSSY_MCP_STATUS_KEY", "test-secret")
    monkeypatch.setattr(mcp_status, "_last_success_monotonic", None)
    monkeypatch.setattr(mcp_status, "_last_success_fingerprint", None)
    original_client = httpx.AsyncClient
    requests = []

    def install(handler):
        def record(request):
            requests.append(request)
            assert request.url == "https://example.test/internal/runtime-heartbeat"
            assert request.headers["Authorization"] == "Bearer test-secret"
            return handler(request)

        def client(**kwargs):
            assert kwargs["follow_redirects"] is False
            return original_client(transport=httpx.MockTransport(record), **kwargs)

        monkeypatch.setattr(mcp_status.httpx, "AsyncClient", client)
        return requests

    return install


def _capabilities():
    return {"optional_heartbeat_fields": sorted(mcp_status.BROKER_HALT_TELEMETRY_FIELDS)}


@pytest.mark.anyio
@pytest.mark.parametrize("legacy", [False, True])
async def test_publish_negotiates_new_and_legacy_bridges(heartbeat_transport, legacy):
    requests = heartbeat_transport(lambda request: (
        httpx.Response(405) if legacy else httpx.Response(200, json=_capabilities())
    ) if request.method == "GET" else httpx.Response(202))
    payload = _halt_heartbeat(broker_entry_halted=True, broker_entry_halt_reason="other")

    result = await publish_runtime_heartbeat(payload)

    assert result == (True, "sent:legacy-telemetry" if legacy else "sent")
    assert [request.method for request in requests] == ["GET", "POST"]
    expected = {key: value for key, value in payload.items()
                if not legacy or key not in mcp_status.BROKER_HALT_TELEMETRY_FIELDS}
    assert json.loads(requests[1].content) == expected
    assert payload["broker_entry_halted"] is True  # No in-place downgrade.


@pytest.mark.anyio
@pytest.mark.parametrize("status", [401, 403, 404, 422, 500, 302])
async def test_capability_http_failure_never_downgrades_or_posts(heartbeat_transport, status):
    requests = heartbeat_transport(lambda request: httpx.Response(status))
    assert await publish_runtime_heartbeat(_halt_heartbeat()) == (
        False, "http-error:HTTPStatusError",
    )
    assert len(requests) == 1
    assert mcp_status._last_success_monotonic is None


@pytest.mark.anyio
@pytest.mark.parametrize("capabilities,expected", [
    (None, "invalid-heartbeat-capabilities"),
    ({}, "invalid-heartbeat-capabilities"),
    ([], "invalid-heartbeat-capabilities"),
    ({"optional_heartbeat_fields": "broker_entry_halted"}, "invalid-heartbeat-capabilities"),
    ({"optional_heartbeat_fields": [{}]}, "invalid-heartbeat-capabilities"),
    ({"optional_heartbeat_fields": []}, "unsupported-heartbeat-capabilities"),
    ({"optional_heartbeat_fields": ["broker_entry_halted"]}, "unsupported-heartbeat-capabilities"),
])
async def test_invalid_capabilities_never_downgrade_or_post(
    heartbeat_transport, capabilities, expected
):
    requests = heartbeat_transport(lambda request: httpx.Response(200, json=capabilities))
    assert await publish_runtime_heartbeat(_halt_heartbeat()) == (False, expected)
    assert len(requests) == 1


@pytest.mark.anyio
async def test_invalid_capability_json_never_downgrades_or_posts(heartbeat_transport):
    requests = heartbeat_transport(lambda request: httpx.Response(200, text="not json"))
    assert await publish_runtime_heartbeat(_halt_heartbeat()) == (
        False, "invalid-heartbeat-capabilities",
    )
    assert len(requests) == 1


@pytest.mark.anyio
@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("status", [401, 422, 500])
async def test_post_failure_never_retries_a_downgrade(heartbeat_transport, legacy, status):
    requests = heartbeat_transport(lambda request: (
        httpx.Response(405) if legacy else httpx.Response(200, json=_capabilities())
    ) if request.method == "GET" else httpx.Response(status))
    assert await publish_runtime_heartbeat(_halt_heartbeat()) == (
        False, "http-error:HTTPStatusError",
    )
    assert len(requests) == 2
    assert mcp_status._last_success_monotonic is None


@pytest.mark.anyio
@pytest.mark.parametrize("failed_method", ["GET", "POST"])
async def test_transport_failure_never_downgrades(heartbeat_transport, failed_method):
    def handler(request):
        if request.method == failed_method:
            raise httpx.ConnectError("offline", request=request)
        return httpx.Response(200, json=_capabilities())

    requests = heartbeat_transport(handler)
    assert await publish_runtime_heartbeat(_halt_heartbeat()) == (
        False, "http-error:ConnectError",
    )
    assert len(requests) == (1 if failed_method == "GET" else 2)


@pytest.mark.anyio
async def test_throttle_precedes_negotiation_and_halt_change_bypasses_it(
    heartbeat_transport, monkeypatch
):
    requests = heartbeat_transport(lambda request: httpx.Response(
        200 if request.method == "GET" else 202, json=_capabilities()
    ))
    clear = _halt_heartbeat(broker_entry_halted=False)
    monkeypatch.setattr(mcp_status.time, "monotonic", lambda: 100.0)
    assert await publish_runtime_heartbeat(clear, monitoring_active=True) == (True, "sent")
    assert len(requests) == 2
    assert await publish_runtime_heartbeat(clear, monitoring_active=True) == (False, "throttled")
    assert len(requests) == 2
    halted = _halt_heartbeat(broker_entry_halted=True, broker_entry_halt_reason="other")
    assert await publish_runtime_heartbeat(halted, monitoring_active=True) == (True, "sent")
    assert len(requests) == 4


@pytest.mark.anyio
async def test_legacy_bridge_upgrade_is_renegotiated(heartbeat_transport, monkeypatch):
    legacy = True

    def handler(request):
        if request.method == "GET":
            return httpx.Response(405) if legacy else httpx.Response(200, json=_capabilities())
        return httpx.Response(202)

    requests = heartbeat_transport(handler)
    payload = _halt_heartbeat(broker_entry_halted=False)
    monkeypatch.setattr(mcp_status.time, "monotonic", lambda: 100.0)
    assert await publish_runtime_heartbeat(payload, monitoring_active=True) == (True, "sent:legacy-telemetry")
    legacy = False
    monkeypatch.setattr(mcp_status.time, "monotonic", lambda: 701.0)
    assert await publish_runtime_heartbeat(payload, monitoring_active=True) == (True, "sent")
    assert json.loads(requests[-1].content)["broker_entry_halted"] is False
