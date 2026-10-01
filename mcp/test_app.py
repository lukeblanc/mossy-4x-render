from __future__ import annotations

import copy
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
        "entry_window_state": "off_session",
        "last_verified_entry_age_bucket": "under_24h",
        "revision": "unit-test-revision",
    }


def _metrics(*, trades: int = 10) -> dict:
    if trades == 0:
        return {
            "trades": 0,
            "win_rate": None,
            "closed_net_pnl_aud": None,
            "expectancy_aud": None,
            "profit_factor": None,
            "profit_factor_state": "no_sample",
            "max_drawdown_aud": None,
        }
    return {
        "trades": trades,
        "win_rate": 0.6,
        "closed_net_pnl_aud": 4.0,
        "expectancy_aud": 0.4,
        "profit_factor": 1.5,
        "profit_factor_state": "finite",
        "max_drawdown_aud": 2.0,
    }


def _learning_snapshot(*, observed_at: datetime | None = None) -> dict:
    observed = observed_at or datetime.now(timezone.utc)
    revision = "a" * 40
    return {
        "schema_version": "mossy.learning-review.v1",
        "observed_at_utc": observed.isoformat(),
        "source_revision": revision,
        "freshness": {"state": "fresh", "age_seconds": 0.0},
        "learning_status": {
            "overall": "active",
            "adaptive_tuner": {
                "state": "active",
                "mode": "reduce_only",
                "lookback_closed_trades": 80,
                "minimum_sample": 20,
                "sample_count": 40,
                "risk_multiplier": 0.75,
                "reason_code": "normal",
            },
            "setup_policy": {
                "state": "active",
                "mode": "reduce_or_block_only",
            },
            "shadow_learning": {
                "state": "active",
                "mode": "advisory_only",
                "evidence_scope": "executed_trade_subsets",
                "can_evaluate_more_trade_hypotheses": False,
                "cohort_start_utc": (observed - timedelta(days=100)).isoformat(),
                "clean_trades": 100,
                "train_trades": 70,
                "validation_trades": 30,
                "minimum_train": 50,
                "minimum_validation": 30,
                "minimum_coverage": 0.5,
                "auto_apply": False,
            },
            "observation_capture": {
                "state": "active",
                "mode": "best_effort",
                "submitted": 100,
                "persisted_events": 95,
                "duplicate_events": 3,
                "queue_drops": 0,
                "write_errors": 0,
                "pending_events": 2,
            },
        },
        "pair_performance": [
            {
                "instrument": "AUD_USD",
                "window": "last_40_broker_confirmed_closed",
                "sample_count": 4,
                "wins": 2,
                "losses": 1,
                "flat": 1,
                "win_rate": 0.5,
                "closed_net_pnl_aud": 2.0,
                "expectancy_aud": 0.5,
                "profit_factor": 2.0,
                "profit_factor_state": "finite",
                "max_drawdown_aud": 1.0,
                "evidence_status": "small_sample",
            },
            {
                "instrument": "GBP_USD",
                "window": "last_40_broker_confirmed_closed",
                "sample_count": 0,
                "wins": 0,
                "losses": 0,
                "flat": 0,
                "win_rate": None,
                "closed_net_pnl_aud": None,
                "expectancy_aud": None,
                "profit_factor": None,
                "profit_factor_state": "no_sample",
                "max_drawdown_aud": None,
                "evidence_status": "no_sample",
            },
        ],
        "challenger_review": {
            "registry_version": "mossy-shadow-registry.v1",
            "evidence_generated_at_utc": observed.isoformat(),
            "evidence_revision": revision,
            "baseline_validation": _metrics(),
            "candidates": [
                {
                    "candidate_id": "momentum_50",
                    "definition_version": "shadow-filter.v1",
                    "status": "rejected",
                    "train": _metrics(),
                    "validation": _metrics(),
                    "validation_coverage": 0.5,
                    "passed_gates": [
                        "sufficient_train_sample",
                        "sufficient_validation_sample",
                    ],
                    "failed_gates": ["validation_expectancy_gate_failed"],
                }
            ],
            "readiness": {
                "state": "no_candidate_passed",
                "leading_candidate_id": None,
                "reason_codes": ["validation_expectancy_gate_failed"],
                "next_allowed_action": "collect_more_evidence",
                "luke_approval_required": True,
                "auto_promotion_permitted": False,
            },
        },
        "governance": {
            "read_only": True,
            "demo_only": True,
            "can_trade": False,
            "can_write_configuration": False,
            "can_run_optimisation": False,
            "can_deploy": False,
            "can_promote": False,
        },
    }


def _mark_ready(snapshot: dict, *, prospective: bool, all_proofs: bool) -> None:
    if prospective:
        snapshot["learning_status"]["shadow_learning"]["evidence_scope"] = (
            "prospective_shadow_observations"
        )
        snapshot["learning_status"]["shadow_learning"][
            "can_evaluate_more_trade_hypotheses"
        ] = True
    candidate = snapshot["challenger_review"]["candidates"][0]
    snapshot["learning_status"]["shadow_learning"].update(
        {
            "clean_trades": 300,
            "train_trades": 200,
            "validation_trades": 100,
        }
    )
    snapshot["challenger_review"]["baseline_validation"] = {
        "trades": 100,
        "win_rate": 0.6,
        "closed_net_pnl_aud": 10.0,
        "expectancy_aud": 0.1,
        "profit_factor": 1.2,
        "profit_factor_state": "finite",
        "max_drawdown_aud": 10.0,
    }
    candidate["status"] = "ready_for_luke_review"
    candidate["train"] = {
        "trades": 150,
        "win_rate": 0.6,
        "closed_net_pnl_aud": 22.5,
        "expectancy_aud": 0.15,
        "profit_factor": 1.3,
        "profit_factor_state": "finite",
        "max_drawdown_aud": 8.0,
    }
    candidate["validation"] = {
        "trades": 100,
        "win_rate": 0.6,
        "closed_net_pnl_aud": 20.0,
        "expectancy_aud": 0.2,
        "profit_factor": 1.3,
        "profit_factor_state": "finite",
        "max_drawdown_aud": 8.0,
    }
    candidate["validation_coverage"] = 1.0
    candidate["failed_gates"] = []
    candidate["passed_gates"] = [
        "sufficient_train_sample",
        "sufficient_validation_sample",
        "sufficient_validation_coverage",
        "training_expectancy_positive",
        "training_profit_factor_at_least_one",
        "validation_expectancy_gate_passed",
        "validation_profit_factor_gate_passed",
        "validation_drawdown_gate_passed",
    ]
    if all_proofs:
        candidate["passed_gates"].extend(
            [
                "immutable_candidate_registered",
                "post_registration_evidence",
                "frozen_referee",
                "false_discovery_control_passed",
                "minimum_trade_count_passed",
                "minimum_pair_coverage_passed",
                "forward_shadow_passed",
            ]
        )
    snapshot["challenger_review"]["readiness"].update(
        {
            "state": "ready_for_luke_review",
            "leading_candidate_id": "momentum_50",
            "reason_codes": [],
            "next_allowed_action": "luke_review_only",
        }
    )


def _save_runtime_revision(bridge, revision: str) -> bool:
    heartbeat = _heartbeat()
    heartbeat["revision"] = revision
    return bridge.save_runtime_heartbeat(heartbeat)


def _save_learning_with_matching_runtime(bridge, snapshot: dict) -> bool:
    _save_runtime_revision(bridge, snapshot["source_revision"])
    return bridge.save_learning_snapshot(snapshot)


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
            "get_learning_review",
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
        assert result.structured_content["runtime"]["entry_window_state"] == "off_session"
        assert (
            result.structured_content["runtime"]["last_verified_entry_age_bucket"]
            == "under_24h"
        )
        assert any(
            "equity floor" in blocker
            for blocker in result.structured_content["blockers"]
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("entry_window_state", "unknown-window"),
        ("last_verified_entry_age_bucket", "37-minutes"),
        ("equity", 1_328.80),
    ],
)
def test_internal_heartbeat_strictly_rejects_unknown_or_sensitive_fields(
    bridge, field, value
):
    heartbeat = _heartbeat()
    heartbeat[field] = value

    with TestClient(bridge.app, base_url="http://localhost") as client:
        response = client.post(
            "/internal/runtime-heartbeat",
            headers={"Authorization": "Bearer unit-test-secret"},
            json=heartbeat,
        )

    assert response.status_code == 422


def test_broker_unavailable_status_is_always_blocked(bridge):
    heartbeat = _heartbeat()
    heartbeat["service_status"] = "broker-unavailable"
    heartbeat["supervisor_floor_breached"] = False
    bridge.save_runtime_heartbeat(heartbeat)

    result = bridge.get_runtime_health()

    assert result["supervisor_status"] == "BLOCKED"
    assert "Worker status is broker-unavailable." in result["blockers"]


def test_heartbeat_capability_route_is_authenticated_and_read_only(bridge, monkeypatch):
    with TestClient(bridge.app, base_url="http://localhost") as client:
        assert client.get("/internal/runtime-heartbeat").status_code == 401
        assert client.get(
            "/internal/runtime-heartbeat", headers={"Authorization": "Bearer wrong"},
        ).status_code == 401
        response = client.get(
            "/internal/runtime-heartbeat",
            headers={"Authorization": "Bearer unit-test-secret"},
        )
        assert response.status_code == 200
        assert response.json() == {"optional_heartbeat_fields": [
            "broker_entry_halted", "broker_entry_halt_reason",
        ]}
        assert bridge.load_runtime_heartbeat() is None
        monkeypatch.delenv("MOSSY_MCP_STATUS_KEY")
        assert client.get("/internal/runtime-heartbeat").status_code == 503


@pytest.mark.parametrize("halt_fields,expected_state,expected_blocker", [
    ({}, None, "Broker entry halt state is unknown"),
    ({"broker_entry_halted": None, "broker_entry_halt_reason": None}, None,
     "Broker entry halt state is unknown"),
    ({"broker_entry_halted": False, "broker_entry_halt_reason": None}, False, None),
    ({"broker_entry_halted": True, "broker_entry_halt_reason": "protective-stop-not-confirmed"},
     True, "Broker entries are halted: protective-stop-not-confirmed"),
])
def test_bridge_distinguishes_legacy_unknown_clear_and_halted_telemetry(
    bridge, halt_fields, expected_state, expected_blocker
):
    heartbeat = {**_heartbeat(), **halt_fields, "supervisor_floor_breached": False}
    with TestClient(bridge.app, base_url="http://localhost") as client:
        response = client.post(
            "/internal/runtime-heartbeat",
            headers={"Authorization": "Bearer unit-test-secret"}, json=heartbeat,
        )
        assert response.status_code == 202
    result = bridge.get_runtime_health()
    assert result["runtime"]["broker_entry_halted"] is expected_state
    if expected_blocker:
        assert result["supervisor_status"] == "BLOCKED"
        assert any(expected_blocker in blocker for blocker in result["blockers"])
    else:
        assert result["supervisor_status"] == "READY_FOR_REVIEW"
        assert result["blockers"] == []


@pytest.mark.parametrize("halt_fields", [
    {"broker_entry_halted": "false"},
    {"broker_entry_halted": "true", "broker_entry_halt_reason": "other"},
    {"broker_entry_halted": 0},
    {"broker_entry_halted": 1, "broker_entry_halt_reason": "other"},
    {"broker_entry_halted": False, "broker_entry_halt_reason": "other"},
    {"broker_entry_halted": None, "broker_entry_halt_reason": "other"},
    {"broker_entry_halt_reason": "other"},
    {"broker_entry_halted": True},
    {"broker_entry_halted": True, "broker_entry_halt_reason": "private account detail"},
    {"broker_entry_halted": True, "broker_entry_halt_reason": {"secret": "value"}},
])
def test_bridge_rejects_ambiguous_or_unbounded_halt_telemetry(bridge, halt_fields):
    with TestClient(bridge.app, base_url="http://localhost") as client:
        response = client.post(
            "/internal/runtime-heartbeat",
            headers={"Authorization": "Bearer unit-test-secret"},
            json={**_heartbeat(), **halt_fields},
        )
    assert response.status_code == 422
    assert bridge.load_runtime_heartbeat() is None


def test_worker_and_bridge_halt_reason_allowlists_match(bridge):
    # The bridge deliberately has no dependency on worker HTTP/trading packages.
    import ast

    source = (MCP_DIR.parent / "src" / "mcp_status.py").read_text(encoding="utf-8")
    assignment = next(
        node for node in ast.parse(source).body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "BROKER_ENTRY_HALT_REASONS"
                for target in node.targets)
    )
    reasons = ast.literal_eval(assignment.value.args[0])
    for reason in reasons:
        model = bridge.RuntimeHeartbeat.model_validate({
            **_heartbeat(), "broker_entry_halted": True,
            "broker_entry_halt_reason": reason,
        })
        assert model.broker_entry_halt_reason == reason


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


def test_internal_learning_snapshot_requires_authentication(bridge):
    snapshot = _learning_snapshot()
    with TestClient(bridge.app, base_url="http://localhost") as client:
        assert client.post("/internal/learning-snapshot", json=snapshot).status_code == 401
        assert (
            client.post(
                "/internal/learning-snapshot",
                headers={"Authorization": "Bearer wrong"},
                json=snapshot,
            ).status_code
            == 401
        )
        accepted = client.post(
            "/internal/learning-snapshot",
            headers={"Authorization": "Bearer unit-test-secret"},
            json=snapshot,
        )

    assert accepted.status_code == 202
    assert accepted.json() == {"accepted": True, "stored": True}


def test_learning_snapshot_accepts_low_win_rate_reduction_reason(bridge):
    snapshot = _learning_snapshot()
    snapshot["learning_status"]["adaptive_tuner"]["reason_code"] = "low_win_rate"

    with TestClient(bridge.app, base_url="http://localhost") as client:
        response = client.post(
            "/internal/learning-snapshot",
            headers={"Authorization": "Bearer unit-test-secret"},
            json=snapshot,
        )

    assert response.status_code == 202
    _save_runtime_revision(bridge, snapshot["source_revision"])
    result = bridge.get_learning_review()
    assert result["learning_status"]["adaptive_tuner"]["reason_code"] == (
        "low_win_rate"
    )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload.update({"equity": 5_000}),
        lambda payload: payload.update({"balance": 5_000}),
        lambda payload: payload.update({"account_id": "redacted"}),
        lambda payload: payload.update({"order_id": "redacted"}),
        lambda payload: payload.update({"token": "redacted"}),
        lambda payload: payload.update({"raw_trades": []}),
        lambda payload: payload.update({"unexpected": True}),
        lambda payload: payload["pair_performance"].pop(),
        lambda payload: payload["pair_performance"].__setitem__(
            1, copy.deepcopy(payload["pair_performance"][0])
        ),
        lambda payload: payload["challenger_review"].update(
            {"registry_version": "unreviewed-registry.v2"}
        ),
        lambda payload: payload["challenger_review"]["candidates"][0].update(
            {"definition_version": "mutable-definition.v2"}
        ),
        lambda payload: payload["pair_performance"][0].update({"price": 1.2345}),
        lambda payload: payload["challenger_review"]["candidates"][0].update(
            {"signal": "BUY"}
        ),
        lambda payload: payload["challenger_review"]["candidates"][0].update(
            {"candidate_id": "free_form_candidate"}
        ),
        lambda payload: payload["learning_status"]["shadow_learning"].update(
            {"auto_apply": True}
        ),
        lambda payload: payload["governance"].update({"can_trade": True}),
        lambda payload: payload["governance"].update(
            {"can_write_configuration": True}
        ),
        lambda payload: payload["governance"].update({"can_deploy": True}),
        lambda payload: payload["governance"].update({"can_promote": True}),
        lambda payload: payload["challenger_review"]["readiness"].update(
            {"next_allowed_action": "luke_review_only"}
        ),
        lambda payload: payload["learning_status"]["observation_capture"].update(
            {"submitted": True}
        ),
        lambda payload: payload["learning_status"]["observation_capture"].update(
            {"persisted_events": True}
        ),
        lambda payload: payload["learning_status"]["observation_capture"].update(
            {"duplicate_events": True}
        ),
        lambda payload: payload["learning_status"]["observation_capture"].update(
            {"queue_drops": True}
        ),
        lambda payload: payload["learning_status"]["observation_capture"].update(
            {"write_errors": True}
        ),
        lambda payload: payload["learning_status"]["observation_capture"].update(
            {"pending_events": True}
        ),
        lambda payload: payload["learning_status"]["observation_capture"].update(
            {"submitted": -1}
        ),
        lambda payload: payload["learning_status"]["observation_capture"].update(
            {"queue_drops": 1}
        ),
        lambda payload: payload["learning_status"]["observation_capture"].update(
            {"state": "degraded"}
        ),
        lambda payload: payload["learning_status"]["observation_capture"].update(
            {"raw_rows": []}
        ),
    ],
)
def test_learning_snapshot_rejects_sensitive_unknown_or_authority_fields(
    bridge, mutate
):
    snapshot = copy.deepcopy(_learning_snapshot())
    mutate(snapshot)

    with TestClient(bridge.app, base_url="http://localhost") as client:
        response = client.post(
            "/internal/learning-snapshot",
            headers={"Authorization": "Bearer unit-test-secret"},
            json=snapshot,
        )

    assert response.status_code == 422


def test_degraded_observation_capture_exposes_only_aggregate_counters(bridge):
    snapshot = _learning_snapshot()
    snapshot["learning_status"]["overall"] = "degraded"
    snapshot["learning_status"]["observation_capture"].update(
        {
            "state": "degraded",
            "queue_drops": 2,
            "write_errors": 1,
            "pending_events": None,
        }
    )
    with TestClient(bridge.app, base_url="http://localhost") as client:
        response = client.post(
            "/internal/learning-snapshot",
            headers={"Authorization": "Bearer unit-test-secret"},
            json=snapshot,
        )
    assert response.status_code == 202
    _save_runtime_revision(bridge, snapshot["source_revision"])

    result = bridge.get_learning_review()

    assert result["learning_status"]["overall"] == "degraded"
    assert result["learning_status"]["observation_capture"] == {
        "state": "degraded",
        "mode": "best_effort",
        "submitted": 100,
        "persisted_events": 95,
        "duplicate_events": 3,
        "queue_drops": 2,
        "write_errors": 1,
        "pending_events": None,
    }


def test_out_of_order_learning_snapshot_cannot_replace_newer_evidence(bridge):
    newer = _learning_snapshot()
    newer["source_revision"] = "b" * 40
    newer["challenger_review"]["evidence_revision"] = "b" * 40
    older = _learning_snapshot(
        observed_at=datetime.now(timezone.utc) - timedelta(minutes=1)
    )

    with TestClient(bridge.app, base_url="http://localhost") as client:
        first = client.post(
            "/internal/learning-snapshot",
            headers={"Authorization": "Bearer unit-test-secret"},
            json=newer,
        )
        replay = client.post(
            "/internal/learning-snapshot",
            headers={"Authorization": "Bearer unit-test-secret"},
            json=older,
        )

    assert first.json()["stored"] is True
    assert replay.json() == {"accepted": True, "stored": False}
    _save_runtime_revision(bridge, "b" * 40)
    assert bridge.get_learning_review()["source_revision"] == "b" * 40


def test_learning_snapshot_ordering_uses_utc_instants_not_timestamp_text(bridge):
    newer = _learning_snapshot(
        observed_at=datetime(2026, 9, 27, 10, 0, tzinfo=timezone.utc)
    )
    lexically_larger_but_older = _learning_snapshot(
        observed_at=datetime(
            2026,
            9,
            27,
            23,
            0,
            tzinfo=timezone(timedelta(hours=14)),
        )
    )

    assert bridge.save_learning_snapshot(newer) is True
    assert bridge.save_learning_snapshot(lexically_larger_but_older) is False
    stored = bridge.load_learning_snapshot()

    assert stored["observed_at_utc"] == "2026-09-27T10:00:00+00:00"
    assert stored["payload"]["observed_at_utc"] == "2026-09-27T10:00:00+00:00"


def test_learning_review_missing_cache_is_honestly_unavailable(bridge):
    result = bridge.get_learning_review()

    assert result["observed_at_utc"] == "1970-01-01T00:00:00+00:00"
    assert result["source_revision"] == "unknown"
    assert result["freshness"] == {"state": "unknown", "age_seconds": None}
    assert result["learning_status"]["overall"] == "degraded"
    assert result["learning_status"]["observation_capture"] == {
        "state": "degraded",
        "mode": "best_effort",
        "submitted": 0,
        "persisted_events": 0,
        "duplicate_events": 0,
        "queue_drops": 0,
        "write_errors": 0,
        "pending_events": None,
    }
    assert result["challenger_review"]["readiness"]["state"] == "unavailable"
    assert result["challenger_review"]["readiness"]["leading_candidate_id"] is None
    assert bridge.LearningReviewSnapshot.model_validate(result)
    assert result["governance"] == {
        "read_only": True,
        "demo_only": True,
        "can_trade": False,
        "can_write_configuration": False,
        "can_run_optimisation": False,
        "can_deploy": False,
        "can_promote": False,
    }


def test_stale_learning_evidence_cannot_remain_review_ready(bridge):
    snapshot = _learning_snapshot(
        observed_at=datetime.now(timezone.utc) - timedelta(hours=3)
    )
    snapshot["challenger_review"]["candidates"][0]["status"] = (
        "ready_for_luke_review"
    )
    snapshot["challenger_review"]["candidates"][0]["failed_gates"] = []
    snapshot["challenger_review"]["readiness"].update(
        {
            "state": "ready_for_luke_review",
            "leading_candidate_id": "momentum_50",
            "reason_codes": [],
            "next_allowed_action": "luke_review_only",
        }
    )
    assert _save_learning_with_matching_runtime(bridge, snapshot) is True

    result = bridge.get_learning_review()

    assert result["freshness"]["state"] == "stale"
    assert result["learning_status"]["overall"] == "degraded"
    assert result["challenger_review"]["readiness"]["state"] == "stale_evidence"
    assert result["challenger_review"]["readiness"]["leading_candidate_id"] is None
    assert result["challenger_review"]["readiness"]["reason_codes"] == [
        "evidence_stale"
    ]
    assert result["challenger_review"]["readiness"]["next_allowed_action"] == "none"
    assert result["challenger_review"]["candidates"][0]["status"] == "stale"


def test_old_challenger_evidence_cannot_be_rewrapped_as_fresh_ready(bridge):
    snapshot = _learning_snapshot()
    _mark_ready(snapshot, prospective=True, all_proofs=True)
    snapshot["challenger_review"]["evidence_generated_at_utc"] = (
        datetime.now(timezone.utc) - timedelta(hours=3)
    ).isoformat()
    assert _save_learning_with_matching_runtime(bridge, snapshot)

    result = bridge.get_learning_review()

    assert result["freshness"]["state"] == "stale"
    assert result["challenger_review"]["candidates"][0]["status"] == "stale"
    assert result["challenger_review"]["readiness"]["state"] == "stale_evidence"
    assert result["challenger_review"]["readiness"]["next_allowed_action"] == "none"


def test_executed_trade_subset_can_never_surface_as_ready(bridge):
    snapshot = _learning_snapshot()
    _mark_ready(snapshot, prospective=False, all_proofs=True)
    assert _save_learning_with_matching_runtime(bridge, snapshot)

    result = bridge.get_learning_review()
    candidate = result["challenger_review"]["candidates"][0]
    readiness = result["challenger_review"]["readiness"]

    assert candidate["status"] == "unavailable"
    assert "prospective_validation_required" in candidate["failed_gates"]
    assert "forward_shadow_required" in candidate["failed_gates"]
    assert "promotion_protocol_not_implemented" in candidate["failed_gates"]
    assert readiness["state"] == "unavailable"
    assert readiness["leading_candidate_id"] is None
    assert readiness["next_allowed_action"] == "none"


def test_prospective_candidate_needs_every_stronger_governance_proof(bridge):
    snapshot = _learning_snapshot()
    _mark_ready(snapshot, prospective=True, all_proofs=False)
    assert _save_learning_with_matching_runtime(bridge, snapshot)

    result = bridge.get_learning_review()
    candidate = result["challenger_review"]["candidates"][0]

    assert candidate["status"] == "unavailable"
    assert "immutable_registration_required" in candidate["failed_gates"]
    assert "post_registration_evidence_required" in candidate["failed_gates"]
    assert "frozen_referee_required" in candidate["failed_gates"]
    assert "false_discovery_control_required" in candidate["failed_gates"]
    assert "minimum_trade_count_required" in candidate["failed_gates"]
    assert "minimum_pair_coverage_required" in candidate["failed_gates"]
    assert "forward_shadow_required" in candidate["failed_gates"]
    assert "promotion_protocol_not_implemented" in candidate["failed_gates"]


def test_v1_candidate_with_all_proofs_is_forced_non_ready(bridge):
    snapshot = _learning_snapshot()
    _mark_ready(snapshot, prospective=True, all_proofs=True)
    assert _save_learning_with_matching_runtime(bridge, snapshot)

    result = bridge.get_learning_review()
    candidate = result["challenger_review"]["candidates"][0]
    readiness = result["challenger_review"]["readiness"]

    assert candidate["status"] == "unavailable"
    assert candidate["failed_gates"] == ["promotion_protocol_not_implemented"]
    assert readiness == {
        "state": "unavailable",
        "leading_candidate_id": None,
        "reason_codes": ["promotion_protocol_not_implemented"],
        "next_allowed_action": "none",
        "luke_approval_required": True,
        "auto_promotion_permitted": False,
    }


def test_ready_labels_cannot_override_failed_validation_metrics(bridge):
    snapshot = _learning_snapshot()
    _mark_ready(snapshot, prospective=True, all_proofs=True)
    validation = snapshot["challenger_review"]["candidates"][0]["validation"]
    validation["closed_net_pnl_aud"] = 5.0
    validation["expectancy_aud"] = 0.05
    assert _save_learning_with_matching_runtime(bridge, snapshot)

    result = bridge.get_learning_review()
    candidate = result["challenger_review"]["candidates"][0]
    readiness = result["challenger_review"]["readiness"]

    assert candidate["status"] == "unavailable"
    assert "validation_expectancy_gate_failed" in candidate["failed_gates"]
    assert "promotion_protocol_not_implemented" in candidate["failed_gates"]
    assert readiness["state"] == "unavailable"
    assert readiness["next_allowed_action"] == "none"


def test_ready_labels_cannot_override_minimum_trade_count(bridge):
    snapshot = _learning_snapshot()
    _mark_ready(snapshot, prospective=True, all_proofs=True)
    snapshot["learning_status"]["shadow_learning"].update(
        {"clean_trades": 100, "train_trades": 70, "validation_trades": 30}
    )
    snapshot["challenger_review"]["baseline_validation"].update(
        {
            "trades": 30,
            "closed_net_pnl_aud": 3.0,
            "expectancy_aud": 0.1,
        }
    )
    candidate = snapshot["challenger_review"]["candidates"][0]
    candidate["train"].update(
        {"trades": 60, "closed_net_pnl_aud": 9.0, "expectancy_aud": 0.15}
    )
    candidate["validation"].update(
        {"trades": 30, "closed_net_pnl_aud": 6.0, "expectancy_aud": 0.2}
    )
    candidate["validation_coverage"] = 1.0
    assert _save_learning_with_matching_runtime(bridge, snapshot)

    result = bridge.get_learning_review()
    candidate = result["challenger_review"]["candidates"][0]

    assert candidate["status"] == "unavailable"
    assert "minimum_trade_count_required" in candidate["failed_gates"]
    assert "promotion_protocol_not_implemented" in candidate["failed_gates"]
    assert result["challenger_review"]["readiness"]["state"] == "unavailable"


def test_candidate_ready_cannot_contradict_global_non_ready_state(bridge):
    snapshot = _learning_snapshot()
    _mark_ready(snapshot, prospective=True, all_proofs=True)
    snapshot["challenger_review"]["readiness"].update(
        {
            "state": "no_candidate_passed",
            "leading_candidate_id": None,
            "next_allowed_action": "none",
        }
    )
    assert _save_learning_with_matching_runtime(bridge, snapshot)

    result = bridge.get_learning_review()

    assert result["challenger_review"]["candidates"][0]["status"] == "unavailable"
    assert "promotion_protocol_not_implemented" in result["challenger_review"]["candidates"][0][
        "failed_gates"
    ]


def test_future_learning_snapshot_is_rejected(bridge):
    snapshot = _learning_snapshot(
        observed_at=datetime.now(timezone.utc) + timedelta(minutes=2)
    )

    with TestClient(bridge.app, base_url="http://localhost") as client:
        response = client.post(
            "/internal/learning-snapshot",
            headers={"Authorization": "Bearer unit-test-secret"},
            json=snapshot,
        )

    assert response.status_code == 422
    assert bridge.get_learning_review()["freshness"]["state"] == "unknown"


def test_future_cached_evidence_is_never_actionable(bridge):
    snapshot = _learning_snapshot(
        observed_at=datetime.now(timezone.utc) + timedelta(minutes=2)
    )
    _mark_ready(snapshot, prospective=True, all_proofs=True)
    assert _save_learning_with_matching_runtime(bridge, snapshot)

    result = bridge.get_learning_review()

    assert result["freshness"]["state"] == "future"
    assert result["learning_status"]["overall"] == "degraded"
    assert result["challenger_review"]["readiness"]["state"] == "unavailable"
    assert result["challenger_review"]["readiness"]["leading_candidate_id"] is None
    assert result["challenger_review"]["readiness"]["next_allowed_action"] == "none"
    assert result["challenger_review"]["candidates"][0]["status"] == "unavailable"


def test_revision_mismatch_is_unavailable_with_no_action(bridge):
    snapshot = _learning_snapshot()
    snapshot["challenger_review"]["evidence_revision"] = "b" * 40
    _save_runtime_revision(bridge, snapshot["source_revision"])
    assert bridge.save_learning_snapshot(snapshot)

    result = bridge.get_learning_review()
    readiness = result["challenger_review"]["readiness"]

    assert result["learning_status"]["overall"] == "degraded"
    assert readiness["state"] == "unavailable"
    assert readiness["reason_codes"] == ["revision_mismatch"]
    assert readiness["next_allowed_action"] == "none"


def test_missing_runtime_revision_fails_learning_provenance_closed(bridge):
    snapshot = _learning_snapshot()
    _mark_ready(snapshot, prospective=True, all_proofs=True)
    assert bridge.save_learning_snapshot(snapshot)

    result = bridge.get_learning_review()
    readiness = result["challenger_review"]["readiness"]

    assert readiness["state"] == "unavailable"
    assert readiness["leading_candidate_id"] is None
    assert readiness["reason_codes"] == ["revision_mismatch"]
    assert readiness["next_allowed_action"] == "none"
    assert result["challenger_review"]["candidates"][0]["status"] == "unavailable"


def test_unknown_revisions_never_match_each_other(bridge):
    snapshot = _learning_snapshot()
    snapshot["source_revision"] = "unknown"
    snapshot["challenger_review"]["evidence_revision"] = "unknown"
    _mark_ready(snapshot, prospective=True, all_proofs=True)
    _save_runtime_revision(bridge, "unknown")
    assert bridge.save_learning_snapshot(snapshot)

    result = bridge.get_learning_review()

    assert result["challenger_review"]["readiness"]["state"] == "unavailable"
    assert result["challenger_review"]["readiness"]["reason_codes"] == [
        "revision_mismatch"
    ]


def test_runtime_revision_mismatch_fails_learning_provenance_closed(bridge):
    snapshot = _learning_snapshot()
    _mark_ready(snapshot, prospective=True, all_proofs=True)
    _save_runtime_revision(bridge, "b" * 40)
    assert bridge.save_learning_snapshot(snapshot)

    result = bridge.get_learning_review()

    assert result["source_revision"] == "a" * 40
    assert result["challenger_review"]["evidence_revision"] == "a" * 40
    assert result["challenger_review"]["readiness"]["state"] == "unavailable"
    assert result["challenger_review"]["readiness"]["reason_codes"] == [
        "revision_mismatch"
    ]


@pytest.mark.parametrize(
    ("observed_offset", "received_offset"),
    [
        (timedelta(minutes=-16), timedelta()),
        (timedelta(), timedelta(minutes=-16)),
        (timedelta(minutes=2), timedelta()),
        (timedelta(), timedelta(minutes=2)),
    ],
)
def test_stale_or_future_runtime_heartbeat_fails_learning_provenance_closed(
    bridge, observed_offset, received_offset
):
    now = datetime.now(timezone.utc)
    heartbeat = _heartbeat()
    heartbeat["revision"] = "a" * 40
    heartbeat["observed_at"] = (now + observed_offset).isoformat()
    bridge.save_runtime_heartbeat(
        heartbeat,
        received_at=now + received_offset,
    )
    snapshot = _learning_snapshot(observed_at=now)
    _mark_ready(snapshot, prospective=True, all_proofs=True)
    assert bridge.save_learning_snapshot(snapshot)

    result = bridge.get_learning_review()

    assert result["challenger_review"]["readiness"]["state"] == "unavailable"
    assert result["challenger_review"]["readiness"]["reason_codes"] == [
        "revision_mismatch"
    ]
    assert result["challenger_review"]["readiness"]["next_allowed_action"] == "none"


def test_corrupt_cached_learning_payload_is_unavailable(bridge):
    assert bridge.save_learning_snapshot(
        {
            "observed_at_utc": datetime.now(timezone.utc).isoformat(),
            "raw_trades": [{"order_id": "must-not-cross-boundary"}],
        }
    )

    result = bridge.get_learning_review()

    assert result["freshness"]["state"] == "unknown"
    assert result["challenger_review"]["readiness"]["reason_codes"] == [
        "report_invalid"
    ]
    assert "raw_trades" not in result


def test_learning_review_never_exposes_sensitive_keys(bridge):
    snapshot = _learning_snapshot()
    assert _save_learning_with_matching_runtime(bridge, snapshot)
    result = bridge.get_learning_review()
    forbidden = {
        "account_id",
        "balance",
        "credential",
        "equity",
        "features",
        "order_id",
        "price",
        "raw_rows",
        "raw_signal",
        "secret",
        "signal",
        "token",
        "trade_id",
        "transaction_id",
    }

    def walk(value):
        if isinstance(value, dict):
            assert forbidden.isdisjoint(value)
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(result)
    assert result["learning_status"]["shadow_learning"]["auto_apply"] is False
    assert result["challenger_review"]["readiness"][
        "auto_promotion_permitted"
    ] is False


@pytest.mark.anyio
async def test_learning_review_is_a_read_only_mcp_tool(bridge):
    snapshot = _learning_snapshot()
    _save_learning_with_matching_runtime(bridge, snapshot)

    async with Client(bridge.mcp, raise_exceptions=True) as client:
        listed = await client.list_tools()
        tool_by_name = {tool.name: tool for tool in listed.tools}
        learning_tool = tool_by_name["get_learning_review"]
        assert learning_tool.annotations.read_only_hint is True
        assert learning_tool.annotations.destructive_hint is False
        assert not any(
            fragment in name
            for name in tool_by_name
            for fragment in ("write", "trade", "promote", "deploy", "optimise")
        )

        result = await client.call_tool("get_learning_review", {})
        assert result.is_error is not True
        assert result.structured_content["governance"]["can_trade"] is False
        assert result.structured_content["governance"]["can_promote"] is False
