from __future__ import annotations

import os
import secrets
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Literal

from mcp.server import MCPServer
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import ToolAnnotations
from pydantic import BaseModel, ConfigDict, Field
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from bridge_state import load_runtime_heartbeat, save_runtime_heartbeat


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
STATUS_KEY_ENV = "MOSSY_MCP_STATUS_KEY"
STATUS_STALE_SECONDS = 900
SUPERVISOR_EQUITY_FLOOR_AUD = 5_000.0

READ_ONLY_ANNOTATIONS = ToolAnnotations(
    read_only_hint=True,
    destructive_hint=False,
    idempotent_hint=True,
    open_world_hint=False,
)


class RuntimeHeartbeat(BaseModel):
    """Sanitised worker telemetry. No account ID, equity, orders, or secrets."""

    model_config = ConfigDict(extra="forbid")

    observed_at: datetime
    service_status: Literal["starting", "running", "broker-unavailable"]
    mode: str = Field(min_length=1, max_length=16)
    oanda_environment: str = Field(min_length=1, max_length=16)
    scheduler_alive: bool
    decision_cycle_fresh: bool
    broker_sync_fresh: bool
    has_open_trades: bool | None = None
    supervisor_floor_breached: bool
    revision: str = Field(min_length=1, max_length=120)


mcp = MCPServer(
    "Mossy 4X Read-Only Monitor",
    version="1.0.0",
    instructions=(
        "Read-only supervision for Mossy 4X. This server cannot place or close "
        "orders, change strategy parameters, run optimisation, write GitHub, or deploy."
    ),
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _bearer_token(request: Request) -> str:
    header = request.headers.get("authorization", "")
    scheme, separator, value = header.partition(" ")
    if not separator or scheme.lower() != "bearer":
        return ""
    return value.strip()


def _runtime_supervision(snapshot: dict | None) -> dict:
    if snapshot is None:
        return {
            "bridge_status": "ok",
            "telemetry_received": False,
            "telemetry_fresh": False,
            "supervisor_status": "BLOCKED",
            "blockers": ["No worker heartbeat has been received."],
            "read_only": True,
        }

    try:
        received_at = datetime.fromisoformat(snapshot["received_at"])
        payload_model = RuntimeHeartbeat.model_validate(snapshot["payload"])
        observed_at = payload_model.observed_at
        if received_at.tzinfo is None:
            received_at = received_at.replace(tzinfo=timezone.utc)
        if observed_at.tzinfo is None:
            observed_at = observed_at.replace(tzinfo=timezone.utc)
        received_at = received_at.astimezone(timezone.utc)
        observed_at = observed_at.astimezone(timezone.utc)
        payload = payload_model.model_dump(mode="json")
    except Exception:
        return {
            "bridge_status": "degraded",
            "telemetry_received": True,
            "telemetry_fresh": False,
            "supervisor_status": "BLOCKED",
            "blockers": ["Stored worker telemetry is invalid."],
            "read_only": True,
        }

    now = _utc_now()
    heartbeat_age_seconds = max(0.0, (now - observed_at).total_seconds())
    transport_age_seconds = max(0.0, (now - received_at).total_seconds())
    telemetry_fresh = (
        -60.0 <= (now - observed_at).total_seconds() <= STATUS_STALE_SECONDS
        and -60.0 <= (now - received_at).total_seconds() <= STATUS_STALE_SECONDS
    )

    blockers: list[str] = []
    if not telemetry_fresh:
        blockers.append("Worker telemetry is stale.")
    if payload["service_status"] != "running":
        blockers.append(f"Worker status is {payload['service_status']}.")
    if payload["mode"].lower() != "demo":
        blockers.append("MODE is not demo.")
    if payload["oanda_environment"].lower() != "practice":
        blockers.append("OANDA environment is not practice.")
    if not payload["scheduler_alive"]:
        blockers.append("Scheduler is not alive.")
    if not payload["decision_cycle_fresh"]:
        blockers.append("Decision cycle is stale.")
    if not payload["broker_sync_fresh"]:
        blockers.append("Broker sync is stale.")
    if payload["supervisor_floor_breached"]:
        blockers.append(
            f"The AUD {SUPERVISOR_EQUITY_FLOOR_AUD:,.0f} supervisory equity floor is breached."
        )

    return {
        "bridge_status": "ok",
        "telemetry_received": True,
        "telemetry_fresh": telemetry_fresh,
        "heartbeat_age_seconds": round(heartbeat_age_seconds, 1),
        "transport_age_seconds": round(transport_age_seconds, 1),
        "last_observed_utc": observed_at.isoformat(),
        "last_heartbeat_received_utc": received_at.isoformat(),
        "runtime": payload,
        "supervisor_status": "BLOCKED" if blockers else "READY_FOR_REVIEW",
        "blockers": blockers,
        "read_only": True,
        "authority_note": (
            "READY_FOR_REVIEW is not permission to trade. The deterministic Mossy 4X "
            "algorithm remains the only execution path, and Luke retains change authority."
        ),
    }


@mcp.tool(
    title="Get Mossy runtime health",
    description=(
        "Return the latest sanitised Mossy 4X worker heartbeat and supervisory blockers. "
        "This is read-only and cannot affect trading."
    ),
    annotations=READ_ONLY_ANNOTATIONS,
    structured_output=True,
)
def get_runtime_health() -> dict[str, Any]:
    """Inspect sanitised runtime health without reading credentials or account values."""

    return _runtime_supervision(load_runtime_heartbeat())


@mcp.tool(
    title="Get Mossy guardrails",
    description="Return the non-negotiable Mossy 4X supervisory safety contract.",
    annotations=READ_ONLY_ANNOTATIONS,
    structured_output=True,
)
def get_algo_guardrails() -> dict[str, Any]:
    """Return the fixed advisory and execution boundaries for the Mossy agent."""

    return {
        "execution_authority": "Deterministic Mossy 4X algorithm only",
        "agent_role": "Read-only supervisor and analyst",
        "allowed_environment": {"mode": "demo", "oanda": "practice"},
        "live_trading_permitted": False,
        "supervisor_equity_floor_aud": SUPERVISOR_EQUITY_FLOOR_AUD,
        "normal_risk_limit_pct": 1.0,
        "absolute_risk_limit_pct": 2.0,
        "requires_luke_approval": [
            "strategy or parameter changes",
            "Champion changes",
            "deployment",
            "spend",
            "any live-trading enablement",
        ],
        "mcp_capabilities": [
            "read sanitised runtime health",
            "read this safety contract",
            "read the latest public algorithm report",
        ],
        "mcp_forbidden_capabilities": [
            "place, amend, or close orders",
            "run or apply optimisation",
            "write GitHub or secrets",
            "trigger a deploy",
            "change strategy configuration",
        ],
    }


@mcp.tool(
    title="Get latest Mossy algorithm report",
    description=(
        "Read the newest public weekly algorithm report committed in the repository. "
        "Reports are research context only and never auto-apply changes."
    ),
    annotations=READ_ONLY_ANNOTATIONS,
    structured_output=True,
)
def get_latest_algo_report() -> dict[str, Any]:
    """Return the latest committed weekly report, bounded to a safe text size."""

    report_dir = REPOSITORY_ROOT / "reports" / "algo-weekly"
    reports = sorted(report_dir.glob("*.md"), reverse=True)
    if not reports:
        return {"available": False, "reason": "No weekly report is committed."}

    report_path = reports[0]
    text = report_path.read_text(encoding="utf-8")
    limit = 30_000
    try:
        report_date = date.fromisoformat(report_path.stem)
        report_age_days = max(0, (_utc_now().date() - report_date).days)
    except ValueError:
        report_date = None
        report_age_days = None
    stale = report_age_days is None or report_age_days > 14
    return {
        "available": True,
        "report_path": str(report_path.relative_to(REPOSITORY_ROOT)),
        "report": text[:limit],
        "truncated": len(text) > limit,
        "report_date": report_date.isoformat() if report_date else None,
        "report_age_days": report_age_days,
        "stale": stale,
        "freshness_warning": (
            "This report is stale and must not be treated as current evidence."
            if stale
            else None
        ),
        "governance": "Research context only; no automatic strategy changes.",
    }


@mcp.custom_route("/health", methods=["GET"])
async def health(_: Request) -> Response:
    key_configured = bool(os.getenv(STATUS_KEY_ENV, "").strip())
    snapshot = load_runtime_heartbeat()
    supervision = _runtime_supervision(snapshot)
    return JSONResponse(
        {
            "status": "ok" if key_configured else "degraded",
            "read_only": True,
            "status_key_configured": key_configured,
            "telemetry_received": supervision["telemetry_received"],
            "telemetry_fresh": supervision["telemetry_fresh"],
        },
        status_code=200 if key_configured else 503,
    )


@mcp.custom_route("/internal/runtime-heartbeat", methods=["POST"])
async def ingest_runtime_heartbeat(request: Request) -> Response:
    expected = os.getenv(STATUS_KEY_ENV, "").strip()
    if not expected:
        return JSONResponse(
            {"detail": f"Server missing {STATUS_KEY_ENV}."}, status_code=503
        )
    supplied = _bearer_token(request)
    if not supplied or not secrets.compare_digest(supplied, expected):
        return JSONResponse({"detail": "Unauthorized."}, status_code=401)

    try:
        payload = RuntimeHeartbeat.model_validate(await request.json())
    except Exception:
        return JSONResponse({"detail": "Invalid heartbeat payload."}, status_code=422)

    observed_at = payload.observed_at
    if observed_at.tzinfo is None:
        observed_at = observed_at.replace(tzinfo=timezone.utc)
    observed_age = (_utc_now() - observed_at).total_seconds()
    if observed_age < -60 or observed_age > STATUS_STALE_SECONDS:
        return JSONResponse({"detail": "Heartbeat timestamp is stale."}, status_code=422)

    payload_json = payload.model_dump(mode="json")
    payload_json["observed_at"] = observed_at.astimezone(timezone.utc).isoformat()
    stored = save_runtime_heartbeat(payload_json)
    return JSONResponse(
        {"accepted": True, "stored": stored},
        status_code=202,
    )


def _transport_security() -> TransportSecuritySettings:
    render_host = os.getenv("RENDER_EXTERNAL_HOSTNAME", "").strip()
    allowed_hosts = ["127.0.0.1", "127.0.0.1:*", "localhost", "localhost:*"]
    if render_host:
        allowed_hosts.extend([render_host, f"{render_host}:*"])
    return TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=allowed_hosts,
        allowed_origins=[
            "http://127.0.0.1:*",
            "http://localhost:*",
            "https://platform.openai.com",
            "https://chatgpt.com",
        ],
    )


app = mcp.streamable_http_app(
    json_response=True,
    stateless_http=True,
    transport_security=_transport_security(),
)
