from __future__ import annotations

import json
import math
import os
import time
from datetime import datetime, timezone

import httpx


DEFAULT_SUPERVISOR_EQUITY_FLOOR_AUD = 5_000.0
DEFAULT_FRESHNESS_SECONDS = 180.0
# Keep the free bridge warm only while Mossy can act (or has an open trade).
# Outside those periods the bridge may sleep; missing telemetry fails closed.
ACTIVE_PUBLISH_INTERVAL_SECONDS = 600.0
IDLE_PUBLISH_INTERVAL_SECONDS = 14_400.0

_last_success_monotonic: float | None = None
_last_success_fingerprint: str | None = None


def _fresh(age_seconds: float | None, threshold: float) -> bool:
    return age_seconds is not None and 0.0 <= float(age_seconds) <= threshold


def build_runtime_heartbeat(
    *,
    service_status: str,
    mode: str,
    oanda_environment: str,
    scheduler_alive: bool,
    last_cycle_age_sec: float | None,
    last_broker_sync_age_sec: float | None,
    open_trades_count: int | None,
    equity: float | None,
    revision: str,
    observed_at: datetime | None = None,
    supervisor_equity_floor_aud: float = DEFAULT_SUPERVISOR_EQUITY_FLOOR_AUD,
    freshness_seconds: float = DEFAULT_FRESHNESS_SECONDS,
) -> dict:
    """Build a deliberately sanitised monitoring payload.

    Exact equity, account identifiers, orders, signals, and credentials are never
    included. The equity floor is represented only as a boolean safety condition.
    """

    timestamp = observed_at or datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    try:
        equity_value = float(equity) if equity is not None else None
    except (TypeError, ValueError):
        equity_value = None
    floor_breached = (
        equity_value is None
        or not math.isfinite(equity_value)
        or equity_value < float(supervisor_equity_floor_aud)
    )

    return {
        "observed_at": timestamp.astimezone(timezone.utc).isoformat(),
        "service_status": service_status,
        "mode": mode.strip().lower(),
        "oanda_environment": oanda_environment.strip().lower(),
        "scheduler_alive": bool(scheduler_alive),
        "decision_cycle_fresh": _fresh(last_cycle_age_sec, freshness_seconds),
        "broker_sync_fresh": _fresh(last_broker_sync_age_sec, freshness_seconds),
        "has_open_trades": None
        if open_trades_count is None
        else bool(open_trades_count > 0),
        "supervisor_floor_breached": floor_breached,
        "revision": revision,
    }


def _safety_fingerprint(payload: dict) -> str:
    """Fingerprint state changes while deliberately ignoring the timestamp."""

    safety_state = {key: value for key, value in payload.items() if key != "observed_at"}
    return json.dumps(safety_state, sort_keys=True, separators=(",", ":"))


async def publish_runtime_heartbeat(
    payload: dict, *, monitoring_active: bool = False
) -> tuple[bool, str]:
    """Send monitoring data without ever raising into the trading loop."""

    global _last_success_fingerprint, _last_success_monotonic

    url = os.getenv("MOSSY_MCP_STATUS_URL", "").strip()
    key = os.getenv("MOSSY_MCP_STATUS_KEY", "").strip()
    if not url or not key:
        return False, "disabled"
    now_monotonic = time.monotonic()
    fingerprint = _safety_fingerprint(payload)
    interval = (
        ACTIVE_PUBLISH_INTERVAL_SECONDS
        if monitoring_active
        else IDLE_PUBLISH_INTERVAL_SECONDS
    )
    if (
        _last_success_monotonic is not None
        and _last_success_fingerprint == fingerprint
        and now_monotonic - _last_success_monotonic < interval
    ):
        return False, "throttled"
    if not url.startswith("https://") and not os.getenv(
        "MOSSY_MCP_ALLOW_INSECURE_HTTP", ""
    ).strip().lower() in {"1", "true", "yes"}:
        return False, "insecure-url-blocked"

    try:
        async with httpx.AsyncClient(timeout=3.0, follow_redirects=False) as client:
            response = await client.post(
                url,
                headers={"Authorization": f"Bearer {key}"},
                json=payload,
            )
            response.raise_for_status()
    except httpx.HTTPError as exc:
        return False, f"http-error:{type(exc).__name__}"
    except Exception as exc:  # pragma: no cover - defensive fail-closed guard
        return False, f"error:{type(exc).__name__}"
    _last_success_monotonic = now_monotonic
    _last_success_fingerprint = fingerprint
    return True, "sent"
