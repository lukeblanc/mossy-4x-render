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

ENTRY_WINDOW_STATES = frozenset(
    {"weekend_locked", "in_configured_session", "off_session"}
)
VERIFIED_ENTRY_AGE_BUCKETS = frozenset(
    {
        "never",
        "under_1h",
        "under_24h",
        "one_to_three_days",
        "over_three_days",
        "unknown",
    }
)
BROKER_ENTRY_HALT_REASONS = frozenset(
    {
        "account-currency-mismatch",
        "protective-stop-audit-unavailable",
        "unprotected-open-trade",
        "invalid-order-response",
        "no-confirmed-trade-opening",
        "fill-does-not-match-request",
        "protective-stop-not-confirmed",
        "order-http-state-uncertain",
        "order-transport-state-uncertain",
        "other",
    }
)
BROKER_HALT_TELEMETRY_FIELDS = frozenset(
    {"broker_entry_halted", "broker_entry_halt_reason"}
)

_last_success_monotonic: float | None = None
_last_success_fingerprint: str | None = None


def _fresh(age_seconds: float | None, threshold: float) -> bool:
    return age_seconds is not None and 0.0 <= float(age_seconds) <= threshold


def classify_verified_entry_age(
    latest_entry_at: datetime | None, *, observed_at: datetime | None = None
) -> str:
    """Reduce a verified entry timestamp to a non-sensitive recency bucket."""

    if latest_entry_at is None:
        return "never"
    if not isinstance(latest_entry_at, datetime):
        return "unknown"

    timestamp = observed_at or datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    entry_timestamp = latest_entry_at
    if entry_timestamp.tzinfo is None:
        entry_timestamp = entry_timestamp.replace(tzinfo=timezone.utc)

    try:
        age_seconds = (
            timestamp.astimezone(timezone.utc)
            - entry_timestamp.astimezone(timezone.utc)
        ).total_seconds()
    except (OverflowError, ValueError):
        return "unknown"

    # Tolerate ordinary clock skew, but do not describe a materially future
    # timestamp as recent activity.
    if age_seconds < -60.0:
        return "unknown"
    age_seconds = max(0.0, age_seconds)
    if age_seconds < 3_600.0:
        return "under_1h"
    if age_seconds < 86_400.0:
        return "under_24h"
    if age_seconds < 259_200.0:
        return "one_to_three_days"
    return "over_three_days"


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
    entry_window_state: str,
    last_verified_entry_age_bucket: str,
    broker_entry_halted: bool | None = None,
    broker_entry_halt_reason: str | None = None,
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
    normalized_entry_window = str(entry_window_state).strip().lower()
    if normalized_entry_window not in ENTRY_WINDOW_STATES:
        raise ValueError("invalid entry window state")
    normalized_entry_age = str(last_verified_entry_age_bucket).strip().lower()
    if normalized_entry_age not in VERIFIED_ENTRY_AGE_BUCKETS:
        raise ValueError("invalid verified entry age bucket")
    if broker_entry_halted is not None and type(broker_entry_halted) is not bool:
        raise ValueError("invalid broker entry halt flag")
    if broker_entry_halted is not True and broker_entry_halt_reason is not None:
        raise ValueError("broker entry halt reason requires a confirmed halt")
    # Persisted reasons may come from older/future code. Never transmit free text.
    bounded_halt_reason = None
    if broker_entry_halted is True:
        bounded_halt_reason = (
            broker_entry_halt_reason
            if isinstance(broker_entry_halt_reason, str)
            and broker_entry_halt_reason in BROKER_ENTRY_HALT_REASONS
            else "other"
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
        "entry_window_state": normalized_entry_window,
        "last_verified_entry_age_bucket": normalized_entry_age,
        "broker_entry_halted": broker_entry_halted,
        "broker_entry_halt_reason": bounded_halt_reason,
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
            headers = {"Authorization": f"Bearer {key}"}
            outbound_payload = payload
            legacy_telemetry = False
            if BROKER_HALT_TELEMETRY_FIELDS.intersection(payload):
                # Old bridges reject all extra fields with a generic 422. Only
                # an unsupported capability route (405) permits a downgrade.
                capability_response = await client.get(url, headers=headers)
                if capability_response.status_code == 405:
                    outbound_payload = {
                        field: value for field, value in payload.items()
                        if field not in BROKER_HALT_TELEMETRY_FIELDS
                    }
                    legacy_telemetry = True
                else:
                    capability_response.raise_for_status()
                    try:
                        capabilities = capability_response.json()
                    except ValueError:
                        return False, "invalid-heartbeat-capabilities"
                    fields = (
                        capabilities.get("optional_heartbeat_fields")
                        if isinstance(capabilities, dict) else None
                    )
                    if (
                        not isinstance(fields, list)
                        or not all(isinstance(field, str) for field in fields)
                    ):
                        return False, "invalid-heartbeat-capabilities"
                    if not BROKER_HALT_TELEMETRY_FIELDS.issubset(fields):
                        return False, "unsupported-heartbeat-capabilities"
            response = await client.post(
                url,
                headers=headers,
                json=outbound_payload,
            )
            response.raise_for_status()
    except httpx.HTTPError as exc:
        return False, f"http-error:{type(exc).__name__}"
    except Exception as exc:  # pragma: no cover - defensive fail-closed guard
        return False, f"error:{type(exc).__name__}"
    _last_success_monotonic = now_monotonic
    _last_success_fingerprint = fingerprint
    return True, "sent:legacy-telemetry" if legacy_telemetry else "sent"
