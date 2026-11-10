"""Bounded private handoff evidence from an already-audited demo snapshot.

This module performs no I/O, broker calls, logging, sizing, or order creation.
The caller may log the returned allowlisted object to its private Render logs.
It must never publish it through the public aggregate-only MCP bridge.

A complete source snapshot still is not a destination order: the destination
broker's account, current quote, specifications and Luke's risk must be checked
separately. Snapshots are not a durable trade event stream.
"""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
import re
from typing import Any


SCHEMA_VERSION = "mossy.demo-trade-handoff.v1"
MAX_OPEN_TRADES = 10
MAX_SNAPSHOT_AGE_SECONDS = 60.0
FUTURE_TOLERANCE_SECONDS = 5.0
_INSTRUMENT = re.compile(r"[A-Z]{3}_[A-Z]{3}\Z", re.ASCII)
_REVISION = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z", re.ASCII)


def _ticket(value: object) -> str | None:
    if type(value) not in {str, int}:
        return None
    if type(value) is int and value.bit_length() > 107:
        return None
    text = str(value)
    if not 1 <= len(text) <= 32 or not text.isascii() or not text.isdigit():
        return None
    return text if int(text) > 0 else None


def _number(value: object, *, positive: bool = True) -> str | None:
    if type(value) not in {str, int, float, Decimal}:
        return None
    try:
        text = str(value).strip()
    except (ValueError, OverflowError):
        return None
    if not 1 <= len(text) <= 64:
        return None
    try:
        number = Decimal(text)
    except InvalidOperation:
        return None
    if (
        not number.is_finite()
        or number == 0
        or (positive and number < 0)
        or abs(number) > Decimal("1e12")
        or number.adjusted() < -12
    ):
        return None
    rendered = format(number, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered


def _time(value: object) -> datetime | None:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and len(value) <= 64:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except (TypeError, ValueError, OverflowError):
            return None
    else:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    try:
        return parsed.astimezone(timezone.utc)
    except (ValueError, OverflowError):
        return None


def _time_text(value: datetime | None) -> str | None:
    return value.isoformat().replace("+00:00", "Z") if value is not None else None


def _word(value: object, allowed: set[str]) -> str | None:
    if not isinstance(value, str) or len(value) > 32:
        return None
    text = value.strip().lower()
    return text if text in allowed else None


def _demo_floor(value: object) -> float | None:
    if type(value) not in {int, float}:
        return None
    try:
        floor = float(value)
    except (ValueError, OverflowError):
        return None
    return floor if math.isfinite(floor) and floor >= 500.0 else None


def _order(
    value: object, *, trade_id: str, kind: str, summary_order_id: object = None
) -> dict[str, str] | None:
    if not isinstance(value, dict):
        return None
    order_id = _ticket(value.get("id"))
    linked_trade = _ticket(value.get("tradeID"))
    price = _number(value.get("price"))
    if (
        order_id is None
        or linked_trade != trade_id
        or price is None
        or value.get("state") != "PENDING"
        or value.get("type") != kind
    ):
        return None
    if summary_order_id is not None and _ticket(summary_order_id) != order_id:
        return None
    return {
        "order_id": order_id, "trade_id": trade_id,
        "type": kind, "state": "PENDING", "price": price,
    }


def _trade(
    summary: object, detail: object, *, now: datetime, audited: bool
) -> dict[str, Any]:
    blockers: list[str] = []
    summary = summary if isinstance(summary, dict) else {}
    detail = detail if isinstance(detail, dict) else {}
    trade_id = _ticket(summary.get("id"))
    instrument = detail.get("instrument")
    instrument = instrument if isinstance(instrument, str) and _INSTRUMENT.fullmatch(instrument) else None
    if trade_id is None or _ticket(detail.get("id")) != trade_id:
        blockers.append("trade-id-unverified")
    if instrument is None or summary.get("instrument") != instrument:
        blockers.append("trade-instrument-unverified")
    if detail.get("state") != "OPEN" or (
        "state" in summary and summary["state"] != "OPEN"
    ):
        blockers.append("trade-state-unverified")
    units = _number(detail.get("currentUnits"), positive=False)
    entry = _number(detail.get("price"))
    opened_at = _time(detail.get("openTime"))
    if units is None:
        blockers.append("trade-units-unverified")
    if entry is None:
        blockers.append("trade-fill-price-unverified")
    if opened_at is None or (opened_at - now).total_seconds() > FUTURE_TOLERANCE_SECONDS:
        blockers.append("trade-open-time-unverified")
        opened_at = None
    for key, actual, parser in (
        ("currentUnits", units, lambda value: _number(value, positive=False)),
        ("price", entry, _number),
        ("openTime", opened_at, _time),
    ):
        if key in summary and (actual is None or parser(summary[key]) != actual):
            blockers.append("trade-summary-detail-mismatch")
            break

    stop = None
    regular = detail.get("stopLossOrder")
    guaranteed = detail.get("guaranteedStopLossOrder")
    if regular is not None and guaranteed is not None:
        blockers.append("protective-stop-ambiguous")
    elif trade_id is not None:
        if regular is not None:
            stop = _order(
                regular, trade_id=trade_id, kind="STOP_LOSS",
                summary_order_id=summary.get("stopLossOrderID"),
            )
        elif guaranteed is not None:
            stop = _order(
                guaranteed, trade_id=trade_id, kind="GUARANTEED_STOP_LOSS",
                summary_order_id=summary.get("guaranteedStopLossOrderID"),
            )
    if stop is None:
        blockers.append("protective-stop-unverified")

    target = None
    target_state = "unknown"
    if "takeProfitOrder" not in detail and summary.get("takeProfitOrderID") is None:
        # The absence is meaningful only in the exact, authenticated OPEN trade
        # detail from the successful protection audit, not in a summary alone.
        if audited and detail.get("state") == "OPEN" and _ticket(detail.get("id")) == trade_id:
            target_state = "absent"
    elif trade_id is not None:
        target = _order(
            detail.get("takeProfitOrder"), trade_id=trade_id, kind="TAKE_PROFIT",
            summary_order_id=summary.get("takeProfitOrderID"),
        )
        if target is not None:
            target_state = "pending"
    if target_state == "unknown":
        blockers.append("take-profit-state-unverified")
    if not audited:
        blockers.append("protection-audit-unverified")

    return {
        "source_trade_id": trade_id,
        "instrument": instrument,
        "direction": ("BUY" if Decimal(units) > 0 else "SELL") if units is not None else None,
        "signed_units": units,
        "entry_fill_price": entry,
        "opened_at_utc": _time_text(opened_at),
        "state": "OPEN" if detail.get("state") == "OPEN" else "unverified",
        "open_state_verified": (
            audited and trade_id is not None and _ticket(detail.get("id")) == trade_id
            and instrument is not None and summary.get("instrument") == instrument
            and detail.get("state") == "OPEN"
            and ("state" not in summary or summary["state"] == "OPEN")
        ),
        "stop_loss": stop,
        "take_profit": target,
        "take_profit_state": target_state,
        "take_profit_verified": audited and target_state in {"pending", "absent"},
        "take_profit_absence_verified": audited and target_state == "absent",
        "protective_stop_verified": audited and stop is not None,
        "source_verified": not blockers,
        "blockers": blockers,
    }


def build_demo_trade_handoff(
    *,
    open_trades: object,
    trade_details: object,
    protection_audited: object,
    mode: object,
    oanda_environment: object,
    account_currency: object,
    revision: object,
    demo_equity_floor_aud: object,
    snapshot_observed_at: object,
    observed_at: datetime,
    broker_entry_halted: object = None,
    demo_equity_floor_halted: object = None,
    max_age_seconds: float = MAX_SNAPSHOT_AGE_SECONDS,
) -> dict[str, Any]:
    """Return allowlisted source evidence, never a destination order or signal.

    ``trade_details`` must come from the *same* successful, fresh audited broker
    list as ``open_trades``. The caller passes the broker audit timestamp, not a
    new timestamp for an old cache. No raw account data or unknown input fields
    are copied into the result. A missing snapshot is distinct from verified flat.
    """
    if (
        type(max_age_seconds) not in {int, float}
        or not math.isfinite(max_age_seconds)
        or not 0 < max_age_seconds <= MAX_SNAPSHOT_AGE_SECONDS
    ):
        raise ValueError("snapshot maximum age must be between zero and 60 seconds")

    now = _time(observed_at)
    snapshot_at = _time(snapshot_observed_at)
    source_mode = _word(mode, {"demo", "simulation", "paper", "live"})
    source_environment = _word(oanda_environment, {"practice", "live"})
    source_currency = account_currency if isinstance(account_currency, str) and account_currency in {"AUD", "USD", "EUR", "GBP"} else None
    source_revision = revision if isinstance(revision, str) and _REVISION.fullmatch(revision) else None
    floor = _demo_floor(demo_equity_floor_aud)
    audited = protection_audited is True
    blockers: list[str] = []
    if source_mode != "demo" or source_environment != "practice":
        blockers.append("source-not-demo-practice")
    if source_currency != "AUD":
        blockers.append("source-currency-unverified")
    if source_revision is None:
        blockers.append("worker-revision-unverified")
    if floor is None:
        blockers.append("demo-equity-floor-unverified")
    if now is None:
        blockers.append("observation-time-unverified")

    age = None
    freshness = "unknown"
    if snapshot_at is None or now is None:
        blockers.append("snapshot-time-unverified")
    else:
        signed_age = (now - snapshot_at).total_seconds()
        age = round(max(signed_age, 0.0), 3)
        if signed_age < -FUTURE_TOLERANCE_SECONDS:
            freshness = "future"
            blockers.append("snapshot-in-future")
        elif signed_age > max_age_seconds:
            freshness = "stale"
            blockers.append("snapshot-stale")
        else:
            freshness = "fresh"
    if not audited:
        blockers.append("protection-audit-unverified")
    for flag, name in (
        (broker_entry_halted, "broker-entry"),
        (demo_equity_floor_halted, "demo-equity-floor"),
    ):
        if type(flag) is not bool:
            blockers.append(f"{name}-halt-unknown")
        elif flag:
            blockers.append(f"{name}-halted")

    records: list[dict[str, Any]] = []
    count = len(open_trades) if isinstance(open_trades, list) else None
    status = "unavailable"
    if not isinstance(open_trades, list):
        blockers.append("open-trade-snapshot-unavailable")
    elif count > MAX_OPEN_TRADES:
        blockers.append("open-trade-limit-exceeded")
    elif any(not isinstance(trade, dict) for trade in open_trades):
        blockers.append("open-trade-snapshot-invalid")
    elif (
        source_mode == "demo" and source_environment == "practice"
        and source_currency == "AUD" and source_revision is not None
        and now is not None and freshness == "fresh" and audited
    ):
        status = "flat" if count == 0 else "open"
        details = trade_details if isinstance(trade_details, dict) else {}
        identifiers = [_ticket(trade.get("id")) for trade in open_trades]
        if None in identifiers or len(set(identifiers)) != len(identifiers):
            blockers.append("open-trade-identity-ambiguous")
            status = "incomplete"
        else:
            for summary, trade_id in zip(open_trades, identifiers):
                record = _trade(summary, details.get(trade_id), now=now, audited=audited)
                records.append(record)
            if not isinstance(trade_details, dict) or set(details) != set(identifiers):
                blockers.append("audited-detail-set-mismatch")
                status = "incomplete"
            if any(not record["source_verified"] for record in records):
                blockers.append("trade-evidence-incomplete")
                status = "incomplete"

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "broker": "oanda", "mode": source_mode or "unknown",
            "environment": source_environment or "unknown",
            "account_currency": source_currency,
            "worker_revision": source_revision,
            "protection_audit_verified": audited,
        },
        "safety": {
            "demo_equity_floor_aud": floor,
            "demo_equity_floor_halted": demo_equity_floor_halted if type(demo_equity_floor_halted) is bool else None,
            "broker_entry_halted": broker_entry_halted if type(broker_entry_halted) is bool else None,
        },
        "observed_at_utc": _time_text(now),
        "broker_observed_at_utc": _time_text(snapshot_at),
        "snapshot_age_seconds": age,
        "freshness": freshness,
        "read_status": status,
        "open_trade_count": count,
        "trades": records,
        "handoff_ready": status == "open" and not blockers,
        "destination_order_ready": False,
        "blockers": blockers,
    }
    payload["snapshot_id"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()[:32]
    return payload
