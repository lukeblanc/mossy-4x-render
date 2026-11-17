"""Explicit, incident-bound demo halt recovery before any worker threads start.

No broker writes, journal repairs, risk resets, or automatic clean-poll recovery.
An operator request expires and matches the persisted incident's modification
time. Archiving that exact marker makes the request unusable for future halts.
"""
from __future__ import annotations

import json
import hashlib
import os
import sqlite3
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from app.broker import PRACTICE
from app.config import settings
from app.trade_transactions import read_full_close
from src.account_reconciliation import number, timestamp
from src.cash_risk import configured_cash_risk_limit

REASON = "protective-stop-audit-unavailable"


class RecoveryBlocked(ValueError):
    pass


def require(condition, reason):
    if not condition:
        raise RecoveryBlocked(reason)


def _id(value):
    text = str(value)
    require(text.isascii() and text.isdigit() and int(text) > 0, "invalid-request-id")
    return text


def _get(client, account, suffix):
    response = client.get(f"/v3/accounts/{account}/{suffix}")
    require(response.status_code == 200, "broker-read-unavailable")
    payload = response.json()
    require(isinstance(payload, dict), "invalid-broker-response")
    return payload


def _flat(client, account):
    payload = _get(client, account, "summary")
    summary = payload.get("account", {})
    require(isinstance(summary, dict) and summary.get("id") == account
            and summary.get("currency") == "AUD", "account-unverified")
    require(all(type(summary.get(k)) is int and summary[k] == 0 for k in
                ("openTradeCount", "openPositionCount", "pendingOrderCount")), "account-not-flat")
    cursor = _id(payload.get("lastTransactionID"))
    require(str(summary.get("lastTransactionID", cursor)) == cursor, "cursor-conflict")
    for suffix, key in (("openTrades", "trades"), ("openPositions", "positions"),
                        ("pendingOrders", "orders")):
        detail = _get(client, account, suffix)
        require(detail.get(key) == [] and str(detail.get("lastTransactionID")) == cursor,
                "exposure-or-cursor-changed")
    return cursor


def recover_reviewed_incident(broker, journal_path, raw_request, *, now=None):
    """Run only in the single-threaded boot path with an explicit operator request."""
    if not raw_request:
        return {"status": "not-requested"}
    try:
        now = now or datetime.now(timezone.utc)
        request = json.loads(raw_request)
        require(isinstance(request, dict), "invalid-request")
        trade_id, stop_id = _id(request.get("trade_id")), _id(request.get("stop_order_id"))
        incident = timestamp(request.get("incident_at_utc"))
        expiry = timestamp(request.get("expires_at_utc"))
        require(incident < now < expiry and (expiry - now).total_seconds() <= 3600,
                "request-expired-or-invalid")
        require(broker.mode == "demo" and settings.OANDA_ENV == "practice"
                and broker.base_url == PRACTICE, "demo-practice-required")
        require(0 < configured_cash_risk_limit() <= .20
                and 0 < number(os.getenv("HARD_MAX_LOSS_CCY")) <= Decimal(".20"),
                "approved-cash-limits-required")
        require(os.getenv("MAX_CONCURRENT_POSITIONS") == "1"
                and os.getenv("MAX_OPEN_TRADES") == "1", "single-position-required")
        marker = broker._entry_halt_path
        require(not marker.is_symlink(), "invalid-halt-marker")
        original = marker.read_bytes()
        stat = marker.stat()
        require(original.decode().strip() == REASON and broker.entry_halt_reason == REASON,
                "different-halt-reason")
        require(abs(stat.st_mtime - incident.timestamp()) <= 10, "different-halt-incident")
        archive = marker.with_name(f"broker_entry_halt.recovered-{stat.st_mtime_ns}.txt")
        request_key = hashlib.sha256(f"{trade_id}:{stop_id}:{incident.isoformat()}".encode()).hexdigest()[:24]
        receipt = marker.with_name(f"broker_entry_halt.recovery-{request_key}.json")
        require(not archive.exists() and not receipt.exists(), "request-already-consumed")

        with broker._client() as client:
            before = _flat(client, broker.account)
            closed = read_full_close(client, broker.account, trade_id)
            require(isinstance(closed, dict) and closed.get("state") == "CLOSED",
                    "exact-close-unverified")
            evidence = closed["_close_evidence"]
            closing_id = _id(evidence["closing"]["id"])
            require(int(closing_id) <= int(before), "close-cursor-conflict")
            require(abs((timestamp(closed["closeTime"]) - incident).total_seconds()) <= 120,
                    "close-does-not-match-incident")
            order = _get(client, broker.account, f"orders/{stop_id}").get("order")
            require(isinstance(order, dict) and str(order.get("id")) == stop_id
                    and str(order.get("tradeID")) == trade_id
                    and order.get("type") in {"STOP_LOSS", "GUARANTEED_STOP_LOSS"}
                    and order.get("state") in {"CANCELLED", "FILLED"}, "historical-stop-unverified")
            require(number(order.get("price")) > 0, "invalid-stop-price")
            # Read-only DB connection: recovery cannot manufacture reconciliation.
            with sqlite3.connect(Path(journal_path).resolve().as_uri() + "?mode=ro", uri=True) as db:
                row = db.execute("SELECT instrument, units, entry_price, stop_loss_price, "
                                 "exit_timestamp_utc, exit_price, realized_pnl_ccy, broker_confirmed, side "
                                 "FROM trades WHERE trade_id=?", (trade_id,)).fetchone()
            require(row is not None and row[0] == closed["instrument"] and row[7] == 1,
                    "journal-close-unverified")
            require(row[8] == ("BUY" if number(closed["initialUnits"]) > 0 else "SELL"),
                    "journal-direction-mismatch")
            require(number(row[1]) == abs(number(closed["initialUnits"])), "journal-units-mismatch")
            for actual, expected in ((row[2], closed["price"]), (row[3], order["price"])):
                require(abs(number(actual) - number(expected)) <= Decimal("0.000000001"),
                        "journal-detail-mismatch")
            # The normal profit guard records a confirmed close with no exit
            # price; exact transaction evidence supplies it without rewriting DB.
            if row[5] is not None:
                require(abs(number(row[5]) - number(closed["averageClosePrice"])) <= Decimal("0.000000001"),
                        "journal-exit-price-mismatch")
            require(abs(number(row[6]) - number(closed["realizedPL"])) <= Decimal(".01")
                    and abs((timestamp(row[4]) - timestamp(closed["closeTime"])).total_seconds()) <= 5,
                    "journal-close-mismatch")
            require(_flat(client, broker.account) == before, "account-changed-during-recovery")

        require((marker.stat().st_ino, marker.stat().st_mtime_ns, marker.read_bytes())
                == (stat.st_ino, stat.st_mtime_ns, original), "halt-changed-during-recovery")
        result = {"status": "verified", "trade_id": trade_id, "stop_order_id": stop_id,
                  "closing_transaction_id": closing_id, "last_transaction_id": before,
                  "incident_at_utc": incident.isoformat(), "verified_at_utc": now.isoformat(),
                  "journal_reconciled": True, "journal_exit_price_present": row[5] is not None,
                  "flat_verified": True}
        # Durable receipt first. An interrupted attempt remains blocked for review.
        with receipt.open("x", encoding="utf-8") as out:
            json.dump(result, out, sort_keys=True)
            out.flush()
            os.fsync(out.fileno())
        marker.rename(archive)
        broker._entry_halted_reason = None
        broker._invalidate_demo_trade_evidence()
        result["status"] = "recovered"
    except Exception as exc:
        # Never publish raw error text, broker responses, URLs or identifiers.
        result = {"status": "blocked", "reason": str(exc) if isinstance(exc, RecoveryBlocked)
                  else "evidence-or-storage-unavailable"}
    print("[BROKER][RECOVERY] " + json.dumps(result, sort_keys=True), flush=True)
    return result
