"""Explicit, resumable journal maintenance using read-only broker evidence."""
from collections import Counter
from contextlib import closing
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sqlite3
import tempfile

from app.broker import read_trade_details


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _backup(path, target):
    """Copy SQLite's committed view, including WAL, before any data corrections."""
    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=target.parent, suffix=".sqlite")
    os.close(fd)
    try:
        with closing(sqlite3.connect(path)) as source, closing(sqlite3.connect(temporary)) as destination:
            source.backup(destination)
            destination.execute("PRAGMA journal_mode=DELETE")
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _save_resolution(journal, row, kind, broker_id, evidence, run_id):
    # The original row and event history are retained verbatim. Only an audited
    # classification is added, atomically with the saved original and evidence.
    with journal._connect() as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("BEGIN IMMEDIATE")
        current = conn.execute("SELECT * FROM trades WHERE trade_id=?", (row["trade_id"],)).fetchone()
        if (current is None or current["exit_timestamp_utc"] is not None
                or current["instrument"] != row["instrument"]):
            conn.rollback()
            return False
        if kind == "DUPLICATE_ALIAS":
            canonical = conn.execute("SELECT instrument FROM trades WHERE trade_id=?", (broker_id,)).fetchone()
            if canonical is None or canonical["instrument"] not in (None, row["instrument"]):
                conn.rollback()
                return False
        result = conn.execute("""INSERT OR IGNORE INTO journal_entry_resolutions
            (trade_id,kind,broker_id,resolved_at,run_id,original_row_json,evidence_json)
            VALUES (?,?,?,?,?,?,?)""", (row["trade_id"], kind, broker_id, _utc(), run_id,
                json.dumps(dict(current), sort_keys=True, allow_nan=False),
                json.dumps(evidence, sort_keys=True, allow_nan=False)))
        conn.commit()
        return result.rowcount == 1


def _get(client, account, resource, identifier):
    response = client.get(f"/v3/accounts/{account}/{resource}/{identifier}")
    if response.status_code in (401, 403, 429) or response.status_code >= 500:
        raise RuntimeError(f"broker read unavailable: HTTP {response.status_code}")
    if response.status_code != 200:
        return None
    payload = response.json()
    value = payload.get(resource[:-1]) if isinstance(payload, dict) else None
    if not isinstance(value, dict) or str(value.get("id")) != str(identifier):
        return None
    return value


def _inspect(journal, client, broker, guard, row, run_id):
    ticket = str(row["trade_id"])
    if not ticket.isascii() or not ticket.isdigit():
        return "UNRESOLVED_LOCAL_ID"
    order = _get(client, broker.account, "orders", ticket)
    broker_id = ticket
    evidence = {"account_id": broker.account}
    if order:
        if order.get("instrument") != row["instrument"]:
            return "UNRESOLVED_IDENTITY"
        created = guard._parse_datetime(order.get("createTime"))
        recorded = guard._parse_datetime(row["timestamp_utc"])
        if created is None or recorded is None or abs((created - recorded).total_seconds()) > 300:
            return "UNRESOLVED_ENTRY_TIME"
        evidence["order"] = order
        if order.get("state") == "CANCELLED":
            if (order.get("type") != "MARKET" or any(order.get(key) for key in (
                    "fillingTransactionID", "tradeOpenedID", "tradeReducedID", "tradeClosedIDs"))):
                return "UNRESOLVED_CANCEL_EVIDENCE"
            cancel_id = str(order.get("cancellingTransactionID") or "")
            cancel = _get(client, broker.account, "transactions", cancel_id) if cancel_id.isdigit() else None
            if (not cancel or cancel.get("type") != "ORDER_CANCEL"
                    or str(cancel.get("orderID")) != ticket
                    or guard._parse_datetime(cancel.get("time")) is None):
                return "UNRESOLVED_CANCEL_EVIDENCE"
            evidence["cancellation"] = cancel
            if _save_resolution(journal, row, "CANCELLED_ORDER", ticket, evidence, run_id):
                print(f"[JOURNAL-CLEANUP][CANCELLED] order_id={ticket} "
                      f"reason={cancel.get('reason', 'UNKNOWN')}", flush=True)
                return "CANCELLED_ORDER"
            return "UNRESOLVED_WRITE_CONFLICT"
        if order.get("state") != "FILLED":
            return "UNRESOLVED_ORDER_STATE"
        broker_id = guard._extract_trade_id(order)
        if not broker_id:
            fill_id = str(order.get("fillingTransactionID") or "")
            fill = _get(client, broker.account, "transactions", fill_id) if fill_id.isdigit() else None
            if not fill or str(fill.get("orderID")) != ticket:
                return "UNRESOLVED_FILL_LINK"
            evidence["fill"] = fill
            broker_id = guard._extract_trade_id(fill)
        if not broker_id:
            return "UNRESOLVED_NO_OPENING_FILL"
    details = read_trade_details(client, broker.account, broker_id)
    if not details or details.get("instrument") != row["instrument"]:
        return "UNRESOLVED_TRADE_EVIDENCE"
    evidence["trade"] = details
    if broker_id != ticket:
        with journal._connect() as conn:
            canonical = conn.execute("SELECT 1 FROM trades WHERE trade_id=?", (broker_id,)).fetchone()
        if canonical:
            return ("DUPLICATE_ALIAS" if _save_resolution(journal, row, "DUPLICATE_ALIAS",
                    broker_id, evidence, run_id) else "UNRESOLVED_WRITE_CONFLICT")
    fill = guard._closed_fill_from_details(details)
    if fill is None:
        return "UNRESOLVED_CLOSE_EVIDENCE"
    return ("RECOVERED_CLOSE" if guard._record_exact_fast_close(row, broker_id, fill,
            now_utc=datetime.now(timezone.utc)) else "UNRESOLVED_CLOSE_RECORD")


def run_cleanup(journal, broker, guard, run_id, *, demo_mode, oanda_env, open_positions_count):
    """Sweep a frozen set once; persist progress and refuse to guess missing data."""
    if not demo_mode or oanda_env != "practice" or getattr(broker, "mode", None) != "demo":
        return {"status": "practice-demo-required"}
    if type(open_positions_count) is not int or open_positions_count != 0:
        return {"status": "confirmed-flat-account-required"}
    if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", run_id):
        return {"status": "invalid-run-id"}
    with journal._connect() as conn:
        conn.row_factory = sqlite3.Row
        saved = conn.execute("SELECT * FROM journal_cleanup_runs WHERE run_id=?", (run_id,)).fetchone()
    if saved:
        outcomes = json.loads(saved["outcomes_json"])
        if saved["completed_at"]:
            return {"status": "already-completed", "counts": dict(Counter(outcomes.values()))}
        snapshot = json.loads(saved["snapshot_json"])
        backup = Path(saved["backup_path"])
    else:
        with journal._connect() as conn:
            snapshot = [str(row[0]) for row in conn.execute(
                "SELECT trade_id FROM trades WHERE exit_timestamp_utc IS NULL "
                "AND NOT EXISTS (SELECT 1 FROM journal_entry_resolutions r "
                "WHERE r.trade_id=trades.trade_id) ORDER BY timestamp_utc, trade_id").fetchall()]
        if len(snapshot) > 500:
            return {"status": "manual-batch-required", "candidates": len(snapshot)}
        backup = journal.path.parent / "journal-backups" / f"before-{run_id}.sqlite"
        _backup(journal.path, backup)
        outcomes = {}
        with journal._connect() as conn:
            conn.execute("INSERT INTO journal_cleanup_runs VALUES (?,?,NULL,?,?,?)",
                         (run_id, _utc(), json.dumps(snapshot), "{}", str(backup)))
    print(f"[JOURNAL-CLEANUP][START] id={run_id} candidates={len(snapshot)} backup={backup}", flush=True)
    try:
        # A single connection is reused for read-only requests throughout the sweep.
        with broker._client() as client:
            for ticket in snapshot:
                if ticket in outcomes:
                    continue
                with journal._connect() as conn:
                    conn.row_factory = sqlite3.Row
                    row = conn.execute("SELECT * FROM trades WHERE trade_id=?", (ticket,)).fetchone()
                    resolved = conn.execute("SELECT kind FROM journal_entry_resolutions WHERE trade_id=?", (ticket,)).fetchone()
                if resolved:
                    outcome = resolved["kind"]
                elif row is None:
                    outcome = "UNRESOLVED_MISSING_ROW"
                elif row["exit_timestamp_utc"] is not None:
                    outcome = "ALREADY_CLOSED"
                else:
                    outcome = _inspect(journal, client, broker, guard, row, run_id)
                outcomes[ticket] = outcome
                with journal._connect() as conn:
                    conn.execute("UPDATE journal_cleanup_runs SET outcomes_json=? WHERE run_id=?",
                                 (json.dumps(outcomes, sort_keys=True), run_id))
                if len(outcomes) % 10 == 0:
                    print(f"[JOURNAL-CLEANUP][PROGRESS] checked={len(outcomes)}/{len(snapshot)} "
                          f"counts={dict(Counter(outcomes.values()))}", flush=True)
    except Exception as exc:
        print(f"[JOURNAL-CLEANUP][DEFER] id={run_id} checked={len(outcomes)} error={exc}", flush=True)
        return {"status": "incomplete", "counts": dict(Counter(outcomes.values()))}
    with journal._connect() as conn:
        conn.execute("UPDATE journal_cleanup_runs SET completed_at=? WHERE run_id=?", (_utc(), run_id))
    result = {"status": "completed", "counts": dict(Counter(outcomes.values())),
              "unresolved_ids": [key for key, value in outcomes.items() if value.startswith("UNRESOLVED")],
              "backup": str(backup)}
    print(f"[JOURNAL-CLEANUP][COMPLETE] id={run_id} result={json.dumps(result, sort_keys=True)}", flush=True)
    return result
