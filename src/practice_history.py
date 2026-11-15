"""Private, GET-only practice history and after-cost research (never trading input).

Broker amounts are Decimal. Multi-trade commission is an explicitly labelled
units allocation, not a broker-provided per-trade charge. Spread is diagnostic:
it is already reflected in fill-based P&L and is NEVER deducted again.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import io
import json
import os
import sqlite3
import tempfile
import threading
from collections import defaultdict
from contextlib import closing
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import httpx

from src.account_reconciliation import (
    EvidenceError, PracticeReader, ZERO, _components, _net, identifier, number, time_key,
)
from src.execution_quality import summarize_fill_execution, aggregate_execution_quality

SCHEMA = "mossy.practice-history.v1"
MAX_NEW_TRANSACTIONS = 20000
BALANCE_TOLERANCE = Decimal("0.0001")
MONEY_FIELDS = ("gross_pl", "financing", "commission", "guaranteed_fee", "spread_cost")


def _json(value):
    return json.dumps(value, sort_keys=True, default=str, allow_nan=False)


def _allocate(total, legs):
    """Deterministic units allocation; final remainder preserves the exact total."""
    weights = [abs(number(leg["units"])) for _, leg in legs]
    if not weights or any(w <= 0 for w in weights):
        raise EvidenceError("invalid-fill-units")
    denominator = sum(weights, ZERO)
    amounts = [(total * w / denominator).quantize(Decimal("0.000000000001")) for w in weights[:-1]]
    return amounts + [total - sum(amounts, ZERO)]


def build_review(transactions, *, account, cursor, balance, journal_rows=()):
    """Rebuild from complete immutable transactions; fail closed on missing evidence."""
    ordered = sorted(transactions, key=lambda tx: identifier(tx.get("id")))
    if not ordered or ordered[0].get("type") != "CREATE" or len(ordered) != cursor:
        raise EvidenceError("incomplete-history-from-account-creation")
    trades, fills, protections = {}, [], defaultdict(list)
    by_id = {str(tx["id"]): tx for tx in ordered}
    totals = _components({"type": "CREATE"})
    running, previous_time, checkpoints = ZERO, None, 0
    for expected, tx in enumerate(ordered, 1):
        if identifier(tx["id"]) != expected or tx.get("accountID") != account:
            raise EvidenceError("transaction-gap-duplicate-or-account-mismatch")
        when = time_key(tx["time"])
        if previous_time is not None and when < previous_time:
            raise EvidenceError("transaction-time-order-mismatch")
        previous_time = when
        parts = _components(tx)
        for key, amount in parts.items():
            totals[key] += amount
        running += _net(parts)
        if _net(parts) and "accountBalance" not in tx:
            raise EvidenceError("monetary-balance-missing")
        if "accountBalance" in tx:
            checkpoints += 1
            if abs(number(tx["accountBalance"]) - running) > BALANCE_TOLERANCE:
                raise EvidenceError("intermediate-balance-mismatch")
        kind = tx["type"]
        if kind in {"STOP_LOSS_ORDER", "TAKE_PROFIT_ORDER", "TRAILING_STOP_LOSS_ORDER",
                    "GUARANTEED_STOP_LOSS_ORDER"} and tx.get("tradeID"):
            protections[str(tx["tradeID"])].append({
                key: tx[key] for key in ("id", "time", "type", "price", "distance", "replacesOrderID") if key in tx
            })
        if kind == "DIVIDEND_ADJUSTMENT" and parts["dividend_adjustment"]:
            raise EvidenceError("dividend-allocation-not-supported")
        if kind == "DAILY_FINANCING":
            daily = [leg for p in tx.get("positionFinancings", []) for leg in p.get("openTradeFinancings", [])]
            if sum((number(leg.get("financing")) for leg in daily), ZERO) != parts["financing"]:
                raise EvidenceError("daily-financing-breakdown-mismatch")
            for leg in daily:
                trade = trades.get(str(leg["tradeID"]))
                if trade is None or trade["remaining_units"] == 0:
                    raise EvidenceError("daily-financing-trade-missing-or-closed")
                trade["financing"] += number(leg["financing"])
            continue
        if kind != "ORDER_FILL":
            continue
        legs = []
        if tx.get("tradeOpened"):
            legs.append(("open", tx["tradeOpened"]))
        legs.extend(("close", leg) for leg in tx.get("tradesClosed", []))
        if tx.get("tradeReduced"):
            legs.append(("reduce", tx["tradeReduced"]))
        if len({str(leg["tradeID"]) for _, leg in legs}) != len(legs):
            raise EvidenceError("duplicate-trade-in-fill")
        commissions = _allocate(parts["commission_cost"], legs)
        for source, expected_amount in (("realizedPL", parts["realized_pl"]),
                                       ("financing", parts["financing"]),
                                       ("guaranteedExecutionFee", parts["guaranteed_fee_cost"])):
            if sum((number(leg.get(source, "0")) for _, leg in legs), ZERO) != expected_amount:
                raise EvidenceError("fill-cost-breakdown-mismatch")
        fill_units = number(tx["units"])
        if fill_units == 0 or sum((abs(number(leg["units"])) for _, leg in legs), ZERO) != abs(fill_units):
            raise EvidenceError("fill-unit-breakdown-mismatch")
        quality_payload = dict(tx)
        quality_payload.setdefault("fullVWAP", tx.get("price"))
        quality = summarize_fill_execution({"orderFillTransaction": quality_payload},
                                          side="BUY" if fill_units > 0 else "SELL",
                                          instrument=tx["instrument"])
        fills.append({"transaction_id": tx["id"], "instrument": tx["instrument"],
                      "time": tx["time"], "reason": tx.get("reason"),
                      "commission_ccy": str(parts["commission_cost"]),
                      "trade_count": len(legs), "execution_quality": quality})
        for (action, leg), commission in zip(legs, commissions):
            trade_id = str(leg["tradeID"])
            units = abs(number(leg["units"]))
            if action == "open":
                if trade_id in trades:
                    raise EvidenceError("duplicate-trade-open")
                order = by_id.get(str(tx.get("orderID")), {})
                trades[trade_id] = {
                    "trade_id": trade_id, "instrument": tx["instrument"],
                    "side": "BUY" if number(leg["units"]) > 0 else "SELL",
                    "entry_timestamp_utc": tx["time"], "entry_price": leg.get("price", tx.get("price")),
                    "entry_order_id": tx.get("orderID"), "entry_transaction_id": tx["id"],
                    "initial_units": units, "remaining_units": units,
                    "close_timestamp_utc": None, "exit_price": None, "exit_reason": None,
                    "partial_reductions": 0, "commission_allocation": "BROKER_DIRECT",
                    "entry_execution_quality": quality,
                    "requested_stop_on_fill": order.get("stopLossOnFill"),
                    "requested_target_on_fill": order.get("takeProfitOnFill"),
                    **dict.fromkeys(MONEY_FIELDS, ZERO),
                }
            trade = trades.get(trade_id)
            if trade is None or trade["instrument"] != tx["instrument"]:
                raise EvidenceError("missing-open-or-instrument-mismatch")
            if action != "open":
                if trade["close_timestamp_utc"] is not None or units > trade["remaining_units"]:
                    raise EvidenceError("duplicate-close-or-over-reduction")
                trade["remaining_units"] -= units
                if (action == "close") != (trade["remaining_units"] == 0):
                    raise EvidenceError("close-unit-mismatch")
                if action == "close":
                    trade.update(close_timestamp_utc=tx["time"], exit_price=leg.get("price", tx.get("price")),
                                 exit_reason=tx.get("reason"))
                else:
                    trade["partial_reductions"] += 1
            trade["gross_pl"] += number(leg.get("realizedPL", "0"))
            trade["financing"] += number(leg.get("financing", "0"))
            trade["guaranteed_fee"] += number(leg.get("guaranteedExecutionFee", "0"))
            trade["spread_cost"] += number(leg.get("halfSpreadCost", "0"))
            trade["commission"] += commission
            if len(legs) > 1 and commission:
                trade["commission_allocation"] = "ALLOCATED_BY_UNITS"
    if not checkpoints or abs(running - number(balance)) > BALANCE_TOLERANCE:
        raise EvidenceError("summary-balance-mismatch")
    journal = {str(row["trade_id"]): row for row in journal_rows}
    rows = []
    for trade_id, trade in trades.items():
        trade["realized_net_to_date"] = trade["gross_pl"] + trade["financing"] - trade["commission"] - trade["guaranteed_fee"]
        trade["closed_net_pl"] = trade["realized_net_to_date"] if trade["close_timestamp_utc"] else None
        trade["protection_order_history"] = protections[trade_id]
        trade["journal_context"] = None
        row = journal.get(trade_id)
        # IDs alone are insufficient: reject an unrelated journal/account collision.
        if (row and row.get("instrument") == trade["instrument"] and row.get("timestamp_utc")
                and abs(time_key(row["timestamp_utc"]) - time_key(trade["entry_timestamp_utc"])) <= Decimal("0.001")):
            trade["journal_context"] = {k: row.get(k) for k in (
                "run_tag", "session_id", "session_mode", "stop_loss_price", "take_profit_price",
                "gating_flags", "indicators_snapshot", "exit_reason")}
        rows.append(trade)
    # All realized charges, including those on still-open trades, reconcile.
    account_net = running - totals["transfers"]
    allocated_net = sum((t["realized_net_to_date"] for t in rows), ZERO)
    if abs(account_net - allocated_net) > Decimal("0.00000001"):
        raise EvidenceError("allocated-cost-total-mismatch")
    closed = [t for t in rows if t["close_timestamp_utc"]]
    closed.sort(key=lambda t: (time_key(t["close_timestamp_utc"]), identifier(t["trade_id"])))
    def metrics(items):
        net = [t["closed_net_pl"] for t in items]
        wins = [v for v in net if v > 0]
        losses = [v for v in net if v < 0]
        return {"trades": len(items), "wins": len(wins), "losses": len(losses), "flat": net.count(ZERO),
                "gross_pl_ccy": str(sum((t["gross_pl"] for t in items), ZERO)),
                "net_pl_ccy": str(sum(net, ZERO)),
                "profit_factor": str(sum(wins, ZERO) / -sum(losses, ZERO)) if losses else None,
                "allocated_commission_trades": sum(t["commission_allocation"] == "ALLOCATED_BY_UNITS" for t in items)}
    instruments = sorted({t["instrument"] for t in rows})
    summary = {
        "schema": SCHEMA, "ledger_status": "VERIFIED", "currency": "AUD",
        "account_fingerprint": hashlib.sha256(account.encode()).hexdigest()[:16],
        "last_transaction_id": str(cursor), "transaction_count": len(ordered), "fill_count": len(fills),
        "first_transaction_utc": ordered[0]["time"], "last_transaction_utc": ordered[-1]["time"],
        "balance_checkpoints_verified": checkpoints, "balance_ccy": str(running),
        "account_net_excluding_transfers_ccy": str(account_net), "allocation_residual_ccy": str(account_net - allocated_net),
        "components": {k: str(v) for k, v in totals.items()}, "closed": metrics(closed),
        "recent_80_closed": metrics(closed[-80:]), "open_trade_count": len(rows) - len(closed),
        "open_trades_realized_charges_and_pl_ccy": str(sum((t["realized_net_to_date"] for t in rows if not t["close_timestamp_utc"]), ZERO)),
        "by_instrument": {i: metrics([t for t in closed if t["instrument"] == i]) for i in instruments},
        "by_run_tag": {tag: metrics([t for t in closed if (t.get("journal_context") or {}).get("run_tag") == tag])
                       for tag in sorted({(t.get("journal_context") or {}).get("run_tag") for t in closed
                                          if isinstance((t.get("journal_context") or {}).get("run_tag"), str)})},
        "execution_by_instrument": {i: aggregate_execution_quality([f for f in fills if f["instrument"] == i]) for i in instruments},
        "journal_context_matches": sum(t["journal_context"] is not None for t in rows),
        "limitations": ["Multi-trade commission is allocated by absolute filled units; trade results and win counts using that allocation are estimates.",
                        "Spread is already in realized P&L; spread_cost is diagnostic only.",
                        "Fill-snapshot price impact is not decision-to-fill slippage.",
                        "Requested stops and targets and historical protection orders are not proof of current protection.",
                        "Missing historical strategy reasons and versions remain unknown. No strategy or risk settings are changed."],
    }
    return summary, rows, fills


def read_context(path):
    """Read original journal only; never migrate, relabel or overwrite its P&L."""
    if not Path(path).exists():
        return []
    try:
        with closing(sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True, timeout=2)) as conn:
            conn.row_factory = sqlite3.Row
            resolutions = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' "
                                       "AND name='journal_entry_resolutions'").fetchone()
            exclude = (" WHERE NOT EXISTS (SELECT 1 FROM journal_entry_resolutions r "
                       "WHERE r.trade_id=trades.trade_id)" if resolutions else "")
            # broker_confirmed verifies an EXIT, not an entry. Open rows leave
            # it NULL. The broker ID/instrument/time join in build_review is
            # the authority for attaching entry context to open or closed trades.
            rows = [dict(r) for r in conn.execute("SELECT * FROM trades" + exclude)]
        for row in rows:
            for key in ("gating_flags", "indicators_snapshot"):
                row[key] = json.loads(row[key]) if row.get(key) else None
        return rows
    except (sqlite3.Error, ValueError):
        return []


def _atomic(path, content):
    fd, temporary = tempfile.mkstemp(prefix=".history-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _csv(rows):
    if not rows:
        return ""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    for row in rows:
        writer.writerow({k: _json(v) if isinstance(v, (dict, list)) else v for k, v in row.items()})
    return buffer.getvalue()


def _seed(account_dir):
    candidates = list(account_dir.glob("history-*/trade_history.sqlite"))
    candidates.sort(key=lambda p: identifier(p.parent.name.removeprefix("history-")), reverse=True)
    if not candidates:
        return []
    with closing(sqlite3.connect(candidates[0].resolve().as_uri() + "?mode=ro", uri=True)) as conn:
        return [json.loads(r[0]) for r in conn.execute("SELECT raw_json FROM broker_transactions")]


def sync_history(data_dir, journal_path, *, transport=None):
    """Capture a fixed broker cursor, fetch only missing IDs, commit all-or-nothing."""
    if os.getenv("MODE", "").lower() != "demo" or os.getenv("OANDA_ENV", "").lower() != "practice":
        raise EvidenceError("practice-demo-mode-required")
    account = os.getenv("OANDA_ACCOUNT_ID") or os.getenv("ACCOUNT_ID")
    token = os.getenv("OANDA_API_KEY") or os.getenv("OANDA_API_TOKEN")
    if not account or not token:
        raise EvidenceError("broker-read-credentials-unavailable")
    fingerprint = hashlib.sha256(account.encode()).hexdigest()[:16]
    account_dir = Path(data_dir) / "memory" / "oanda-practice" / fingerprint
    output = account_dir / "enriched"
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(output, 0o700)
    # Cross-process exclusion also covers --once and a worker running together.
    lock_fd = os.open(output / "sync.lock", os.O_CREAT | os.O_RDWR, 0o600)
    with os.fdopen(lock_fd, "w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise EvidenceError("history-sync-already-running") from None
        db = output / "history.sqlite"
        fd = os.open(db, os.O_CREAT | os.O_RDWR, 0o600)
        os.close(fd)
        os.chmod(db, 0o600)
        with closing(sqlite3.connect(db, timeout=2)) as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS transactions (id INTEGER PRIMARY KEY, raw_json TEXT NOT NULL)")
            conn.execute("CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            conn.execute("CREATE TABLE IF NOT EXISTS trade_review (trade_id TEXT PRIMARY KEY, record_json TEXT NOT NULL)")
            stored = [json.loads(r[0]) for r in conn.execute("SELECT raw_json FROM transactions ORDER BY id")]
            rows = stored or _seed(account_dir)
            previous = max((identifier(t["id"]) for t in rows), default=0)
            with httpx.Client(headers={"Authorization": f"Bearer {token}"}, timeout=5,
                              follow_redirects=False, transport=transport) as client:
                reader = PracticeReader(client, account)
                payload = reader.get(reader.root + "/summary")
                summary = payload.get("account", {})
                cursor = identifier(payload.get("lastTransactionID", summary.get("lastTransactionID")))
                if (summary.get("id") != account or summary.get("currency") != "AUD"
                        or identifier(summary.get("lastTransactionID", cursor)) != cursor):
                    raise EvidenceError("account-identity-currency-or-cursor-mismatch")
                if not previous <= cursor <= previous + MAX_NEW_TRANSACTIONS:
                    raise EvidenceError("cursor-regressed-or-sync-budget-exceeded")
                for lower in range(previous + 1, cursor + 1, 1000):
                    upper = min(lower + 999, cursor)
                    page = reader.get(reader.root + "/transactions/idrange", {"from": str(lower), "to": str(upper)})
                    batch = page.get("transactions")
                    if not isinstance(batch, list) or len(batch) != upper - lower + 1:
                        raise EvidenceError("incomplete-incremental-page")
                    if sorted(identifier(t.get("id")) for t in batch) != list(range(lower, upper + 1)):
                        raise EvidenceError("incremental-page-range-mismatch")
                    rows.extend(batch)
            review, trades, fills = build_review(rows, account=account, cursor=cursor,
                                                  balance=summary.get("balance"), journal_rows=read_context(journal_path))
            if review["open_trade_count"] != int(summary["openTradeCount"]):
                raise EvidenceError("summary-open-trade-count-mismatch")
            now = datetime.now(timezone.utc).isoformat()
            review.update(generated_at_utc=now, sync_interval_seconds=3600, new_transactions=cursor - previous)
            # Store raw evidence, derived rows and cursor in the SAME SQLite transaction.
            with conn:
                conn.executemany("INSERT OR IGNORE INTO transactions VALUES (?, ?)",
                                 [(identifier(t["id"]), _json(t)) for t in rows if not stored or identifier(t["id"]) > previous])
                conn.execute("DELETE FROM trade_review")
                conn.executemany("INSERT INTO trade_review VALUES (?, ?)", [(t["trade_id"], _json(t)) for t in trades])
                conn.execute("INSERT OR REPLACE INTO state VALUES ('review', ?)", (_json(review),))
                conn.execute("INSERT OR REPLACE INTO state VALUES ('last_transaction_id', ?)", (str(cursor),))
            # Materialized views are regenerated after every successful sync, including no-op retries.
            _atomic(output / "trades.csv", _csv(trades))
            _atomic(output / "fills.csv", _csv(fills))
            _atomic(output / "review.json", _json(review))
            _atomic(output / "status.json", _json({"status": "VERIFIED", "last_success_utc": now,
                                                     "last_transaction_id": str(cursor), "schema": SCHEMA}))
            _atomic(account_dir / "enriched-latest.json", _json({"schema": SCHEMA, "review_path": str(output / "review.json"),
                                                               "database_path": str(db), "generated_at_utc": now,
                                                               "last_transaction_id": str(cursor)}))
            return review


def run_once(data_dir, journal_path):
    try:
        review = sync_history(data_dir, journal_path)
        print("[PRACTICE-HISTORY] " + _json({k: review[k] for k in (
            "ledger_status", "last_transaction_id", "transaction_count", "new_transactions",
            "account_net_excluding_transfers_ccy", "closed", "journal_context_matches", "generated_at_utc")}), flush=True)
        # Fixed allowlisted cohort only; arbitrary journal tags never enter logs.
        if "MINI_RUN" in review["by_run_tag"]:
            print("[PRACTICE-HISTORY-COHORT] " + _json({"run_tag": "MINI_RUN",
                  "account_currency": "AUD", "metrics": review["by_run_tag"]["MINI_RUN"],
                  "generated_at_utc": review["generated_at_utc"],
                  "basis": "after-cost broker ledger; not adaptive learner input"}), flush=True)
        return review
    except Exception as exc:
        reason = str(exc) if isinstance(exc, EvidenceError) else type(exc).__name__
        status = {"ledger_status": "UNVERIFIED", "reason": reason,
                  "observed_utc": datetime.now(timezone.utc).isoformat()}
        print("[PRACTICE-HISTORY] " + _json(status), flush=True)
        account = os.getenv("OANDA_ACCOUNT_ID") or os.getenv("ACCOUNT_ID")
        if account and os.getenv("MODE", "").lower() == "demo" and os.getenv("OANDA_ENV", "").lower() == "practice":
            output = Path(data_dir) / "memory" / "oanda-practice" / hashlib.sha256(account.encode()).hexdigest()[:16] / "enriched"
            if output.exists() and reason != "history-sync-already-running":
                try:
                    _atomic(output / "status.json", _json(status))
                except OSError:
                    pass
        return None


def start_history_monitor(data_dir, journal_path):
    if os.getenv("PRACTICE_HISTORY_ENABLED", "true").lower() != "true":
        return None
    if os.getenv("MODE", "").lower() != "demo" or os.getenv("OANDA_ENV", "").lower() != "practice":
        return None
    def monitor():
        while True:
            run_once(data_dir, journal_path)
            threading.Event().wait(3600)
    worker = threading.Thread(target=monitor, name="practice-history-review", daemon=True)
    worker.start()
    return worker


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(os.getenv("DATA_DIR", "/var/data")))
    parser.add_argument("--journal", type=Path)
    args = parser.parse_args()
    raise SystemExit(0 if run_once(args.data_dir, args.journal or args.data_dir / "trade_journal.db") else 1)
