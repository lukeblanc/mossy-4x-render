"""Read-only, same-window OANDA balance and journal reconciliation.

Amounts are in the broker's account currency and summed with Decimal. This module
never imports Broker, places orders, repairs journal rows, or changes risk state.
Schema: https://developer.oanda.com/rest-live-v20/transaction-df/
Pagination: https://developer.oanda.com/rest-live-v20/transaction-ep/
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import time
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import httpx

HOST = "https://api-fxpractice.oanda.com"
TOLERANCE = Decimal("0.01")
MAX_PAGES = 20
PAGE_SIZE = 1000
ZERO = Decimal("0")
# Explicit allowlist: an unfamiliar transaction must not be assumed harmless.
NO_BALANCE_TYPES = frozenset({
    "CREATE", "CLOSE", "REOPEN", "CLIENT_CONFIGURE", "CLIENT_CONFIGURE_REJECT",
    "TRANSFER_FUNDS_REJECT", "MARKET_ORDER", "MARKET_ORDER_REJECT",
    "FIXED_PRICE_ORDER", "LIMIT_ORDER", "LIMIT_ORDER_REJECT", "STOP_ORDER",
    "STOP_ORDER_REJECT", "MARKET_IF_TOUCHED_ORDER", "MARKET_IF_TOUCHED_ORDER_REJECT",
    "TAKE_PROFIT_ORDER", "TAKE_PROFIT_ORDER_REJECT", "STOP_LOSS_ORDER",
    "STOP_LOSS_ORDER_REJECT", "GUARANTEED_STOP_LOSS_ORDER", "GUARANTEED_STOP_LOSS_ORDER_REJECT",
    "TRAILING_STOP_LOSS_ORDER", "TRAILING_STOP_LOSS_ORDER_REJECT", "ORDER_CANCEL",
    "ORDER_CANCEL_REJECT", "ORDER_CLIENT_EXTENSIONS_MODIFY", "ORDER_CLIENT_EXTENSIONS_MODIFY_REJECT",
    "TRADE_CLIENT_EXTENSIONS_MODIFY", "TRADE_CLIENT_EXTENSIONS_MODIFY_REJECT",
    "MARGIN_CALL_ENTER", "MARGIN_CALL_EXTEND", "MARGIN_CALL_EXIT", "DELAYED_TRADE_CLOSURE",
    "RESET_RESETTABLE_PL",
})


class EvidenceError(ValueError):
    """An evidence failure with a safe, non-secret diagnostic code."""


def number(value: object) -> Decimal:
    if value is None or isinstance(value, bool):
        raise EvidenceError("missing-or-invalid-amount")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise EvidenceError("invalid-amount") from None
    if not result.is_finite():
        raise EvidenceError("non-finite-amount")
    return result


def timestamp(value: object) -> datetime:
    try:
        result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        raise EvidenceError("invalid-timestamp") from None
    if result.tzinfo is None:
        raise EvidenceError("timezone-required")
    return result.astimezone(timezone.utc)


def time_key(value: object) -> Decimal:
    """Preserve OANDA nanoseconds for exact reporting-window membership."""
    text = str(value)
    dt = timestamp(value)
    match = re.search(r"[T ]\d{2}:\d{2}:\d{2}(?:\.(\d{1,9}))?(?:Z|[+-]\d{2}:\d{2})$", text)
    if not match:
        raise EvidenceError("unsupported-timestamp-precision")
    fraction = Decimal("0." + (match.group(1) or "0"))
    return Decimal(int(dt.replace(microsecond=0).timestamp())) + fraction


def identifier(value: object) -> int:
    text = str(value)
    if not text.isascii() or not text.isdigit():
        raise EvidenceError("invalid-transaction-id")
    return int(text)


def unavailable(reason: str) -> dict[str, Any]:
    return {"status": "UNVERIFIED", "ledger_status": "UNVERIFIED",
            "journal_status": "UNVERIFIED", "reasons": [reason],
            "unexplained_balance_delta": None, "journal_vs_broker_pl_delta": None,
            "account_net_excluding_transfers": None}


def _components(tx: dict[str, Any]) -> dict[str, Decimal]:
    parts = dict.fromkeys(("realized_pl", "financing", "commission_cost",
                          "guaranteed_fee_cost", "dividend_adjustment", "transfers"), ZERO)
    kind = tx["type"]
    if kind == "ORDER_FILL":
        parts["realized_pl"] = number(tx.get("pl"))
        parts["financing"] = number(tx.get("financing", "0"))
        parts["commission_cost"] = number(tx.get("commission", "0"))
        parts["guaranteed_fee_cost"] = number(tx.get("guaranteedExecutionFee", "0"))
        if parts["commission_cost"] < 0 or parts["guaranteed_fee_cost"] < 0:
            raise EvidenceError("negative-fee-cost")
    elif kind == "DAILY_FINANCING":
        parts["financing"] = number(tx.get("financing"))
    elif kind == "DIVIDEND_ADJUSTMENT":
        parts["dividend_adjustment"] = number(tx.get("dividendAdjustment"))
    elif kind == "TRANSFER_FUNDS":
        parts["transfers"] = number(tx.get("amount"))
    elif kind not in NO_BALANCE_TYPES:
        raise EvidenceError("unsupported-transaction-type")
    elif kind != "TRANSFER_FUNDS_REJECT" and any(
        key in tx and number(tx[key]) != 0 for key in
        ("pl", "financing", "commission", "guaranteedExecutionFee", "dividendAdjustment", "amount")
    ):
        raise EvidenceError("unexpected-money-on-nonmonetary-transaction")
    # halfSpreadCost and nested per-trade financing are informational/breakdowns,
    # not additional balance debits. Do not subtract them a second time.
    return parts


def _net(parts: dict[str, Decimal]) -> Decimal:
    return (parts["realized_pl"] + parts["financing"] + parts["dividend_adjustment"]
            + parts["transfers"] - parts["commission_cost"] - parts["guaranteed_fee_cost"])


def read_journal(path: Path, start: datetime, end: datetime) -> list[dict[str, Any]]:
    """Read original journal rows without creating a missing DB or migrating it."""
    try:
        with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=2) as conn:
            conn.execute("PRAGMA query_only=ON")
            conn.row_factory = sqlite3.Row
            resolved = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' "
                                    "AND name='journal_entry_resolutions'").fetchone()
            exclude = (" AND NOT EXISTS (SELECT 1 FROM journal_entry_resolutions r "
                       "WHERE r.trade_id=trades.trade_id)" if resolved else "")
            rows = conn.execute("SELECT trade_id, instrument, exit_timestamp_utc, realized_pnl_ccy, "
                                "broker_confirmed FROM trades WHERE exit_timestamp_utc IS NOT NULL"
                                + exclude).fetchall()
    except sqlite3.Error:
        raise EvidenceError("journal-unreadable-or-schema-missing") from None
    return [dict(row) for row in rows
            if time_key(start.isoformat()) < time_key(row["exit_timestamp_utc"]) <= time_key(end.isoformat())]


def reconcile(
    *, transactions: list[dict[str, Any]], expected_count: int,
    account_id: str, currency: str, start: datetime, end: datetime,
    journal_rows: list[dict[str, Any]], checkpoint: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Reconcile complete broker history through end; compare only (start, end].

    Opening/closing balances are authoritative accountBalance fields immediately
    before each boundary, not approximate heartbeat equities or fabricated NAVs.
    Closed-trade IDs AND amounts must match; offsetting missing rows cannot pass.
    Partial reductions remain UNVERIFIED pending explicit lifetime allocation.
    """
    try:
        start, end = timestamp(start.isoformat()), timestamp(end.isoformat())
        if start >= end or currency != "AUD":
            raise EvidenceError("invalid-window-or-account-currency")
        if type(expected_count) is not int or expected_count != len(transactions):
            raise EvidenceError("incomplete-transaction-history")
        if len(transactions) > MAX_PAGES * PAGE_SIZE:
            raise EvidenceError("transaction-budget-exceeded")
        ordered = sorted(transactions, key=lambda tx: identifier(tx.get("id")))
        seen: set[int] = set()
        previous_time: Decimal | None = None
        opening: dict[str, Any] | None = None
        closing: dict[str, Any] | None = None
        window: list[dict[str, Any]] = []
        for tx in ordered:
            tid, when = identifier(tx.get("id")), time_key(tx.get("time"))
            if tid in seen or tx.get("accountID") != account_id:
                raise EvidenceError("duplicate-id-or-account-mismatch")
            if not isinstance(tx.get("type"), str) or when > time_key(end.isoformat()):
                raise EvidenceError("invalid-type-or-window-contamination")
            if previous_time is not None and when < previous_time:
                raise EvidenceError("transaction-time-order-mismatch")
            previous_time = when
            seen.add(tid)
            if "accountBalance" in tx:
                number(tx["accountBalance"])
                closing = tx
                if when <= time_key(start.isoformat()):
                    opening = tx
            if when > time_key(start.isoformat()):
                window.append(tx)
        if opening is None or closing is None:
            raise EvidenceError("opening-balance-checkpoint-missing")

        opening_balance, closing_balance = number(opening["accountBalance"]), number(closing["accountBalance"])
        running = opening_balance
        totals = _components({"type": "RESET_RESETTABLE_PL"})
        ledger_reasons: list[str] = []
        journal_reasons: list[str] = []
        expected_closes: dict[str, tuple[Decimal, str, datetime]] = {}
        for tx in window:
            try:
                parts = _components(tx)
                for key, value in parts.items():
                    totals[key] += value
                running += _net(parts)
                if tx["type"] not in NO_BALANCE_TYPES and "accountBalance" not in tx:
                    raise EvidenceError("monetary-transaction-balance-missing")
                if "accountBalance" in tx and abs(number(tx["accountBalance"]) - running) > TOLERANCE:
                    ledger_reasons.append("intermediate-balance-mismatch")
            except EvidenceError as exc:
                ledger_reasons.append(str(exc))
            if tx["type"] == "ORDER_FILL":
                if tx.get("tradeReduced"):
                    journal_reasons.append("partial-close-needs-lifetime-allocation")
                closed = tx.get("tradesClosed", [])
                if not isinstance(closed, list):
                    raise EvidenceError("invalid-trades-closed")
                for trade in closed:
                    key = str(identifier(trade.get("tradeID")))
                    if key in expected_closes:
                        journal_reasons.append("duplicate-broker-close")
                    expected_closes[key] = (number(trade.get("realizedPL")), str(tx.get("instrument")), timestamp(tx["time"]))

        residual = closing_balance - opening_balance - _net(totals)
        if abs(residual) > TOLERANCE:
            ledger_reasons.append("unexplained-balance-movement")
        actual_closes: dict[str, dict[str, Any]] = {}
        journal_total = ZERO
        for row in journal_rows:
            when = timestamp(row.get("exit_timestamp_utc"))
            if not start < when <= end:
                raise EvidenceError("journal-window-mismatch")
            if row.get("broker_confirmed") not in (1, True, "1"):
                journal_reasons.append("unconfirmed-journal-close")
                continue
            key = str(identifier(row.get("trade_id")))
            if key in actual_closes:
                journal_reasons.append("duplicate-journal-close")
                continue
            actual_closes[key] = row
            journal_total += number(row.get("realized_pnl_ccy"))
        missing = sorted(set(expected_closes) - set(actual_closes))
        extra = sorted(set(actual_closes) - set(expected_closes))
        if missing:
            journal_reasons.append("missing-journal-closes")
        if extra:
            journal_reasons.append("journal-closes-absent-from-broker-window")
        for key in set(actual_closes) & set(expected_closes):
            pnl, instrument, when = expected_closes[key]
            row = actual_closes[key]
            if (abs(number(row.get("realized_pnl_ccy")) - pnl) > TOLERANCE
                    or row.get("instrument") != instrument
                    or abs((timestamp(row["exit_timestamp_utc"]) - when).total_seconds()) > 0.001):
                journal_reasons.append("journal-close-details-mismatch")
        journal_delta = journal_total - totals["realized_pl"]
        if abs(journal_delta) > TOLERANCE:
            journal_reasons.append("journal-vs-broker-realized-pl-mismatch")
        # A current summary is useful only at an identical transaction cursor.
        # A later live balance is never substituted for a historical period end.
        if checkpoint and ordered:
            summary_cursor = identifier(checkpoint["last_transaction_id"])
            period_cursor = identifier(ordered[-1]["id"])
            if summary_cursor < period_cursor:
                ledger_reasons.append("summary-cursor-behind-report-period")
            elif summary_cursor == period_cursor and abs(number(checkpoint["balance"]) - closing_balance) > TOLERANCE:
                ledger_reasons.append("current-summary-balance-mismatch")
        reasons = sorted(set(ledger_reasons + journal_reasons))
        return {
            "status": "UNVERIFIED" if reasons else "VERIFIED",
            "ledger_status": "UNVERIFIED" if ledger_reasons else "VERIFIED",
            "journal_status": "UNVERIFIED" if journal_reasons else "VERIFIED",
            "currency": currency, "account_fingerprint": hashlib.sha256(account_id.encode()).hexdigest()[:16],
            "window_start_utc": start.isoformat(), "window_end_utc": end.isoformat(),
            "boundary_convention": "start-exclusive,end-inclusive", "transaction_count": len(window),
            "opening_checkpoint": {"as_of_utc": start.isoformat(), "balance": str(opening_balance),
                                   "balance_transaction_id": str(opening["id"])},
            "closing_checkpoint": {"as_of_utc": end.isoformat(), "balance": str(closing_balance),
                                   "balance_transaction_id": str(closing["id"])},
            "current_checkpoint": checkpoint,
            "components": {key: str(value) for key, value in totals.items()},
            "balance_change": str(closing_balance - opening_balance),
            "account_net_excluding_transfers": str(closing_balance - opening_balance - totals["transfers"]) if not ledger_reasons else None,
            "unexplained_balance_delta": str(residual),
            "journal_trade_pl": str(journal_total), "journal_vs_broker_pl_delta": str(journal_delta),
            "broker_closed_trades": len(expected_closes), "journal_closed_trades": len(actual_closes),
            "missing_journal_close_count": len(missing), "extra_journal_close_count": len(extra),
            "tolerance_ccy": str(TOLERANCE), "reasons": reasons,
        }
    except (EvidenceError, KeyError, TypeError, AttributeError) as exc:
        return unavailable(str(exc) if isinstance(exc, EvidenceError) else "malformed-evidence")


class PracticeReader:
    """GET-only reader; validates page destinations before attaching credentials."""
    def __init__(self, client: httpx.Client, account_id: str):
        if not account_id or any(c not in "0123456789-" for c in account_id):
            raise EvidenceError("invalid-account-id")
        self.client, self.account_id = client, account_id
        self.root = f"/v3/accounts/{account_id}"
        self.deadline = time.monotonic() + 60

    def get(self, path: str, params: dict[str, str] | None = None) -> dict[str, Any]:
        if time.monotonic() >= self.deadline:
            raise EvidenceError("account-audit-time-budget-exceeded")
        response = self.client.get(HOST + path, params=params)
        if response.status_code != 200:
            raise EvidenceError(f"oanda-http-{response.status_code}")
        if len(response.content) > 16_000_000:
            raise EvidenceError("oversize-evidence-response")
        payload = response.json()
        if not isinstance(payload, dict):
            raise EvidenceError("invalid-evidence-response")
        return payload

    def history(self, end: datetime) -> tuple[list[dict[str, Any]], int]:
        index = self.get(self.root + "/transactions", {"to": end.isoformat(), "pageSize": str(PAGE_SIZE)})
        pages, count = index.get("pages"), index.get("count")
        if not isinstance(pages, list) or type(count) is not int or count < 0:
            raise EvidenceError("invalid-transaction-index")
        if count > MAX_PAGES * PAGE_SIZE or len(pages) > MAX_PAGES:
            raise EvidenceError("transaction-budget-exceeded")
        rows: list[dict[str, Any]] = []
        seen_pages: set[tuple[int, int]] = set()
        for page in pages:
            parsed = urlsplit(page)
            query = parse_qs(parsed.query, keep_blank_values=True)
            if (parsed.scheme != "https" or parsed.netloc != "api-fxpractice.oanda.com"
                    or parsed.path != self.root + "/transactions/idrange" or parsed.fragment
                    or set(query) != {"from", "to"} or any(len(v) != 1 for v in query.values())):
                raise EvidenceError("unsafe-pagination-url")
            lower, upper = identifier(query["from"][0]), identifier(query["to"][0])
            if lower > upper or (lower, upper) in seen_pages:
                raise EvidenceError("duplicate-or-invalid-page")
            seen_pages.add((lower, upper))
            payload = self.get(self.root + "/transactions/idrange", {"from": str(lower), "to": str(upper)})
            batch = payload.get("transactions")
            if not isinstance(batch, list) or len(batch) > PAGE_SIZE:
                raise EvidenceError("invalid-transaction-page")
            for row in batch:
                if not isinstance(row, dict) or not lower <= identifier(row.get("id")) <= upper:
                    raise EvidenceError("page-range-mismatch")
            rows.extend(batch)
        if len(rows) != count:
            raise EvidenceError("incomplete-transaction-history")
        return rows, count


def collect_reconciliation(
    db_path: Path | str, start_utc: str, end_utc: str, *, transport=None,
) -> dict[str, Any]:
    """One bounded weekly read on the report thread, never on the decision path."""
    try:
        if os.getenv("MODE", "").lower() != "demo" or os.getenv("OANDA_ENV", "").lower() != "practice":
            raise EvidenceError("practice-demo-mode-required")
        account = os.getenv("OANDA_ACCOUNT_ID") or os.getenv("ACCOUNT_ID")
        token = os.getenv("OANDA_API_KEY") or os.getenv("OANDA_API_TOKEN")
        if not account or not token:
            raise EvidenceError("broker-read-credentials-unavailable")
        start, end = timestamp(start_utc), timestamp(end_utc)
        with httpx.Client(headers={"Authorization": f"Bearer {token}"}, timeout=5,
                          follow_redirects=False, transport=transport) as client:
            reader = PracticeReader(client, account)
            summary_payload = reader.get(reader.root + "/summary")
            summary = summary_payload.get("account", {})
            cursor = summary_payload.get("lastTransactionID", summary.get("lastTransactionID"))
            if summary.get("lastTransactionID") is not None and identifier(summary["lastTransactionID"]) != identifier(cursor):
                raise EvidenceError("summary-cursor-mismatch")
            if summary.get("id") != account or summary.get("currency") != "AUD":
                raise EvidenceError("account-identity-or-currency-mismatch")
            checkpoint = {"observed_utc": datetime.now(timezone.utc).isoformat(),
                          "last_transaction_id": str(identifier(cursor)),
                          "balance": str(number(summary.get("balance"))),
                          "nav": str(number(summary.get("NAV"))),
                          "unrealized_pl": str(number(summary.get("unrealizedPL"))),
                          "open_trade_count": summary.get("openTradeCount")}
            rows, count = reader.history(end)
        return reconcile(transactions=rows, expected_count=count, account_id=account, currency="AUD",
                         start=start, end=end, journal_rows=read_journal(Path(db_path), start, end),
                         checkpoint=checkpoint)
    except EvidenceError as exc:
        return unavailable(str(exc))
    except Exception:
        # Never expose raw request URLs, credentials or broker payloads in logs.
        return unavailable("account-audit-read-failed")
