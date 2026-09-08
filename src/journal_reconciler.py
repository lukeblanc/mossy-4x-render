from __future__ import annotations

import os
import sqlite3
import time
from collections import Counter
from datetime import datetime, timezone
from typing import Dict, List, Optional

from src.learning_profit_protection import LearningProfitProtection


class JournalReconcilerProfitProtection(LearningProfitProtection):
    """Recover broker-closed trades against their exact persistent journal row."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._reconcile_cursor = 0
        self._retry_after: dict[str, float] = {}

    def _unclosed_journal_rows(self) -> list[sqlite3.Row]:
        journal = self._journal
        path = getattr(journal, "path", None)
        if path is None:
            return []
        try:
            conn = sqlite3.connect(path, timeout=2.0)
            conn.row_factory = sqlite3.Row
            try:
                return conn.execute(
                    """
                    SELECT trade_id, timestamp_utc, instrument, side, entry_price
                    FROM trades
                    WHERE exit_timestamp_utc IS NULL
                      AND NOT EXISTS (SELECT 1 FROM journal_entry_resolutions r
                                      WHERE r.trade_id = trades.trade_id)
                    ORDER BY timestamp_utc ASC, trade_id ASC
                    """
                ).fetchall()
            finally:
                conn.close()
        except (OSError, sqlite3.Error) as exc:
            print(f"[JOURNAL][WARN] unable to inspect unclosed trades error={exc}", flush=True)
            return []

    def _candidate_rows(self) -> list[sqlite3.Row]:
        rows = self._unclosed_journal_rows()
        if not rows:
            self._reconcile_cursor = 0
            return []
        try:
            attempt_limit = int(os.getenv("JOURNAL_RECONCILE_ATTEMPTS_PER_CYCLE", "3"))
        except ValueError:
            attempt_limit = 3
        attempt_limit = max(1, min(5, attempt_limit))
        start = self._reconcile_cursor % len(rows)
        ordered = rows[start:] + rows[:start]
        now = time.monotonic()
        selected = []
        for offset, row in enumerate(ordered):
            if self._retry_after.get(str(row["trade_id"]), 0) > now:
                continue
            selected.append(row)
            self._reconcile_cursor = (start + offset + 1) % len(rows)
            if len(selected) >= attempt_limit:
                break
        return selected

    def _repair_impossible_exit_rows(self) -> int:
        """Reopen rows corrupted by the previous instrument-fallback matcher.

        That bug could write an old broker close onto a newer journal entry, making
        the recorded exit timestamp earlier than the entry timestamp. Such an exit
        is impossible and can be repaired safely before exact reconciliation.
        """

        journal = self._journal
        path = getattr(journal, "path", None)
        if path is None:
            return 0
        try:
            conn = sqlite3.connect(path, timeout=2.0)
            conn.row_factory = sqlite3.Row
            try:
                rows = conn.execute(
                    """
                    SELECT trade_id, exit_timestamp_utc, instrument,
                           realized_pnl_ccy, exit_reason
                    FROM trades
                    WHERE exit_timestamp_utc IS NOT NULL
                      AND timestamp_utc IS NOT NULL
                      AND exit_reason = 'BROKER_CLOSED'
                      AND broker_confirmed = 1
                      AND julianday(exit_timestamp_utc) < julianday(timestamp_utc)
                    """
                ).fetchall()
                for row in rows:
                    conn.execute(
                        """
                        UPDATE trades
                        SET exit_timestamp_utc = NULL,
                            exit_price = NULL,
                            spread_at_exit = NULL,
                            max_profit_ccy = NULL,
                            realized_pnl_ccy = NULL,
                            exit_reason = NULL,
                            duration_seconds = NULL,
                            broker_confirmed = NULL
                        WHERE trade_id = ?
                        """,
                        (str(row["trade_id"]),),
                    )
                    conn.execute(
                        """
                        DELETE FROM trade_events
                        WHERE id = (
                            SELECT id
                            FROM trade_events
                            WHERE timestamp = ?
                              AND instrument = ?
                              AND profit IS ?
                              AND reason = ?
                            ORDER BY id DESC
                            LIMIT 1
                        )
                        """,
                        (
                            row["exit_timestamp_utc"],
                            row["instrument"],
                            row["realized_pnl_ccy"],
                            row["exit_reason"],
                        ),
                    )
                    print(
                        f"[JOURNAL][REPAIR] reopened_misbound_exit ticket={row['trade_id']} "
                        f"instrument={row['instrument']}",
                        flush=True,
                    )
                conn.commit()
                return len(rows)
            finally:
                conn.close()
        except (OSError, sqlite3.Error) as exc:
            print(f"[JOURNAL][WARN] impossible-exit repair failed error={exc}", flush=True)
            return 0

    @staticmethod
    def _extract_trade_id(payload: object) -> Optional[str]:
        if not isinstance(payload, dict):
            return None
        value = payload.get("tradeOpenedID")
        if value is not None:
            return str(value)
        trade_opened = payload.get("tradeOpened")
        if isinstance(trade_opened, dict):
            value = trade_opened.get("tradeID")
            if value is not None:
                return str(value)
        return None

    def _resolve_trade_id_from_order(self, journal_trade_id: str) -> Optional[str]:
        account = getattr(self.broker, "account", None)
        client_factory = getattr(self.broker, "_client", None)
        if not account or not callable(client_factory):
            return None
        try:
            with client_factory() as client:
                response = client.get(f"/v3/accounts/{account}/orders/{journal_trade_id}")
                if response.status_code != 200:
                    print(f"[JOURNAL][LOOKUP] order_id={journal_trade_id} status={response.status_code}", flush=True)
                if response.status_code == 200:
                    payload = response.json()
                    order = payload.get("order") if isinstance(payload, dict) else None
                    if isinstance(order, dict):
                        trade_id = self._extract_trade_id(order)
                        if trade_id:
                            return trade_id
                        fill_id = order.get("fillingTransactionID")
                        if fill_id is not None:
                            fill_response = client.get(
                                f"/v3/accounts/{account}/transactions/{fill_id}"
                            )
                            if fill_response.status_code == 200:
                                fill_payload = fill_response.json()
                                transaction = (
                                    fill_payload.get("transaction")
                                    if isinstance(fill_payload, dict)
                                    else None
                                )
                                trade_id = self._extract_trade_id(transaction)
                                if trade_id:
                                    return trade_id
                        print(f"[JOURNAL][LOOKUP] order_id={journal_trade_id} "
                              f"state={order.get('state', 'UNKNOWN')} no_trade_opened=True", flush=True)
        except Exception as exc:
            print(
                f"[JOURNAL][WARN] order-to-trade lookup failed order_id={journal_trade_id} "
                f"error={exc}",
                flush=True,
            )
        return None

    def _details_for_journal_id(self, journal_trade_id: str) -> tuple[str, Optional[Dict]]:
        if not journal_trade_id.isascii() or not journal_trade_id.isdigit():
            return journal_trade_id, None
        direct = self._broker_trade_details(journal_trade_id)
        if direct:
            return journal_trade_id, direct
        broker_trade_id = self._resolve_trade_id_from_order(journal_trade_id)
        if not broker_trade_id:
            return journal_trade_id, None
        return broker_trade_id, self._broker_trade_details(broker_trade_id)

    def _record_exact_fast_close(
        self,
        row: sqlite3.Row,
        broker_trade_id: str,
        fill: dict[str, object],
        *,
        now_utc: datetime,
    ) -> bool:
        journal = self._journal
        if journal is None:
            return False
        journal_trade_id = str(row["trade_id"] or "")
        instrument = str(row["instrument"] or "")
        closed_at = fill.get("closed_at")
        if not isinstance(closed_at, datetime):
            return False
        exit_ts = closed_at
        opened_at = self._parse_datetime(row["timestamp_utc"])
        if opened_at is None or exit_ts < opened_at:
            print(f"[JOURNAL][DEFER] invalid entry/exit chronology ticket={journal_trade_id}", flush=True)
            return False
        if broker_trade_id != journal_trade_id:
            try:
                with sqlite3.connect(journal.path, timeout=2.0) as conn:
                    canonical = conn.execute("SELECT 1 FROM trades WHERE trade_id = ?",
                                             (broker_trade_id,)).fetchone()
            except sqlite3.Error:
                return False
            if canonical:
                print(f"[JOURNAL][DEFER] legacy_ticket={journal_trade_id} duplicates "
                      f"broker_ticket={broker_trade_id}; preserved for audit.", flush=True)
                return False
        duration_seconds = 0
        if isinstance(opened_at, datetime):
            duration_seconds = max(0, int((exit_ts - opened_at).total_seconds()))
        pnl = float(fill["pnl"]) if fill.get("pnl") is not None else None
        exit_price = float(fill["exit_price"]) if fill.get("exit_price") is not None else None
        reason = str(fill.get("reason") or "BROKER_CLOSED")
        try:
            journal.record_exit(
                trade_id=journal_trade_id,
                exit_timestamp_utc=exit_ts,
                exit_price=exit_price,
                spread_at_exit=None,  # Today's spread is not the historical exit spread.
                max_profit_ccy=None,
                realized_pnl_ccy=pnl,
                exit_reason=reason,
                duration_seconds=duration_seconds,
                broker_confirmed=True,
                run_tag=None,
                instrument=instrument,
                direction=row["side"],
                entry_price=row["entry_price"],
                equity_after=None,  # Historical account balance is unknown here.
                broker_evidence=fill.get("broker_evidence"),
            )
        except Exception as exc:
            print(
                f"[JOURNAL][WARN] exact fast-close save failed ticket={journal_trade_id} "
                f"broker_ticket={broker_trade_id} error={exc}",
                flush=True,
            )
            return False
        print(
            f"[TRADE_CLOSED] ticket={journal_trade_id} broker_ticket={broker_trade_id} "
            f"instrument={instrument} pnl={float(pnl or 0.0):.2f} "
            f"reason={reason} duration_sec={duration_seconds}",
            flush=True,
        )
        print(
            f"[JOURNAL][FAST-CLOSE] journal_ticket={journal_trade_id} "
            f"broker_ticket={broker_trade_id} instrument={instrument}",
            flush=True,
        )
        return True

    def _reconcile_untracked_journal_rows(
        self,
        open_trades: List[Dict],
        *,
        now_utc: datetime,
    ) -> list[str]:
        current_ids = {
            str(trade_id)
            for trade in open_trades or []
            if isinstance(trade, dict)
            for trade_id in [self._trade_id(trade)]
            if trade_id
        }
        recovered: list[str] = []
        outcomes: Counter = Counter()
        for row in self._candidate_rows():
            journal_trade_id = str(row["trade_id"] or "")
            instrument = str(row["instrument"] or "")
            if not journal_trade_id or not instrument:
                continue
            if journal_trade_id in current_ids or journal_trade_id in self._state:
                continue
            # Repeated failures must not hold up every trading decision.
            self._retry_after[journal_trade_id] = time.monotonic() + 900.0
            outcomes["attempted"] += 1
            broker_trade_id, details = self._details_for_journal_id(journal_trade_id)
            if broker_trade_id in current_ids or broker_trade_id in self._state:
                continue
            if not details:
                outcomes["unresolved_id"] += 1
                continue
            if str(details.get("id")) != broker_trade_id or details.get("instrument") != instrument:
                outcomes["identity_mismatch"] += 1
                continue
            fill = self._closed_fill_from_details(details)
            if fill is None:
                outcomes["unconfirmed_close"] += 1
                continue
            if self._record_exact_fast_close(
                row,
                broker_trade_id,
                fill,
                now_utc=now_utc,
            ):
                recovered.append(broker_trade_id)
                self._retry_after.pop(journal_trade_id, None)
                outcomes["recovered"] += 1
            else:
                outcomes["save_or_chronology_rejected"] += 1
        if outcomes:
            print("[JOURNAL][RECONCILE] " + " ".join(f"{key}={value}" for key, value in sorted(outcomes.items()))
                  + " retry_seconds=900 max_attempts=5", flush=True)
        return recovered

    def process_open_trades(
        self,
        open_trades: List[Dict],
        *,
        now_utc: Optional[datetime] = None,
    ) -> List[str]:
        now_val = now_utc or datetime.now(timezone.utc)
        self._repair_impossible_exit_rows()
        closed = super().process_open_trades(open_trades, now_utc=now_val)
        for trade_id in self._reconcile_untracked_journal_rows(
            open_trades,
            now_utc=now_val,
        ):
            if trade_id not in closed:
                closed.append(trade_id)
        return closed


__all__ = ["JournalReconcilerProfitProtection"]
