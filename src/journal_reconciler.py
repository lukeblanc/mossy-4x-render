from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from typing import Dict, List, Optional

from src.learning_profit_protection import LearningProfitProtection


class JournalReconcilerProfitProtection(LearningProfitProtection):
    """Recover broker-closed trades against their exact persistent journal row."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._reconcile_cursor = 0

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
                    ORDER BY timestamp_utc ASC
                    LIMIT 500
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
            attempt_limit = int(os.getenv("JOURNAL_RECONCILE_ATTEMPTS_PER_CYCLE", "25"))
        except ValueError:
            attempt_limit = 25
        attempt_limit = max(1, min(100, attempt_limit))
        start = self._reconcile_cursor % len(rows)
        ordered = rows[start:] + rows[:start]
        selected = ordered[:attempt_limit]
        self._reconcile_cursor = (start + len(selected)) % len(rows)
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
        for key in ("tradeID", "tradeId", "tradeOpenedID"):
            value = payload.get(key)
            if value is not None:
                return str(value)
        trade_opened = payload.get("tradeOpened")
        if isinstance(trade_opened, dict):
            for key in ("tradeID", "tradeId", "id"):
                value = trade_opened.get(key)
                if value is not None:
                    return str(value)
        trades_closed = payload.get("tradesClosed")
        if isinstance(trades_closed, list):
            for trade in trades_closed:
                if not isinstance(trade, dict):
                    continue
                for key in ("tradeID", "tradeId", "id"):
                    value = trade.get(key)
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
                since_response = client.get(
                    f"/v3/accounts/{account}/transactions/sinceid",
                    params={"id": journal_trade_id},
                )
                if since_response.status_code != 200:
                    return None
                since_payload = since_response.json()
                transactions = (
                    since_payload.get("transactions", [])
                    if isinstance(since_payload, dict)
                    else []
                )
                for transaction in transactions:
                    if not isinstance(transaction, dict):
                        continue
                    linked_order = transaction.get("orderID") or transaction.get("batchID")
                    if linked_order is None or str(linked_order) != str(journal_trade_id):
                        continue
                    trade_id = self._extract_trade_id(transaction)
                    if trade_id:
                        return trade_id
        except Exception as exc:
            print(
                f"[JOURNAL][WARN] order-to-trade lookup failed order_id={journal_trade_id} "
                f"error={exc}",
                flush=True,
            )
        return None

    def _details_for_journal_id(self, journal_trade_id: str) -> tuple[str, Optional[Dict]]:
        direct = self._broker_trade_details(journal_trade_id)
        if direct:
            return journal_trade_id, direct
        broker_trade_id = self._resolve_trade_id_from_order(journal_trade_id)
        if not broker_trade_id:
            return journal_trade_id, None
        return broker_trade_id, self._broker_trade_details(broker_trade_id)

    @classmethod
    def _closed_fill_from_details(cls, details: Optional[Dict]) -> Optional[dict[str, object]]:
        if not details:
            return None
        state = str(details.get("state") or "").upper()
        closed = state == "CLOSED"
        current_units = details.get("currentUnits")
        if not closed and current_units is not None:
            try:
                closed = float(current_units) == 0.0
            except (TypeError, ValueError):
                closed = False
        if not closed:
            return None

        pnl: Optional[float] = None
        for key in ("realizedPL", "pl"):
            raw = details.get(key)
            if raw is None:
                continue
            try:
                pnl = float(raw)
                break
            except (TypeError, ValueError):
                continue

        exit_price: Optional[float] = None
        for key in ("averageClosePrice", "closePrice"):
            raw = details.get(key)
            if raw is None:
                continue
            try:
                exit_price = float(raw)
                break
            except (TypeError, ValueError):
                continue

        return {
            "pnl": pnl,
            "exit_price": exit_price,
            "closed_at": cls._parse_datetime(details.get("closeTime")),
            "reason": "BROKER_CLOSED",
        }

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
        exit_ts = closed_at if isinstance(closed_at, datetime) else now_utc
        opened_at = self._parse_datetime(row["timestamp_utc"])
        duration_seconds = 0
        if isinstance(opened_at, datetime):
            duration_seconds = max(0, int((exit_ts - opened_at).total_seconds()))
        pnl = float(fill["pnl"]) if fill.get("pnl") is not None else None
        exit_price = float(fill["exit_price"]) if fill.get("exit_price") is not None else None
        spread = self._current_spread(instrument)
        equity_after = None
        try:
            equity_after = float(self.broker.account_equity())
        except Exception:
            pass
        reason = str(fill.get("reason") or "BROKER_CLOSED")
        try:
            journal.record_exit(
                trade_id=journal_trade_id,
                exit_timestamp_utc=exit_ts,
                exit_price=exit_price,
                spread_at_exit=spread,
                max_profit_ccy=None,
                realized_pnl_ccy=pnl,
                exit_reason=reason,
                duration_seconds=duration_seconds,
                broker_confirmed=True,
                run_tag=None,
                instrument=instrument,
                direction=row["side"],
                entry_price=row["entry_price"],
                equity_after=equity_after,
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
        for row in self._candidate_rows():
            journal_trade_id = str(row["trade_id"] or "")
            instrument = str(row["instrument"] or "")
            if not journal_trade_id or not instrument:
                continue
            if journal_trade_id in current_ids or journal_trade_id in self._state:
                continue
            broker_trade_id, details = self._details_for_journal_id(journal_trade_id)
            if broker_trade_id in current_ids or broker_trade_id in self._state:
                continue
            fill = self._closed_fill_from_details(details)
            if fill is None:
                continue
            if self._record_exact_fast_close(
                row,
                broker_trade_id,
                fill,
                now_utc=now_utc,
            ):
                recovered.append(broker_trade_id)
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
