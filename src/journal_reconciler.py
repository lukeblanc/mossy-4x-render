from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Dict, List, Optional

from src.learning_profit_protection import LearningProfitProtection
from src.profit_protection import TrailingState


class JournalReconcilerProfitProtection(LearningProfitProtection):
    """Recover closed broker trades that never reached in-memory tracking.

    A market order can hit its broker-side stop-loss or take-profit before the
    next decision cycle sees it in ``openTrades``. In that case the normal
    trailing state is never primed. This reconciler treats the persistent
    journal as the source of outstanding entries, resolves legacy OANDA order
    transaction IDs to trade IDs when necessary, and only records an exit after
    OANDA explicitly reports the trade as closed.
    """

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
        current_instruments = {
            str(trade.get("instrument") or "")
            for trade in open_trades or []
            if isinstance(trade, dict) and trade.get("instrument")
        }
        recovered: list[str] = []

        for row in self._unclosed_journal_rows():
            journal_trade_id = str(row["trade_id"] or "")
            instrument = str(row["instrument"] or "")
            if not journal_trade_id or not instrument:
                continue
            if journal_trade_id in current_ids or instrument in current_instruments:
                continue
            if journal_trade_id in self._state:
                continue

            broker_trade_id, details = self._details_for_journal_id(journal_trade_id)
            fill = self._closed_fill_from_details(details)
            if fill is None:
                continue

            state = TrailingState()
            setattr(state, "instrument", instrument)
            setattr(state, "side", row["side"])
            setattr(state, "entry_price", row["entry_price"])
            state.open_time = self._parse_datetime(row["timestamp_utc"])

            self._close_fills[broker_trade_id] = fill
            self._close_fills[f"instrument:{instrument}"] = fill
            closed_at = fill.get("closed_at")
            self._reconcile_closed(
                broker_trade_id,
                instrument,
                open_trades,
                state,
                reason=str(fill.get("reason") or "BROKER_CLOSED"),
                closed_by="broker_confirmed",
                final_profit=float(fill["pnl"]) if fill.get("pnl") is not None else None,
                now_utc=closed_at if isinstance(closed_at, datetime) else now_utc,
                spread_pips=self._current_spread(instrument),
            )
            recovered.append(broker_trade_id)
            print(
                f"[JOURNAL][FAST-CLOSE] journal_ticket={journal_trade_id} "
                f"broker_ticket={broker_trade_id} instrument={instrument}",
                flush=True,
            )
        return recovered

    def process_open_trades(
        self,
        open_trades: List[Dict],
        *,
        now_utc: Optional[datetime] = None,
    ) -> List[str]:
        now_val = now_utc or datetime.now(timezone.utc)
        closed = super().process_open_trades(open_trades, now_utc=now_val)
        for trade_id in self._reconcile_untracked_journal_rows(
            open_trades,
            now_utc=now_val,
        ):
            if trade_id not in closed:
                closed.append(trade_id)
        return closed


__all__ = ["JournalReconcilerProfitProtection"]
