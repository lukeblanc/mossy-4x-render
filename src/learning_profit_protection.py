from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Dict, List, Optional

from src.profit_protection import ProfitProtection, TrailingState


class LearningProfitProtection(ProfitProtection):
    """ProfitProtection with reliable journal reconciliation.

    OANDA can expose an order transaction id at entry and a different trade id
    in openTrades. This class links a broker close back to the latest matching
    open journal row, captures broker-reported realised P/L when available, and
    avoids relying on undeclared TrailingState attributes.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._close_fills: dict[str, dict[str, Optional[float]]] = {}

    @staticmethod
    def _entry_price_from_trade(trade: Dict) -> Optional[float]:
        for key in ("price", "entry_price", "openPrice", "open_price", "averagePrice"):
            value = trade.get(key)
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return None

    @staticmethod
    def _side_from_trade(trade: Dict) -> Optional[str]:
        raw = trade.get("side") or trade.get("direction")
        if raw:
            label = str(raw).upper()
            if label in {"BUY", "LONG"}:
                return "BUY"
            if label in {"SELL", "SHORT"}:
                return "SELL"
        for key in ("currentUnits", "current_units", "units"):
            if key not in trade:
                continue
            try:
                units = float(trade.get(key))
            except (TypeError, ValueError):
                continue
            if units > 0:
                return "BUY"
            if units < 0:
                return "SELL"
        return None

    def _prime_state(self, trade: Dict) -> None:
        trade_id = self._trade_id(trade)
        if not trade_id:
            return
        state = self._state.get(trade_id) or TrailingState()
        if not hasattr(state, "side") or getattr(state, "side", None) is None:
            setattr(state, "side", self._side_from_trade(trade))
        if not hasattr(state, "entry_price") or getattr(state, "entry_price", None) is None:
            setattr(state, "entry_price", self._entry_price_from_trade(trade))
        self._state[trade_id] = state

    def process_open_trades(
        self,
        open_trades: List[Dict],
        *,
        now_utc: Optional[datetime] = None,
    ) -> List[str]:
        for trade in list(open_trades or []):
            if isinstance(trade, dict):
                self._prime_state(trade)
        return super().process_open_trades(open_trades, now_utc=now_utc)

    @staticmethod
    def _extract_close_fill(result: Dict) -> dict[str, Optional[float]]:
        payload = result.get("response") if isinstance(result, dict) else None
        payload = payload if isinstance(payload, dict) else {}
        pnl_total = 0.0
        found_pnl = False
        exit_price: Optional[float] = None
        for key, value in payload.items():
            if not isinstance(value, dict) or "FillTransaction" not in key:
                continue
            pnl_raw = value.get("pl")
            if pnl_raw is not None:
                try:
                    pnl_total += float(pnl_raw)
                    found_pnl = True
                except (TypeError, ValueError):
                    pass
            if exit_price is None and value.get("price") is not None:
                try:
                    exit_price = float(value.get("price"))
                except (TypeError, ValueError):
                    exit_price = None
        return {
            "pnl": pnl_total if found_pnl else None,
            "exit_price": exit_price,
        }

    def _execute_closeout(
        self,
        trade_id: str,
        instrument: str,
        long_units: float,
        short_units: float,
    ) -> Dict:
        result = super()._execute_closeout(trade_id, instrument, long_units, short_units)
        fill = self._extract_close_fill(result)
        if fill.get("pnl") is not None or fill.get("exit_price") is not None:
            self._close_fills[str(trade_id)] = fill
            self._close_fills[f"instrument:{instrument}"] = fill
        return result

    def _resolve_journal_trade_id(self, broker_trade_id: str, instrument: str) -> str:
        journal = self._journal
        path = getattr(journal, "path", None)
        if path is None:
            return broker_trade_id
        try:
            conn = sqlite3.connect(path, timeout=2.0)
            try:
                row = conn.execute(
                    "SELECT trade_id FROM trades WHERE trade_id = ? LIMIT 1",
                    (str(broker_trade_id),),
                ).fetchone()
                if row:
                    return str(row[0])
                row = conn.execute(
                    """
                    SELECT trade_id
                    FROM trades
                    WHERE instrument = ? AND exit_timestamp_utc IS NULL
                    ORDER BY timestamp_utc DESC
                    LIMIT 1
                    """,
                    (instrument,),
                ).fetchone()
                if row:
                    print(
                        f"[JOURNAL][RECONCILE] broker_trade_id={broker_trade_id} "
                        f"journal_trade_id={row[0]} instrument={instrument}",
                        flush=True,
                    )
                    return str(row[0])
            finally:
                conn.close()
        except (OSError, sqlite3.Error):
            pass
        return broker_trade_id

    def _reconcile_closed(
        self,
        trade_id: Optional[str],
        instrument: str,
        open_trades: Optional[List[Dict]],
        state: Optional[TrailingState],
        *,
        reason: Optional[str],
        closed_by: str,
        final_profit: Optional[float],
        now_utc: Optional[datetime] = None,
        spread_pips: Optional[float] = None,
    ) -> None:
        now_val = now_utc or datetime.now(timezone.utc)
        state = state or self._state.get(str(trade_id or "")) or TrailingState()
        side = getattr(state, "side", None)
        entry_price = getattr(state, "entry_price", None)
        max_profit = getattr(state, "max_profit_ccy", None)
        open_time = getattr(state, "open_time", None)
        duration_seconds = 0
        if isinstance(open_time, datetime):
            duration_seconds = max(0, int((now_val - open_time).total_seconds()))

        fill = self._close_fills.pop(str(trade_id or ""), None)
        if fill is None:
            fill = self._close_fills.pop(f"instrument:{instrument}", None)
        resolved_pnl = final_profit
        exit_price = None
        if fill:
            if fill.get("pnl") is not None:
                resolved_pnl = float(fill["pnl"])
            if fill.get("exit_price") is not None:
                exit_price = float(fill["exit_price"])
        if resolved_pnl is None:
            last_profit = getattr(state, "last_profit_ccy", None)
            resolved_pnl = float(last_profit) if last_profit is not None else None

        journal = self._journal
        self._journal = None
        try:
            super()._reconcile_closed(
                trade_id,
                instrument,
                open_trades,
                state,
                reason=reason,
                closed_by=closed_by,
                final_profit=resolved_pnl,
                now_utc=now_val,
                spread_pips=spread_pips,
            )
        finally:
            self._journal = journal

        if journal is None:
            return
        journal_trade_id = self._resolve_journal_trade_id(str(trade_id or instrument), instrument)
        try:
            equity_after = None
            try:
                equity_after = float(self.broker.account_equity())
            except Exception:
                pass
            journal.record_exit(
                trade_id=journal_trade_id,
                exit_timestamp_utc=now_val,
                exit_price=exit_price,
                spread_at_exit=spread_pips,
                max_profit_ccy=max_profit,
                realized_pnl_ccy=resolved_pnl,
                exit_reason=self._reason_label(reason),
                duration_seconds=duration_seconds,
                broker_confirmed=closed_by in {"broker_confirmed", "broker_missing_position"},
                run_tag=None,
                instrument=instrument,
                direction=side,
                entry_price=entry_price,
                equity_after=equity_after,
            )
            print(
                f"[TRADE_CLOSED] ticket={journal_trade_id} broker_ticket={trade_id or 'n/a'} "
                f"instrument={instrument} pnl={float(resolved_pnl or 0.0):.2f} "
                f"reason={self._reason_label(reason)} duration_sec={duration_seconds}",
                flush=True,
            )
        except Exception as exc:
            print(
                f"[JOURNAL][WARN] reliable record_exit failed ticket={journal_trade_id} "
                f"instrument={instrument} error={exc}",
                flush=True,
            )


__all__ = ["LearningProfitProtection"]
