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
        self._close_fills: dict[str, dict[str, object]] = {}

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
        for key in ("currentUnits", "current_units", "units", "initialUnits"):
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

    @staticmethod
    def _parse_datetime(value: object) -> Optional[datetime]:
        if isinstance(value, datetime):
            return value.astimezone(timezone.utc)
        if not isinstance(value, str) or not value.strip():
            return None
        try:
            parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    def _prime_state(self, trade: Dict) -> None:
        trade_id = self._trade_id(trade)
        if not trade_id:
            return
        state = self._state.get(trade_id) or TrailingState()
        if not hasattr(state, "side") or getattr(state, "side", None) is None:
            setattr(state, "side", self._side_from_trade(trade))
        if not hasattr(state, "entry_price") or getattr(state, "entry_price", None) is None:
            setattr(state, "entry_price", self._entry_price_from_trade(trade))
        if not hasattr(state, "instrument") or getattr(state, "instrument", None) is None:
            setattr(state, "instrument", trade.get("instrument"))
        state.open_time = state.open_time or self._open_time_from_trade(trade)
        self._state[trade_id] = state

    def _broker_trade_details(self, trade_id: str) -> Optional[Dict]:
        try:
            if hasattr(self.broker, "trade_details"):
                result = self.broker.trade_details(trade_id)
                return result if isinstance(result, dict) else None
        except Exception:
            return None

        account = getattr(self.broker, "account", None)
        client_factory = getattr(self.broker, "_client", None)
        if not account or not callable(client_factory):
            return None
        try:
            with client_factory() as client:
                response = client.get(f"/v3/accounts/{account}/trades/{trade_id}")
                if response.status_code != 200:
                    return None
                payload = response.json()
                trade = payload.get("trade") if isinstance(payload, dict) else None
                return trade if isinstance(trade, dict) else None
        except Exception:
            return None

    def _closed_trade_fill(self, trade_id: str) -> Optional[dict[str, object]]:
        details = self._broker_trade_details(trade_id)
        if not details:
            return None
        state = str(details.get("state") or "").upper()
        current_units = details.get("currentUnits")
        closed = state == "CLOSED"
        if not closed and current_units is not None:
            try:
                closed = float(current_units) == 0.0
            except (TypeError, ValueError):
                closed = False
        if not closed:
            return None

        pnl = None
        for key in ("realizedPL", "pl"):
            if details.get(key) is None:
                continue
            try:
                pnl = float(details.get(key))
                break
            except (TypeError, ValueError):
                continue

        exit_price = None
        for key in ("averageClosePrice", "closePrice"):
            if details.get(key) is None:
                continue
            try:
                exit_price = float(details.get(key))
                break
            except (TypeError, ValueError):
                continue

        return {
            "pnl": pnl,
            "exit_price": exit_price,
            "closed_at": self._parse_datetime(details.get("closeTime")),
            "reason": "BROKER_CLOSED",
        }

    def process_open_trades(
        self,
        open_trades: List[Dict],
        *,
        now_utc: Optional[datetime] = None,
    ) -> List[str]:
        now_val = now_utc or datetime.now(timezone.utc)
        current_ids: set[str] = set()
        for trade in list(open_trades or []):
            if not isinstance(trade, dict):
                continue
            self._prime_state(trade)
            trade_id = self._trade_id(trade)
            if trade_id:
                current_ids.add(str(trade_id))

        unresolved: dict[str, TrailingState] = {}
        externally_closed: list[str] = []
        for tracked_id, state in list(self._state.items()):
            if tracked_id in current_ids:
                continue
            instrument = str(getattr(state, "instrument", "") or "")
            if not instrument:
                unresolved[tracked_id] = state
                continue
            fill = self._closed_trade_fill(tracked_id)
            if fill is None:
                unresolved[tracked_id] = state
                continue
            self._close_fills[tracked_id] = fill
            self._close_fills[f"instrument:{instrument}"] = fill
            self._reconcile_closed(
                tracked_id,
                instrument,
                open_trades,
                state,
                reason=str(fill.get("reason") or "BROKER_CLOSED"),
                closed_by="broker_confirmed",
                final_profit=float(fill["pnl"]) if fill.get("pnl") is not None else None,
                now_utc=fill.get("closed_at") if isinstance(fill.get("closed_at"), datetime) else now_val,
                spread_pips=self._current_spread(instrument),
            )
            externally_closed.append(tracked_id)
            self._state.pop(tracked_id, None)

        closed = super().process_open_trades(open_trades, now_utc=now_val)
        for tracked_id, state in unresolved.items():
            if tracked_id not in current_ids and tracked_id not in externally_closed:
                self._state.setdefault(tracked_id, state)
        for tracked_id in externally_closed:
            if tracked_id not in closed:
                closed.append(tracked_id)
        return closed

    @staticmethod
    def _extract_close_fill(result: Dict) -> dict[str, object]:
        payload = result.get("response") if isinstance(result, dict) else None
        payload = payload if isinstance(payload, dict) else {}
        pnl_total = 0.0
        found_pnl = False
        exit_price: Optional[float] = None
        closed_at: Optional[datetime] = None
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
            if closed_at is None:
                closed_at = LearningProfitProtection._parse_datetime(value.get("time"))
        return {
            "pnl": pnl_total if found_pnl else None,
            "exit_price": exit_price,
            "closed_at": closed_at,
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
            if isinstance(fill.get("closed_at"), datetime):
                now_val = fill["closed_at"]
                if isinstance(open_time, datetime):
                    duration_seconds = max(0, int((now_val - open_time).total_seconds()))
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
