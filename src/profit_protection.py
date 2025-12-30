"""Profit-protection logic for trailing unrealized gains."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List, Optional, Protocol


ARM_AT_CCY = 1.00
GIVEBACK_CCY = 0.50


@dataclass
class TrailingState:
    max_profit_ccy: Optional[float] = None
    armed: bool = False
    last_update: Optional[datetime] = None
    open_time: Optional[datetime] = None
    close_cooldown_until: Optional[datetime] = None
    missing_retry_attempted: bool = False


class BrokerLike(Protocol):
    def get_unrealized_profit(self, instrument: str) -> Optional[float]:
        ...

    def position_snapshot(self, instrument: str) -> Optional[Dict]:
        ...

    def close_position_side(self, instrument: str, long_units: float, short_units: float) -> Dict:
        ...

    def close_trade(self, trade_id: str, instrument: Optional[str] = None) -> Dict:
        ...

    def current_spread(self, instrument: str) -> Optional[float]:
        ...

    def _pip_size(self, instrument: str) -> float:
        ...


class ProfitProtection:
    """Maintain per-trade high-water marks and apply trailing exits."""

    def __init__(
        self,
        broker: BrokerLike,
        trigger: float = ARM_AT_CCY,
        trail: float = GIVEBACK_CCY,
        *,
        arm_pips: float = 0.0,
        giveback_pips: float = 0.0,
        arm_ccy: Optional[float] = None,
        giveback_ccy: Optional[float] = None,
        arm_usd: Optional[float] = None,
        giveback_usd: Optional[float] = None,
        use_pips: bool = False,
        be_arm_pips: float = 0.0,
        be_offset_pips: float = 0.0,
        min_check_interval_sec: float = 0.0,
        aggressive: bool = False,
        aggressive_max_hold_minutes: float = 45.0,
        aggressive_max_loss_ccy: float = 5.0,
        aggressive_max_loss_atr_mult: float = 1.2,
        time_stop_minutes: float = 90.0,
        time_stop_min_pips: float = 2.0,
        time_stop_xau_atr_mult: float = 0.35,
    ) -> None:
        self.broker = broker
        self.arm_pips = float(arm_pips)
        self.giveback_pips = float(giveback_pips)
        arm_value = arm_ccy if arm_ccy is not None else arm_usd
        giveback_value = giveback_ccy if giveback_ccy is not None else giveback_usd
        self.arm_ccy = float(arm_value if arm_value is not None else trigger)
        self.giveback_ccy = float(giveback_value if giveback_value is not None else trail)
        # Maintain legacy attribute names for compatibility with existing callers/tests.
        self.arm_usd = self.arm_ccy
        self.giveback_usd = self.giveback_ccy
        # Preserve legacy attributes for backwards compatibility/tests
        self.trigger = self.arm_ccy
        self.trail = self.giveback_ccy
        # Pip trailing is disabled; retain flag for compatibility only
        self.use_pips = False
        self.be_arm_pips = float(be_arm_pips)
        self.be_offset_pips = float(be_offset_pips)
        self.min_check_interval_sec = float(min_check_interval_sec)
        self.aggressive = aggressive
        self.aggressive_max_hold_minutes = float(aggressive_max_hold_minutes)
        self.aggressive_max_loss_ccy = float(aggressive_max_loss_ccy)
        self.aggressive_max_loss_atr_mult = float(aggressive_max_loss_atr_mult)
        self.time_stop_minutes = float(time_stop_minutes)
        self.time_stop_min_pips = float(time_stop_min_pips)
        self.time_stop_xau_atr_mult = float(time_stop_xau_atr_mult)
        self._state: Dict[str, TrailingState] = {}
        self._locally_closed: set[str] = set()

    def snapshot(self) -> Dict[str, TrailingState]:
        """Return a shallow copy of the current per-trade trailing state (test helper)."""
        return dict(self._state)

    def process_open_trades(self, open_trades: List[Dict], *, now_utc: Optional[datetime] = None) -> List[str]:
        """Inspect open trades, update their high-water marks, and close on trailing giveback."""

        now_utc = now_utc or datetime.now(timezone.utc)
        active_keys = set()
        closed_trades: List[str] = []
        broker_snapshot = self._list_open_trades_quietly()

        for trade in list(open_trades):
            trade_id = self._trade_id(trade)
            instrument = self._instrument_from_trade(trade)
            units = self._units_from_trade(trade)
            if not trade_id or not instrument or units == 0:
                continue
            if self._is_locally_closed(trade_id, instrument):
                instrument_open = None if broker_snapshot is None else self._instrument_open_in_snapshot(broker_snapshot, instrument, trade_id)
                if instrument_open is False:
                    print(
                        f"[TRAIL][INFO] skipping ticket={trade_id} instrument={instrument} reason=locally_closed_and_absent",
                        flush=True,
                    )
                    continue
                if instrument_open:
                    print(
                        f"[TRAIL][INFO] broker_reports_open ticket={trade_id} instrument={instrument} clearing_local_closed_marker",
                        flush=True,
                    )
                    self._unmark_locally_closed(trade_id, instrument)

            instrument_open = None if broker_snapshot is None else self._instrument_open_in_snapshot(broker_snapshot, instrument, trade_id)
            if instrument_open is False:
                print(
                    f"[TRAIL][INFO] reconcile_missing_position ticket={trade_id} instrument={instrument} action=mark_closed",
                    flush=True,
                )
                self._reconcile_closed(trade_id, instrument, open_trades, self._state.get(trade_id))
                if trade_id:
                    closed_trades.append(trade_id)
                continue

            active_keys.add(trade_id)
            state = self._state.get(trade_id, TrailingState())
            state.open_time = state.open_time or self._open_time_from_trade(trade)

            spread_pips = self._current_spread(instrument)
            profit = self._profit_from_trade(trade, instrument)
            pips = self._pips_from_trade(trade, instrument, profit, units)
            atr_val = self._atr_for_trade(trade)
            minutes_open = self._minutes_open(trade, now_utc, state.open_time)

            if self._maybe_time_stop(
                trade_id,
                instrument,
                pips,
                minutes_open,
                atr_val,
                spread_pips,
                open_trades=open_trades,
                state=state,
                units=units,
                now_utc=now_utc,
            ):
                closed_trades.append(trade_id)
                self._state.pop(trade_id, None)
                continue

            self._update_peak_profit(trade_id, state, profit)
            state.last_update = now_utc
            self._state[trade_id] = state

            if self.aggressive and self._maybe_aggressive_exit(
                trade, trade_id, instrument, profit, now_utc, open_trades=open_trades, state=state, units=units
            ):
                closed_trades.append(trade_id)
                self._state.pop(trade_id, None)
                continue

            if profit is None or state.max_profit_ccy is None:
                continue

            trailing_floor = state.max_profit_ccy - self.giveback_ccy if state.max_profit_ccy is not None else None
            if state.max_profit_ccy is not None and profit is not None:
                print(
                    f"[TRAIL][DEBUG] profit={profit:.2f} high={state.max_profit_ccy:.2f} "
                    f"floor={trailing_floor:.2f} armed={state.armed}",
                    flush=True,
                )

            if not state.armed and state.max_profit_ccy is not None and state.max_profit_ccy >= self.arm_ccy:
                state.armed = True
                print(f"[TRAIL][INFO] armed ticket={trade_id} profit={profit:.2f}", flush=True)

            if not state.armed or trailing_floor is None:
                continue

            if profit is not None and profit <= trailing_floor:
                print(
                    f"[TRAIL][INFO] giveback_met ticket={trade_id} instrument={instrument} "
                    f"profit={profit:.2f} floor={trailing_floor:.2f} high_water={state.max_profit_ccy:.2f} giveback={self.giveback_ccy:.2f}",
                    flush=True,
                )
                broker_snapshot = self._list_open_trades_quietly()
                instrument_open = None if broker_snapshot is None else self._instrument_open_in_snapshot(broker_snapshot, instrument, trade_id)
                print(
                    f"[TRAIL][INFO] attempting_close ticket={trade_id} instrument={instrument} reason=pnl_profit_protection"
                    + ("" if instrument_open is None else f" snapshot_open={instrument_open}"),
                    flush=True,
                )
                if self._close_trade(
                    trade_id,
                    instrument,
                    profit,
                    None,
                    trailing_floor,
                    state.max_profit_ccy,
                    spread_pips,
                    reason="pnl_profit_protection",
                    open_trades=open_trades,
                    state=state,
                    units=units,
                    now_utc=now_utc,
                ):
                    closed_trades.append(trade_id)
                    self._state.pop(trade_id, None)

        self._cleanup_stale(active_keys)
        return closed_trades

    def _cleanup_stale(self, active_keys: Iterable[str]) -> None:
        active = set(active_keys)
        stale = [key for key in self._state if key not in active]
        for key in stale:
            self._state.pop(key, None)

    @staticmethod
    def _closed_key(trade_id: Optional[str], instrument: Optional[str]) -> Optional[str]:
        if trade_id:
            return str(trade_id)
        if instrument:
            return f"instrument:{instrument}"
        return None

    def _is_locally_closed(self, trade_id: Optional[str], instrument: Optional[str]) -> bool:
        key = self._closed_key(trade_id, instrument)
        return key is not None and key in self._locally_closed

    def _mark_locally_closed(self, trade_id: Optional[str], instrument: Optional[str]) -> None:
        key = self._closed_key(trade_id, instrument)
        if key is not None:
            self._locally_closed.add(key)

    def _unmark_locally_closed(self, trade_id: Optional[str], instrument: Optional[str]) -> None:
        key = self._closed_key(trade_id, instrument)
        if key is not None:
            self._locally_closed.discard(key)

    @staticmethod
    def _trade_id(trade: Dict) -> Optional[str]:
        if not isinstance(trade, dict):
            return None
        for key in ("id", "tradeID", "ticket", "position_id"):
            value = trade.get(key)
            if value is not None:
                return str(value)
        instrument = trade.get("instrument")
        open_time = trade.get("openTime") or trade.get("open_time")
        if instrument and open_time:
            return f"{instrument}:{open_time}"
        return instrument

    @staticmethod
    def _instrument_from_trade(trade: Dict) -> Optional[str]:
        instrument = trade.get("instrument")
        if not instrument:
            return None
        units = trade.get("currentUnits") or trade.get("current_units")
        if units is not None:
            try:
                if float(units) == 0.0:
                    return None
            except (TypeError, ValueError):
                pass
        return instrument

    @staticmethod
    def _units_from_trade(trade: Dict) -> float:
        units = trade.get("currentUnits") or trade.get("current_units") or trade.get("units")
        try:
            return float(units)
        except (TypeError, ValueError):
            return 0.0

    def _profit_from_trade(self, trade: Dict, instrument: str) -> Optional[float]:
        """Always source unrealized PnL from the broker (account currency)."""
        try:
            return self.broker.get_unrealized_profit(instrument)
        except AttributeError:
            return None
        except Exception as exc:  # pragma: no cover - defensive logging
            print(
                f"[TRAIL] Failed to read unrealized P/L for {instrument}: {exc}",
                flush=True,
            )
            return None

    def _pips_from_trade(
        self,
        trade: Dict,
        instrument: str,
        profit: Optional[float],
        units: float,
    ) -> Optional[float]:
        if isinstance(trade, dict):
            for key in ("unrealizedPips", "unrealized_pips", "pips"):
                value = trade.get(key)
                if value is not None:
                    try:
                        return float(value)
                    except (TypeError, ValueError):
                        break

        pip_size = self._pip_size(instrument)
        if pip_size <= 0 or profit is None or units == 0:
            return None
        try:
            price_diff = profit / units
            pips = (price_diff / pip_size) * (1 if units > 0 else -1)
            return float(pips)
        except (TypeError, ValueError):
            return None

    def _update_peak_profit(
        self,
        trade_id: str,
        state: TrailingState,
        profit: Optional[float],
    ) -> None:
        if profit is None:
            return
        if state.max_profit_ccy is None or profit > state.max_profit_ccy:
            state.max_profit_ccy = profit
            print(f"[TRAIL][INFO] update ticket={trade_id} max_profit_ccy={profit:.2f}", flush=True)

    def _time_stop_threshold(self, instrument: str, atr_value: Optional[float]) -> float:
        base_threshold = max(0.0, self.time_stop_min_pips)
        if instrument == "XAU_USD" and atr_value is not None and atr_value > 0:
            pip_size = self._pip_size(instrument)
            if pip_size > 0:
                atr_pips = atr_value / pip_size
                return max(base_threshold, atr_pips * self.time_stop_xau_atr_mult)
        return base_threshold

    def _interval_blocked(self, state: TrailingState, now_utc: datetime) -> bool:
        if self.min_check_interval_sec <= 0:
            return False
        if state.last_update is None:
            return False
        delta = now_utc - state.last_update
        return delta.total_seconds() < self.min_check_interval_sec

    def _maybe_time_stop(
        self,
        trade_id: str,
        instrument: str,
        pips: Optional[float],
        minutes_open: Optional[float],
        atr_value: Optional[float],
        spread_pips: Optional[float],
        open_trades: Optional[List[Dict]] = None,
        state: Optional[TrailingState] = None,
        units: Optional[float] = None,
        now_utc: Optional[datetime] = None,
    ) -> bool:
        if self.time_stop_minutes <= 0 or self.time_stop_min_pips < 0:
            return False
        if minutes_open is None or minutes_open < self.time_stop_minutes:
            return False
        if pips is None:
            return False

        threshold = self._time_stop_threshold(instrument, atr_value)
        if pips >= threshold:
            return False

        summary = (
            f"TIME_STOP {instrument} age={minutes_open:.1f}m "
            f"pips={pips:.2f} threshold={threshold:.2f}"
        )
        now_val = now_utc or datetime.now(timezone.utc)
        if self._close_trade(
            trade_id,
            instrument,
            None,
            pips,
            threshold,
            threshold,
            spread_pips,
            log_prefix="[TIME-STOP]",
            reason="TIME_STOP",
            summary=summary,
            open_trades=open_trades,
            state=state,
            units=units,
            now_utc=now_val,
        ):
            return True
        return False

    def _maybe_aggressive_exit(
        self,
        trade: Dict,
        trade_id: str,
        instrument: str,
        profit: float,
        now_utc: datetime,
        open_trades: Optional[List[Dict]] = None,
        state: Optional[TrailingState] = None,
        units: Optional[float] = None,
    ) -> bool:
        if profit is None:
            return False

        minutes_open = self._minutes_open(trade, now_utc)
        if (
            minutes_open is not None
            and minutes_open > self.aggressive_max_hold_minutes
            and profit <= 0
        ):
            if self._close_trade(
                trade_id,
                instrument,
                profit,
                None,
                profit,
                profit,
                None,
                log_prefix="[TIME-EXIT]",
                reason=f"age>{minutes_open:.1f}m",
                summary=f"Closing {instrument} after {minutes_open:.1f} minutes, profit={profit:.2f}",
                open_trades=open_trades,
                state=state,
                units=units,
                now_utc=now_utc,
            ):
                return True

        atr_val = self._atr_for_trade(trade)
        ccy_limit = profit <= -self.aggressive_max_loss_ccy
        atr_limit = atr_val is not None and profit <= -(self.aggressive_max_loss_atr_mult * atr_val)
        if profit <= 0 and (ccy_limit or atr_limit):
            atr_str = "n/a" if atr_val is None else f"{atr_val:.4f}"
            if self._close_trade(
                trade_id,
                instrument,
                profit,
                None,
                profit,
                profit,
                None,
                log_prefix="[LOSS-FLOOR]",
                reason="LOSS_FLOOR",
                summary=f"Closing {instrument} loss={profit:.2f} atr={atr_str}",
                open_trades=open_trades,
                state=state,
                units=units,
                now_utc=now_utc,
            ):
                return True
        return False

    @staticmethod
    def _open_time_from_trade(trade: Dict) -> Optional[datetime]:
        if not isinstance(trade, dict):
            return None
        value = (
            trade.get("openTime")
            or trade.get("open_time")
            or trade.get("open_time_utc")
            or trade.get("open_ts")
        )
        if isinstance(value, datetime):
            return value.astimezone(timezone.utc)
        if isinstance(value, (int, float)):
            try:
                return datetime.fromtimestamp(value, tz=timezone.utc)
            except (OverflowError, ValueError, OSError):
                return None
        if isinstance(value, str):
            txt = value.strip()
            if not txt:
                return None
            try:
                # Handle Zulu suffix by converting to +00:00
                txt = txt.replace("Z", "+00:00")
                return datetime.fromisoformat(txt).astimezone(timezone.utc)
            except ValueError:
                return None
        return None

    def _minutes_open(self, trade: Dict, now_utc: datetime, open_time: Optional[datetime] = None) -> Optional[float]:
        opened_at = open_time or self._open_time_from_trade(trade)
        if opened_at is None:
            return None
        delta = now_utc - opened_at
        if delta < timedelta(0):
            return None
        return delta.total_seconds() / 60.0

    @staticmethod
    def _atr_for_trade(trade: Dict) -> Optional[float]:
        if not isinstance(trade, dict):
            return None
        atr_val = trade.get("atr") or trade.get("currentAtr") or trade.get("entryAtr")
        if atr_val is None:
            return None
        try:
            return float(atr_val)
        except (TypeError, ValueError):
            return None

    def _close_trade(
        self,
        trade_id: str,
        instrument: str,
        profit: Optional[float],
        pips: Optional[float],
        floor: float,
        high_water: float,
        spread_pips: Optional[float],
        *,
        log_prefix: str = "[TRAIL]",
        reason: str,
        summary: Optional[str] = None,
        open_trades: Optional[List[Dict]] = None,
        state: Optional[TrailingState] = None,
        units: Optional[float] = None,
        now_utc: Optional[datetime] = None,
    ) -> bool:
        """Attempt to close a trade and only return True on confirmed closure."""
        # False-positive closes are dangerous: if we drop a trade from local state
        # before the broker has actually closed it, the strategy can immediately
        # re-enter on the same instrument and double exposure. Guardrails below
        # require either an explicit broker success response or a follow-up check
        # that the trade is no longer present.

        now_utc = now_utc or datetime.now(timezone.utc)
        if state and state.close_cooldown_until and now_utc < state.close_cooldown_until:
            print(
                f"{log_prefix}[INFO] close_cooldown_active ticket={trade_id} instrument={instrument} until={state.close_cooldown_until.isoformat()}",
                flush=True,
            )
            return False

        spread_clause = f" spread={spread_pips:.2f}" if spread_pips is not None else ""
        if pips is not None:
            metric_clause = f"current_pips={pips:.2f}"
        elif profit is not None:
            metric_clause = f"current_profit={profit:.2f}"
        else:
            metric_clause = "current_profit=unknown"

        initial_snapshot = self._position_snapshot(instrument, fallback_units=units)
        long_units, short_units, snapshot_ok = self._snapshot_units(initial_snapshot, trade_id)
        snapshot_clause = "snapshot=missing" if not snapshot_ok else f"long={long_units:.0f} short={short_units:.0f}"
        print(
            f"{log_prefix}[INFO] close_side_detected ticket={trade_id} instrument={instrument} {snapshot_clause}",
            flush=True,
        )

        if snapshot_ok and long_units == 0 and short_units == 0:
            print(
                f"{log_prefix}[INFO] snapshot_shows_closed ticket={trade_id} instrument={instrument} {snapshot_clause}",
                flush=True,
            )
            self._reconcile_closed(trade_id, instrument, open_trades, state)
            return True

        payload = self._close_payload(long_units, short_units)
        print(
            f"{log_prefix}[INFO] attempting_close ticket={trade_id} instrument={instrument} sides=long:{long_units:.0f} short:{short_units:.0f} payload={payload} reason={reason}",
            flush=True,
        )
        result = self._execute_closeout(trade_id, instrument, long_units, short_units)

        status = result.get("status") if isinstance(result, dict) else None
        success = status in {"CLOSED", "SIMULATED", "FILLED"}

        if success:
            if summary:
                print(f"{log_prefix} {summary}{spread_clause} [Broker confirmed close]", flush=True)
            else:
                print(
                    f"{log_prefix} close ticket={trade_id} {metric_clause} floor={floor:.2f} "
                    f"high_water={high_water:.2f} reason={reason}{spread_clause} [Broker confirmed close]",
                    flush=True,
                )
            self._reconcile_closed(trade_id, instrument, open_trades, state)
            return True

        error_code = self._extract_error_code(result)
        missing_position_flag = error_code == "CLOSEOUT_POSITION_DOESNT_EXIST" or self._response_indicates_missing_position(result)

        if missing_position_flag:
            refreshed_snapshot = self._position_snapshot(instrument)
            refreshed_long, refreshed_short, refreshed_ok = self._snapshot_units(refreshed_snapshot, trade_id)
            refreshed_clause = (
                "snapshot=missing" if not refreshed_ok else f"long={refreshed_long:.0f} short={refreshed_short:.0f}"
            )
            if refreshed_ok and refreshed_long == 0 and refreshed_short == 0:
                print(
                    f"{log_prefix}[INFO] treated_as_closed_after_missing_position ticket={trade_id} instrument={instrument} {refreshed_clause}{spread_clause}",
                    flush=True,
                )
                self._reconcile_closed(trade_id, instrument, open_trades, state)
                return True
            if not refreshed_ok:
                snapshot_after = self._list_open_trades_quietly()
                instrument_open = self._instrument_open_in_snapshot(snapshot_after, instrument, trade_id)
                if instrument_open is False:
                    print(
                        f"{log_prefix}[INFO] treated_as_closed_after_missing_position ticket={trade_id} instrument={instrument} snapshot=absent{spread_clause}",
                        flush=True,
                    )
                    self._reconcile_closed(trade_id, instrument, open_trades, state)
                    return True

            if refreshed_ok:
                print(
                    f"{log_prefix}[WARN] broker_missing_position retry ticket={trade_id} instrument={instrument} {refreshed_clause}{spread_clause}",
                    flush=True,
                )
                retry_result = self._execute_closeout(trade_id, instrument, refreshed_long, refreshed_short)
                retry_status = retry_result.get("status") if isinstance(retry_result, dict) else None
                retry_success = retry_status in {"CLOSED", "SIMULATED", "FILLED"}
                if retry_success:
                    self._reconcile_closed(trade_id, instrument, open_trades, state)
                    print(
                        f"{log_prefix}[INFO] retry_close_success ticket={trade_id} instrument={instrument}{spread_clause}",
                        flush=True,
                    )
                    return True

                retry_error = self._extract_error_code(retry_result)
                if retry_error == "CLOSEOUT_POSITION_DOESNT_EXIST" or self._response_indicates_missing_position(retry_result):
                    refreshed_snapshot = self._position_snapshot(instrument)
                    refreshed_long, refreshed_short, refreshed_ok = self._snapshot_units(refreshed_snapshot, trade_id)
                    if refreshed_ok and refreshed_long == 0 and refreshed_short == 0:
                        print(
                            f"{log_prefix}[INFO] retry_missing_position treating_closed ticket={trade_id} instrument={instrument}{spread_clause}",
                            flush=True,
                        )
                        self._reconcile_closed(trade_id, instrument, open_trades, state)
                        return True
                if state:
                    state.missing_retry_attempted = True
                    state.close_cooldown_until = now_utc + timedelta(seconds=60)
                refreshed_clause = (
                    "snapshot=missing" if not refreshed_ok else f"long={refreshed_long:.0f} short={refreshed_short:.0f}"
                )
                print(
                    f"{log_prefix}[WARN] missing_position_but_snapshot_still_open ticket={trade_id} instrument={instrument} {refreshed_clause}{spread_clause}",
                    flush=True,
                )
                return False

        broker_snapshot = self._list_open_trades_quietly()
        closed_status = (
            False
            if self._instrument_open_in_snapshot(broker_snapshot, instrument, trade_id)
            else self._broker_confirms_closed(trade_id, instrument)
        )

        if closed_status is True:
            self._reconcile_closed(trade_id, instrument, open_trades, state)
            print(
                f"{log_prefix}[INFO] Broker confirmed close via snapshot ticket={trade_id} instrument={instrument}{spread_clause}",
                flush=True,
            )
            return True

        print(
            f"{log_prefix}[WARN] Close failed ticket={trade_id} {metric_clause} floor={floor:.2f} "
            f"high_water={high_water:.2f} reason={reason} resp={result}{spread_clause}",
            flush=True,
        )
        return False

    @staticmethod
    def _close_payload(long_units: float, short_units: float) -> Dict[str, str]:
        if short_units != 0 and long_units == 0:
            return {"shortUnits": "ALL"}
        if long_units != 0 and short_units == 0:
            return {"longUnits": "ALL"}
        if long_units != 0 or short_units != 0:
            return {"longUnits": "ALL", "shortUnits": "ALL"}
        return {"longUnits": "0", "shortUnits": "0"}

    def _execute_closeout(self, trade_id: str, instrument: str, long_units: float, short_units: float) -> Dict:
        """Send a side-specific closeout request."""

        try:
            if hasattr(self.broker, "close_position_side"):
                return self.broker.close_position_side(instrument, long_units, short_units)
            if hasattr(self.broker, "close_trade"):
                return self.broker.close_trade(trade_id, instrument=instrument)
        except AttributeError:
            return {"status": "ERROR", "errorCode": "close-not-supported"}
        except Exception as exc:  # pragma: no cover - defensive logging
            print(
                f"[TRAIL][ERROR] Exception closing {instrument}: {exc}",
                flush=True,
            )
            return {"status": "ERROR", "error": str(exc)}
        return {"status": "ERROR", "errorCode": "close-not-supported"}

    @staticmethod
    def _extract_error_code(result: Dict) -> Optional[str]:
        if not isinstance(result, dict):
            return None
        for key in ("errorCode", "error_code"):
            if key in result:
                val = result.get(key)
                if isinstance(val, str):
                    return val
        text = result.get("text")
        if isinstance(text, str):
            try:
                import json

                parsed = json.loads(text)
                for key in ("errorCode", "error_code"):
                    val = parsed.get(key)
                    if isinstance(val, str):
                        return val
            except Exception:
                return None
        return None

    def _response_indicates_missing_position(self, result: Dict) -> bool:
        """Return True when the broker response clearly states no position exists."""

        if not isinstance(result, dict):
            return False

        payload = None
        text = result.get("text")
        if isinstance(text, str):
            try:
                payload = json.loads(text)
            except Exception:
                if "CLOSEOUT_POSITION_DOESNT_EXIST" in text or "POSITION_CLOSEOUT_DOESNT_EXIST" in text:
                    return True

        payload = payload or result

        for key in ("errorCode", "error_code"):
            code = payload.get(key)
            if isinstance(code, str) and code == "CLOSEOUT_POSITION_DOESNT_EXIST":
                return True

        for leg in ("longOrderRejectTransaction", "shortOrderRejectTransaction"):
            reject_reason = (payload.get(leg) or {}).get("rejectReason")
            if isinstance(reject_reason, str) and (
                "CLOSEOUT_POSITION_DOESNT_EXIST" in reject_reason or "POSITION_CLOSEOUT_DOESNT_EXIST" in reject_reason
            ):
                return True

        message = payload.get("errorMessage")
        if isinstance(message, str) and "does not exist" in message:
            return True

        return False

    @staticmethod
    def _raw_units(trade: Dict) -> Optional[float]:
        """Return the raw units value if present, preserving zeros."""

        for key in ("currentUnits", "current_units", "units"):
            if key in trade:
                return trade.get(key)
        return None

    def _broker_confirms_closed(self, trade_id: Optional[str], instrument: str) -> Optional[bool]:
        """Return True only if broker reports no open position for the instrument.

        Returns False when the instrument is still present. Returns None when the broker
        cannot confirm (missing capability or transient failure).
        """

        try:
            if not hasattr(self.broker, "list_open_trades"):
                return None
            trades = self.broker.list_open_trades()
            if trades is None:
                return None
        except Exception as exc:  # pragma: no cover - defensive logging
            print(
                f"[TRAIL][WARN] Unable to confirm closure for {instrument}: {exc}",
                flush=True,
            )
            return None

        return not self._instrument_open_in_snapshot(trades, instrument, trade_id)

    def _pip_size(self, instrument: str) -> float:
        try:
            return float(self.broker._pip_size(instrument))
        except Exception:
            if instrument.endswith("JPY"):
                return 0.01
            if instrument.startswith("XAU"):
                return 0.1
            if instrument.startswith("XAG"):
                return 0.01
            return 0.0001

    def _current_spread(self, instrument: str) -> Optional[float]:
        try:
            spread = self.broker.current_spread(instrument)
            return None if spread is None else float(spread)
        except Exception:
            return None

    def _list_open_trades_quietly(self) -> Optional[List[Dict]]:
        try:
            if not hasattr(self.broker, "list_open_trades"):
                return None
            return self.broker.list_open_trades()
        except Exception:
            return None

    def _instrument_open_in_snapshot(
        self, trades: Optional[List[Dict]], instrument: str, trade_id: Optional[str]
    ) -> bool:
        for trade in trades or []:
            inst = trade.get("instrument")
            if instrument and inst != instrument:
                continue
            raw_units = self._raw_units(trade)
            if raw_units is not None:
                units = self._units_from_trade(trade)
                if units == 0:
                    continue
            if trade_id is None:
                return True
            live_id = trade.get("id") or trade.get("tradeID") or trade.get("position_id")
            if live_id is None:
                return True
            if str(live_id) == str(trade_id):
                return True
        return False

    def _reconcile_closed(
        self,
        trade_id: Optional[str],
        instrument: str,
        open_trades: Optional[List[Dict]],
        state: Optional[TrailingState],
    ) -> None:
        state = state or self._state.get(trade_id or "")
        if state:
            state.armed = False
            state.max_profit_ccy = None
            state.last_update = None
            state.open_time = None
            state.close_cooldown_until = None
            state.missing_retry_attempted = False
        self._mark_locally_closed(trade_id, instrument)
        if open_trades is not None:
            remaining = []
            for trade in open_trades:
                tid = self._trade_id(trade)
                inst = trade.get("instrument")
                if (trade_id is not None and tid is not None and str(tid) == str(trade_id)) or (
                    instrument and inst == instrument
                ):
                    if isinstance(trade, dict):
                        trade["state"] = "CLOSED"
                    continue
                remaining.append(trade)
            open_trades[:] = remaining
        if trade_id is not None:
            self._state.pop(trade_id, None)

    def _position_snapshot(self, instrument: str, fallback_units: Optional[float] = None) -> Optional[Dict]:
        snapshot = None
        try:
            if hasattr(self.broker, "position_snapshot"):
                snapshot = self.broker.position_snapshot(instrument)
        except Exception:  # pragma: no cover - defensive
            return None
        if snapshot is not None:
            return snapshot
        try:
            if hasattr(self.broker, "list_open_trades"):
                for trade in self.broker.list_open_trades() or []:
                    if trade.get("instrument") != instrument:
                        continue
                    units = self._units_from_trade(trade)
                    if units == 0:
                        return {"instrument": instrument, "longUnits": "0", "shortUnits": "0"}
                    if units > 0:
                        return {"instrument": instrument, "longUnits": str(units), "shortUnits": "0"}
                        return {"instrument": instrument, "longUnits": "0", "shortUnits": str(units)}
        except Exception:  # pragma: no cover - defensive
            return None
        if fallback_units is not None:
            if fallback_units == 0:
                return {"instrument": instrument, "longUnits": "0", "shortUnits": "0"}
            if fallback_units > 0:
                return {"instrument": instrument, "longUnits": str(fallback_units), "shortUnits": "0"}
            return {"instrument": instrument, "longUnits": "0", "shortUnits": str(fallback_units)}
        return None

    @staticmethod
    def _snapshot_units(snapshot: Optional[Dict], trade_id: Optional[str]) -> tuple[float, float, bool]:
        """Extract long/short units from a broker snapshot."""
        if not isinstance(snapshot, dict):
            return 0.0, 0.0, False
        long_raw = snapshot.get("longUnits") or (snapshot.get("long") or {}).get("units")
        short_raw = snapshot.get("shortUnits") or (snapshot.get("short") or {}).get("units")
        try:
            long_val = float(long_raw)
        except (TypeError, ValueError):
            long_val = 0.0
        try:
            short_val = float(short_raw)
        except (TypeError, ValueError):
            short_val = 0.0
        return long_val, short_val, True
