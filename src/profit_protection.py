"""Profit-protection logic for trailing unrealized gains."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List, Optional, Protocol


ARM_AT_USD = 0.75
GIVEBACK_USD = 0.50


@dataclass
class TrailingState:
    max_profit_usd: Optional[float] = None
    armed: bool = False
    last_update: Optional[datetime] = None
    open_time: Optional[datetime] = None


class BrokerLike(Protocol):
    def get_unrealized_profit(self, instrument: str) -> Optional[float]:
        ...

    def close_position(self, instrument: str) -> Dict:
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
        trigger: float = ARM_AT_USD,
        trail: float = GIVEBACK_USD,
        *,
        arm_pips: float = 0.0,
        giveback_pips: float = 0.0,
        arm_usd: Optional[float] = None,
        giveback_usd: Optional[float] = None,
        use_pips: bool = False,
        be_arm_pips: float = 0.0,
        be_offset_pips: float = 0.0,
        min_check_interval_sec: float = 0.0,
        aggressive: bool = False,
        aggressive_max_hold_minutes: float = 45.0,
        aggressive_max_loss_usd: float = 5.0,
        aggressive_max_loss_atr_mult: float = 1.2,
        time_stop_minutes: float = 90.0,
        time_stop_min_pips: float = 2.0,
        time_stop_xau_atr_mult: float = 0.35,
    ) -> None:
        self.broker = broker
        self.arm_pips = float(arm_pips)
        self.giveback_pips = float(giveback_pips)
        self.arm_usd = float(arm_usd if arm_usd is not None else trigger)
        self.giveback_usd = float(giveback_usd if giveback_usd is not None else trail)
        # Preserve legacy attributes for backwards compatibility/tests
        self.trigger = self.arm_usd
        self.trail = self.giveback_usd
        # Pip trailing is disabled; retain flag for compatibility only
        self.use_pips = False
        self.be_arm_pips = float(be_arm_pips)
        self.be_offset_pips = float(be_offset_pips)
        self.min_check_interval_sec = float(min_check_interval_sec)
        self.aggressive = aggressive
        self.aggressive_max_hold_minutes = float(aggressive_max_hold_minutes)
        self.aggressive_max_loss_usd = float(aggressive_max_loss_usd)
        self.aggressive_max_loss_atr_mult = float(aggressive_max_loss_atr_mult)
        self.time_stop_minutes = float(time_stop_minutes)
        self.time_stop_min_pips = float(time_stop_min_pips)
        self.time_stop_xau_atr_mult = float(time_stop_xau_atr_mult)
        self._state: Dict[str, TrailingState] = {}

    def snapshot(self) -> Dict[str, TrailingState]:
        """Return a shallow copy of the current per-trade trailing state (test helper)."""
        return dict(self._state)

    def process_open_trades(self, open_trades: List[Dict], *, now_utc: Optional[datetime] = None) -> List[str]:
        """Inspect open trades, update their high-water marks, and close on trailing giveback."""

        now_utc = now_utc or datetime.now(timezone.utc)
        active_keys = set()
        closed_trades: List[str] = []

        for trade in open_trades:
            trade_id = self._trade_id(trade)
            instrument = self._instrument_from_trade(trade)
            units = self._units_from_trade(trade)
            if not trade_id or not instrument or units == 0:
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
            ):
                closed_trades.append(trade_id)
                self._state.pop(trade_id, None)
                continue

            if self._interval_blocked(state, now_utc):
                continue

            self._update_peak_profit(trade_id, state, profit)
            state.last_update = now_utc
            self._state[trade_id] = state

            if self.aggressive and self._maybe_aggressive_exit(trade, trade_id, instrument, profit, now_utc):
                closed_trades.append(trade_id)
                self._state.pop(trade_id, None)
                continue

            if profit is None or state.max_profit_usd is None:
                continue

            if not state.armed and state.max_profit_usd >= self.arm_usd:
                state.armed = True
                print(f"[TRAIL] armed ticket={trade_id} profit_usd={profit:.2f}", flush=True)

            if not state.armed:
                continue

            trailing_floor = state.max_profit_usd - self.giveback_usd
            if (state.max_profit_usd - profit) >= self.giveback_usd:
                if self._close_trade(
                    trade_id,
                    instrument,
                    profit,
                    None,
                    trailing_floor,
                    state.max_profit_usd,
                    spread_pips,
                    reason="usd_profit_protection",
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
        if isinstance(trade, dict):
            for key in ("unrealizedPL", "unrealized_pl", "profit", "floating"):
                if key in trade:
                    try:
                        return float(trade[key])
                    except (TypeError, ValueError):
                        break
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
        if state.max_profit_usd is None or profit > state.max_profit_usd:
            state.max_profit_usd = profit
            print(f"[TRAIL] update ticket={trade_id} max_profit_usd={profit:.2f}", flush=True)

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
            ):
                return True

        atr_val = self._atr_for_trade(trade)
        usd_limit = profit <= -self.aggressive_max_loss_usd
        atr_limit = atr_val is not None and profit <= -(self.aggressive_max_loss_atr_mult * atr_val)
        if profit <= 0 and (usd_limit or atr_limit):
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
    ) -> bool:
        """Attempt to close a trade and only return True on confirmed closure.

        False-positive closes are dangerous: if we drop a trade from local state
        before the broker has actually closed it, the strategy can immediately
        re-enter on the same instrument and double exposure. Guardrails below
        require either an explicit broker success response or a follow-up check
        that the trade is no longer present.
        """

        try:
            if hasattr(self.broker, "close_trade"):
                result = self.broker.close_trade(trade_id, instrument=instrument)
            else:
                result = self.broker.close_position(instrument)
        except AttributeError:
            return False
        except Exception as exc:  # pragma: no cover - defensive logging
            print(
                f"[TRAIL][ERROR] Exception closing {instrument}: {exc}",
                flush=True,
            )
            return False

        status = result.get("status") if isinstance(result, dict) else None
        success = status in {"CLOSED", "SIMULATED", "FILLED"}
        spread_clause = ""
        if spread_pips is not None:
            spread_clause = f" spread={spread_pips:.2f}"
        metric_clause = (
            f"current_pips={pips:.2f}" if pips is not None else f"current_profit={profit:.2f}"
        )

        if success:
            if summary:
                print(f"{log_prefix} {summary}{spread_clause}", flush=True)
            else:
                print(
                    f"{log_prefix} close ticket={trade_id} {metric_clause} floor={floor:.2f} "
                    f"high_water={high_water:.2f} reason={reason}{spread_clause}",
                    flush=True,
                )
            return True

        error_code = self._extract_error_code(result)
        if error_code == "CLOSEOUT_POSITION_DOESNT_EXIST":
            if self._broker_confirms_closed(trade_id, instrument):
                print(
                    f"{log_prefix}[INFO] Trade already closed at broker; marking closed ticket={trade_id} instrument={instrument}{spread_clause}",
                    flush=True,
                )
                return True
            print(
                f"{log_prefix}[WARN] Broker reported CLOSEOUT_POSITION_DOESNT_EXIST but {instrument} still appears open; ticket={trade_id} resp={result}{spread_clause}",
                flush=True,
            )
        else:
            # If the broker did not acknowledge the close, perform a follow-up check
            # to see whether the position already disappeared (e.g., previously closed
            # or closed by another rule). Only then can we safely treat it as closed.
            if self._broker_confirms_closed(trade_id, instrument):
                print(
                    f"{log_prefix}[WARN] Close response inconclusive but {instrument} is no longer open; marking closed",
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

    def _broker_confirms_closed(self, trade_id: Optional[str], instrument: str) -> bool:
        """Return True only if broker reports no open position for the instrument."""

        try:
            if not hasattr(self.broker, "list_open_trades"):
                return False
            trades = self.broker.list_open_trades()
        except Exception as exc:  # pragma: no cover - defensive logging
            print(
                f"[TRAIL][WARN] Unable to confirm closure for {instrument}: {exc}",
                flush=True,
            )
            return False

        for trade in trades or []:
            inst = trade.get("instrument")
            if instrument and inst == instrument:
                # Instrument still open; if IDs match we know the trade is alive.
                if trade_id is None:
                    return False
                live_id = trade.get("id") or trade.get("tradeID") or trade.get("position_id")
                if live_id is None:
                    return False
                if str(live_id) == str(trade_id):
                    return False
        return True

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
