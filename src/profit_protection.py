"""Profit-protection logic for trailing unrealized gains."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List, Optional, Protocol


class BrokerLike(Protocol):
    def get_unrealized_profit(self, instrument: str) -> Optional[float]:
        ...

    def close_position(self, instrument: str) -> Dict:
        ...


class ProfitProtection:
    """Maintain per-instrument high-water marks and apply trailing exits."""

    def __init__(
        self,
        broker: BrokerLike,
        trigger: float = 3.0,
        trail: float = 0.5,
        *,
        aggressive: bool = False,
        aggressive_max_hold_minutes: float = 45.0,
        aggressive_max_loss_usd: float = 5.0,
        aggressive_max_loss_atr_mult: float = 1.2,
    ) -> None:
        self.broker = broker
        self.trigger = float(trigger)
        self.trail = float(trail)
        self.aggressive = aggressive
        self.aggressive_max_hold_minutes = float(aggressive_max_hold_minutes)
        self.aggressive_max_loss_usd = float(aggressive_max_loss_usd)
        self.aggressive_max_loss_atr_mult = float(aggressive_max_loss_atr_mult)
        self._high_water: Dict[str, float] = {}

    def snapshot(self) -> Dict[str, float]:
        """Return a shallow copy of the current high-water marks (useful for tests)."""
        return dict(self._high_water)

    def process_open_trades(self, open_trades: List[Dict]) -> List[str]:
        """
        Inspect open trades, update their high-water marks and close when they
        violate the trailing stop rule.

        Returns a list of instruments that were closed during this invocation.
        """
        active_instruments = set()
        closed_instruments: List[str] = []

        now_utc = datetime.now(timezone.utc)

        for trade in open_trades:
            label: str
            if isinstance(trade, dict):
                label = str(trade.get("instrument") or "<unknown>")
            else:
                label = str(trade)
            print(f"[CHECK-DEBUG] Checking {label}", flush=True)

            instrument = self._instrument_from_trade(trade)
            if not instrument:
                continue
            active_instruments.add(instrument)

            profit = self._safe_profit(instrument)
            if profit is None:
                continue

            high_water = self._high_water.get(instrument)
            if high_water is None or profit > high_water:
                self._high_water[instrument] = profit
                high_water = profit

            print(
                f"[TRAIL-DEBUG] profit={profit:.2f} high_water={high_water:.2f}",
                flush=True,
            )

            if self.aggressive and self._maybe_aggressive_exit(trade, instrument, profit, now_utc):
                closed_instruments.append(instrument)
                self._high_water.pop(instrument, None)
                continue

            if (
                high_water >= self.trigger
                and profit <= high_water - self.trail
                and self._close_position(instrument, profit, high_water)
            ):
                closed_instruments.append(instrument)
                self._high_water.pop(instrument, None)

        self._cleanup_stale(active_instruments)
        return closed_instruments

    def _cleanup_stale(self, active_instruments: Iterable[str]) -> None:
        active = set(active_instruments)
        stale = [inst for inst in self._high_water if inst not in active]
        for inst in stale:
            self._high_water.pop(inst, None)

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

    def _safe_profit(self, instrument: str) -> Optional[float]:
        try:
            profit = self.broker.get_unrealized_profit(instrument)
        except AttributeError:
            return None
        except Exception as exc:  # pragma: no cover - defensive logging
            print(
                f"[TRAIL] Failed to read unrealized P/L for {instrument}: {exc}",
                flush=True,
            )
            return None
        if profit is None:
            return None
        try:
            return float(profit)
        except (TypeError, ValueError):
            return None

    def _maybe_aggressive_exit(
        self,
        trade: Dict,
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
            if self._close_position(
                instrument,
                profit,
                profit,
                log_prefix="[TIME-EXIT]",
                summary=f"Closing {instrument} after {minutes_open:.1f} minutes, profit={profit:.2f}",
            ):
                return True

        atr_val = self._atr_for_trade(trade)
        usd_limit = profit <= -self.aggressive_max_loss_usd
        atr_limit = atr_val is not None and profit <= -(self.aggressive_max_loss_atr_mult * atr_val)
        if profit <= 0 and (usd_limit or atr_limit):
            atr_str = "n/a" if atr_val is None else f"{atr_val:.4f}"
            if self._close_position(
                instrument,
                profit,
                profit,
                log_prefix="[LOSS-FLOOR]",
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

    def _minutes_open(self, trade: Dict, now_utc: datetime) -> Optional[float]:
        opened_at = self._open_time_from_trade(trade)
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

    def _close_position(
        self,
        instrument: str,
        profit: float,
        high_water: float,
        *,
        log_prefix: str = "[TRAIL]",
        summary: Optional[str] = None,
    ) -> bool:
        try:
            result = self.broker.close_position(instrument)
        except AttributeError:
            return False
        except Exception as exc:  # pragma: no cover - defensive logging
            print(
                f"[TRAIL] Exception closing {instrument}: {exc}",
                flush=True,
            )
            return False

        if isinstance(result, dict) and result.get("status") in {"CLOSED", "SIMULATED"}:
            if summary:
                print(f"{log_prefix} {summary}", flush=True)
            else:
                diff = high_water - profit
                print(
                    f"{log_prefix} Closed {instrument} at ${profit:.2f} "
                    f"(fell ${diff:.2f} from high of ${high_water:.2f})",
                    flush=True,
                )
            return True

        # Unknown response but still log to aid debugging
        diff = high_water - profit
        print(
            f"{log_prefix} Attempted to close {instrument} at ${profit:.2f} "
            f"(fell ${diff:.2f} from high of ${high_water:.2f}) resp={result}",
            flush=True,
        )
        return True
