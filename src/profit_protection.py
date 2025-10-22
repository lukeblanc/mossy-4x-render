"""Profit-protection logic for trailing unrealized gains."""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Protocol


class BrokerLike(Protocol):
    def get_unrealized_profit(self, instrument: str) -> Optional[float]:
        ...

    def close_position(self, instrument: str) -> Dict:
        ...


class ProfitProtection:
    """Maintain per-instrument high-water marks and apply trailing exits."""

    def __init__(self, broker: BrokerLike, trigger: float = 3.0, trail: float = 0.5) -> None:
        self.broker = broker
        self.trigger = float(trigger)
        self.trail = float(trail)
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

        for trade in open_trades:
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

    def _close_position(self, instrument: str, profit: float, high_water: float) -> bool:
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
            diff = high_water - profit
            print(
                f"[TRAIL] Closed {instrument} at ${profit:.2f} "
                f"(fell ${diff:.2f} from high of ${high_water:.2f})",
                flush=True,
            )
            return True

        # Unknown response but still log to aid debugging
        diff = high_water - profit
        print(
            f"[TRAIL] Attempted to close {instrument} at ${profit:.2f} "
            f"(fell ${diff:.2f} from high of ${high_water:.2f}) resp={result}",
            flush=True,
        )
        return True

