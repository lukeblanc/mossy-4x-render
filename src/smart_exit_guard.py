from __future__ import annotations

from datetime import datetime, timezone
import os
from typing import Dict, List, Optional

from src.journal_reconciler import JournalReconcilerProfitProtection
from src.profit_protection import TrailingState


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return float(default)


class SmartExitGuard(JournalReconcilerProfitProtection):
    """Always-on cash loss protection plus stepped profit locking.

    This layer is deliberately independent of AGGRESSIVE_MODE so disabling
    aggressive trading can never disable the emergency cash-loss guard.
    The inherited legacy trailing exit is disabled here; this class owns the
    production cash-based exit ladder while the parent still handles time stops,
    broker reconciliation, journal persistence, and close confirmation.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.hard_max_loss_ccy = max(0.0, _env_float("HARD_MAX_LOSS_CCY", 1.50))
        self.profit_protect_trigger_ccy = max(
            0.0, _env_float("PROFIT_PROTECT_TRIGGER_CCY", 1.50)
        )
        self.profit_protect_floor_ccy = _env_float("PROFIT_PROTECT_FLOOR_CCY", 0.0)
        self.profit_trail_arm_ccy = max(
            self.profit_protect_trigger_ccy,
            _env_float("PROFIT_TRAIL_ARM_CCY", 3.00),
        )
        self.profit_trail_giveback_ccy = max(
            0.01, _env_float("PROFIT_TRAIL_GIVEBACK_CCY", 0.50)
        )

        # The base class has an older cash trailing rule that can fire before its
        # arm threshold. Disable only that legacy rule; all reconciliation and
        # time-stop behaviour remains active through super().process_open_trades().
        self.arm_ccy = float("inf")
        self.giveback_ccy = float("inf")
        self.arm_usd = self.arm_ccy
        self.giveback_usd = self.giveback_ccy
        self.trigger = self.arm_ccy
        self.trail = self.giveback_ccy

        print(
            "[SMART-EXIT][CONFIG] "
            f"hard_loss={self.hard_max_loss_ccy:.2f} "
            f"protect_trigger={self.profit_protect_trigger_ccy:.2f} "
            f"protect_floor={self.profit_protect_floor_ccy:.2f} "
            f"trail_arm={self.profit_trail_arm_ccy:.2f} "
            f"trail_giveback={self.profit_trail_giveback_ccy:.2f}",
            flush=True,
        )

    def _smart_close(
        self,
        *,
        trade_id: str,
        instrument: str,
        profit: float,
        peak: float,
        floor: float,
        reason: str,
        open_trades: List[Dict],
        state: TrailingState,
        units: float,
        now_utc: datetime,
    ) -> bool:
        print(
            f"[SMART-EXIT][INFO] close_requested ticket={trade_id} instrument={instrument} "
            f"profit={profit:.2f} peak={peak:.2f} floor={floor:.2f} reason={reason}",
            flush=True,
        )
        return self._close_trade(
            trade_id,
            instrument,
            profit,
            None,
            floor,
            peak,
            self._current_spread(instrument),
            log_prefix="[SMART-EXIT]",
            reason=reason,
            summary=(
                f"Closing {instrument}: profit={profit:.2f}, peak={peak:.2f}, "
                f"floor={floor:.2f}, reason={reason}"
            ),
            open_trades=open_trades,
            state=state,
            units=units,
            now_utc=now_utc,
        )

    def process_open_trades(
        self,
        open_trades: List[Dict],
        *,
        now_utc: Optional[datetime] = None,
    ) -> List[str]:
        now_val = now_utc or datetime.now(timezone.utc)
        closed: List[str] = []

        for trade in list(open_trades or []):
            if not isinstance(trade, dict):
                continue

            self._prime_state(trade)
            trade_id = self._trade_id(trade)
            instrument = self._instrument_from_trade(trade)
            units = self._units_from_trade(trade)
            if not trade_id or not instrument or units == 0:
                continue
            if self._is_locally_closed(trade_id, instrument):
                continue

            profit = self._profit_from_trade(trade, instrument)
            if profit is None:
                continue

            state = self._state.get(trade_id) or TrailingState()
            state.open_time = state.open_time or self._open_time_from_trade(trade)
            self._update_peak_profit(trade_id, state, profit)
            state.last_profit_ccy = profit
            state.last_update = now_val
            self._state[trade_id] = state
            peak = float(state.max_profit_ccy if state.max_profit_ccy is not None else profit)

            close_reason: Optional[str] = None
            close_floor = 0.0

            # Layer 1: absolute emergency loss floor, always active.
            if self.hard_max_loss_ccy > 0 and profit <= -self.hard_max_loss_ccy:
                close_reason = "HARD_CASH_LOSS_FLOOR"
                close_floor = -self.hard_max_loss_ccy

            # Layer 2: once a trade reaches the main profit zone, follow its peak
            # and bank it after a 50c (configurable) giveback.
            elif peak >= self.profit_trail_arm_ccy:
                trailing_floor = peak - self.profit_trail_giveback_ccy
                if profit <= trailing_floor:
                    close_reason = "SMART_PROFIT_TRAIL"
                    close_floor = trailing_floor

            # Layer 3: after a meaningful early win, do not let the trade turn
            # back into a loser. Default floor is break-even (0.00 account ccy).
            elif peak >= self.profit_protect_trigger_ccy and profit <= self.profit_protect_floor_ccy:
                close_reason = "SMART_BREAK_EVEN_PROTECT"
                close_floor = self.profit_protect_floor_ccy

            if close_reason and self._smart_close(
                trade_id=trade_id,
                instrument=instrument,
                profit=profit,
                peak=peak,
                floor=close_floor,
                reason=close_reason,
                open_trades=open_trades,
                state=state,
                units=units,
                now_utc=now_val,
            ):
                closed.append(trade_id)
                self._state.pop(trade_id, None)

        inherited_closed = super().process_open_trades(open_trades, now_utc=now_val)
        for trade_id in inherited_closed:
            if trade_id not in closed:
                closed.append(trade_id)
        return closed


__all__ = ["SmartExitGuard"]
