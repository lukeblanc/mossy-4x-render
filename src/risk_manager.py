from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

from zoneinfo import ZoneInfo

AWST = ZoneInfo("Australia/Perth")

DEFAULT_ATR_STOP_MULT = 1.8


def _state_path() -> Path:
    root = os.getenv("MOSSY_STATE_PATH")
    base = Path(root) if root else Path("state")
    base.mkdir(parents=True, exist_ok=True)
    return base / "risk_state.json"


def _iso(dt: Optional[datetime]) -> Optional[str]:
    return dt.isoformat() if dt else None


def _from_iso(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _sanitize_equity(equity: Optional[float]) -> Optional[float]:
    if equity is None:
        return None
    try:
        value = float(equity)
    except (TypeError, ValueError):
        return None
    if value <= 0.0:
        return None
    return value


@dataclass
class RiskState:
    day_id: Optional[str] = None
    week_id: Optional[str] = None
    day_start_equity: Optional[float] = None
    week_start_equity: Optional[float] = None
    daily_realized_pl: float = 0.0
    weekly_realized_pl: float = 0.0
    last_entry_ts_utc: Optional[datetime] = None
    has_hit_weekly_target: bool = False
    live_halted_on_equity_floor: bool = False

    def to_dict(self) -> Dict:
        return {
            "day_id": self.day_id,
            "week_id": self.week_id,
            "day_start_equity": self.day_start_equity,
            "week_start_equity": self.week_start_equity,
            "daily_realized_pl": self.daily_realized_pl,
            "weekly_realized_pl": self.weekly_realized_pl,
            "last_entry_ts_utc": _iso(self.last_entry_ts_utc),
            "has_hit_weekly_target": self.has_hit_weekly_target,
            "live_halted_on_equity_floor": self.live_halted_on_equity_floor,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "RiskState":
        return cls(
            day_id=data.get("day_id"),
            week_id=data.get("week_id"),
            day_start_equity=data.get("day_start_equity"),
            week_start_equity=data.get("week_start_equity"),
            daily_realized_pl=float(data.get("daily_realized_pl", 0.0)),
            weekly_realized_pl=float(data.get("weekly_realized_pl", 0.0)),
            last_entry_ts_utc=_from_iso(data.get("last_entry_ts_utc")),
            has_hit_weekly_target=bool(data.get("has_hit_weekly_target", False)),
            live_halted_on_equity_floor=bool(
                data.get("live_halted_on_equity_floor", False)
            ),
        )


def _parse_time(value: Optional[str]) -> Optional[Tuple[int, int]]:
    if not value:
        return None
    try:
        hours, minutes = value.split(":", 1)
        return int(hours), int(minutes)
    except ValueError:
        return None


@dataclass
class RiskManager:
    config: Dict
    mode: str = "paper"
    state: RiskState = field(default_factory=RiskState)

    def __post_init__(self) -> None:
        self.mode = (self.mode or "paper").lower()
        self._state_file = _state_path()
        self._load_state()
        self.weekly_profit_target = float(
            self.config.get("weekly_profit_target", 250.0)
        )
        self.no_giveback_below_target = bool(
            self.config.get("no_giveback_below_target", True)
        )
        self.allow_trading_above_target = bool(
            self.config.get("allow_trading_above_target", True)
        )
        self.equity_floor = float(self.config.get("equity_floor", 1000.0))
        self.risk_per_trade_pct = float(
            self.config.get("risk_per_trade_pct", 0.0025)
        )
        self.max_concurrent_positions = int(
            self.config.get("max_concurrent_positions", 1)
        )
        self.cooldown_minutes = int(self.config.get("cooldown_minutes", 45))
        self.daily_loss_cap_pct = float(
            self.config.get("daily_loss_cap_pct", 0.01)
        )
        self.weekly_loss_cap_pct = float(
            self.config.get("weekly_loss_cap_pct", 0.03)
        )

        configured_atr_mult = self.config.get("atr_stop_mult")
        if configured_atr_mult is None:
            atr_mult = DEFAULT_ATR_STOP_MULT
        else:
            try:
                atr_mult = float(configured_atr_mult)
            except (TypeError, ValueError):
                atr_mult = DEFAULT_ATR_STOP_MULT
        if atr_mult <= 0.0:
            atr_mult = DEFAULT_ATR_STOP_MULT
        self.atr_stop_mult = min(atr_mult, DEFAULT_ATR_STOP_MULT)
        self.spread_pips_limit: Dict[str, float] = dict(
            self.config.get("spread_pips_limit", {}) or {}
        )
        self.default_spread_limit = self.config.get("default_spread_pips_limit")
        self.rollover_quiet = self.config.get("rollover_quiet_awst", {}) or {}
        self.rollover_start = _parse_time(self.rollover_quiet.get("start"))
        self.rollover_end = _parse_time(self.rollover_quiet.get("end"))

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def _load_state(self) -> None:
        if not self._state_file.exists():
            return
        try:
            with self._state_file.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception:
            return
        self.state = RiskState.from_dict(payload)

    def _save_state(self) -> None:
        payload = self.state.to_dict()
        try:
            with self._state_file.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def enforce_equity_floor(
        self,
        now_utc: datetime,
        equity: float,
        close_all_cb: Callable[[], None],
    ) -> None:
        self._rollover(now_utc, equity)
        if self.mode != "live":
            if self.state.live_halted_on_equity_floor:
                self.state.live_halted_on_equity_floor = False
                self._save_state()
            return

        if equity <= self.equity_floor:
            if not self.state.live_halted_on_equity_floor:
                try:
                    close_all_cb()
                except Exception:
                    pass
                self.state.live_halted_on_equity_floor = True
                self._save_state()
            return

        if self.state.live_halted_on_equity_floor and equity > self.equity_floor:
            self.state.live_halted_on_equity_floor = False
            self._save_state()

    def should_open(
        self,
        now_utc: datetime,
        equity: float,
        open_positions: list,
        instrument: str,
        spread_pips: Optional[float],
    ) -> Tuple[bool, str]:
        self._rollover(now_utc, equity)

        if self.mode == "live":
            if equity <= self.equity_floor:
                return False, "equity-floor"
            if self.state.live_halted_on_equity_floor:
                if equity > self.equity_floor:
                    self.state.live_halted_on_equity_floor = False
                    self._save_state()
                else:
                    return False, "equity-floor"

        if self.weekly_profit_target > 0:
            if self.state.weekly_realized_pl >= self.weekly_profit_target:
                if not self.state.has_hit_weekly_target:
                    self.state.has_hit_weekly_target = True
                    self._save_state()
            if self.state.has_hit_weekly_target:
                if (
                    self.no_giveback_below_target
                    and self.state.weekly_realized_pl < self.weekly_profit_target
                ):
                    return False, "weekly-target-locked"
                if not self.allow_trading_above_target:
                    return False, "weekly-target-reached"

        if self._breached_daily_loss(equity):
            return False, "daily-loss-cap"

        if self._breached_weekly_loss(equity):
            return False, "weekly-loss-cap"

        if self.max_concurrent_positions > 0 and len(open_positions) >= self.max_concurrent_positions:
            return False, "max-positions"

        if self.cooldown_minutes > 0 and self.state.last_entry_ts_utc:
            delta = now_utc - self.state.last_entry_ts_utc
            if delta < timedelta(minutes=self.cooldown_minutes):
                return False, "cooldown"

        if self._in_rollover_window(now_utc):
            return False, "rollover-window"

        spread_limit = self._spread_limit_for(instrument)
        if spread_limit is not None and spread_pips is not None:
            if spread_pips > spread_limit:
                return False, "spread-too-wide"

        return True, "ok"

    def sl_distance_from_atr(self, atr_price_units: Optional[float]) -> float:
        if atr_price_units is None or atr_price_units <= 0:
            return 0.0
        return max(0.0, atr_price_units * self.atr_stop_mult)

    def register_entry(self, now_utc: datetime) -> None:
        self.state.last_entry_ts_utc = now_utc
        self._save_state()

    def register_exit(self, realized_pl: float) -> None:
        self.state.daily_realized_pl += float(realized_pl)
        self.state.weekly_realized_pl += float(realized_pl)
        self._save_state()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _rollover(self, now_utc: datetime, equity: float) -> None:
        awst_now = now_utc.astimezone(AWST)
        day_id = awst_now.strftime("%Y-%m-%d")
        iso_cal = awst_now.isocalendar()
        week_id = f"{iso_cal.year}-W{iso_cal.week:02d}"
        changed = False
        valid_equity = _sanitize_equity(equity)

        if valid_equity is None:
            if self.state.day_id != day_id:
                self.state.day_id = day_id
                self.state.day_start_equity = None
                changed = True

            if self.state.week_id != week_id:
                self.state.week_id = week_id
                self.state.week_start_equity = None
                self.state.has_hit_weekly_target = False
                self.state.live_halted_on_equity_floor = False
                changed = True

            if changed:
                self._save_state()
            return

        if self.state.day_id != day_id:
            self.state.day_id = day_id
            self.state.day_start_equity = valid_equity
            self.state.daily_realized_pl = 0.0
            changed = True

        if self.state.week_id != week_id:
            self.state.week_id = week_id
            self.state.week_start_equity = valid_equity
            self.state.weekly_realized_pl = 0.0
            self.state.has_hit_weekly_target = False
            self.state.live_halted_on_equity_floor = False
            changed = True

        if self.state.day_start_equity is None and valid_equity is not None:
            self.state.day_start_equity = valid_equity
            if self.state.day_id == day_id:
                self.state.daily_realized_pl = 0.0
            changed = True

        if self.state.week_start_equity is None and valid_equity is not None:
            self.state.week_start_equity = valid_equity
            if self.state.week_id == week_id:
                self.state.weekly_realized_pl = 0.0
            changed = True

        if changed:
            self._save_state()

    def _breached_daily_loss(self, equity: float) -> bool:
        if self.daily_loss_cap_pct <= 0:
            return False
        if self.state.day_start_equity is None:
            return False
        drawdown = self.state.day_start_equity - equity
        return drawdown >= self.state.day_start_equity * self.daily_loss_cap_pct

    def _breached_weekly_loss(self, equity: float) -> bool:
        if self.weekly_loss_cap_pct <= 0:
            return False
        if self.state.week_start_equity is None:
            return False
        drawdown = self.state.week_start_equity - equity
        return drawdown >= self.state.week_start_equity * self.weekly_loss_cap_pct

    def _in_rollover_window(self, now_utc: datetime) -> bool:
        if not self.rollover_start or not self.rollover_end:
            return False
        awst_now = now_utc.astimezone(AWST)
        start_h, start_m = self.rollover_start
        end_h, end_m = self.rollover_end
        start = awst_now.replace(hour=start_h, minute=start_m, second=0, microsecond=0)
        end = awst_now.replace(hour=end_h, minute=end_m, second=0, microsecond=0)
        if end < start:
            end += timedelta(days=1)
            if awst_now < start:
                awst_now += timedelta(days=1)
        return start <= awst_now < end

    def _spread_limit_for(self, instrument: str) -> Optional[float]:
        if instrument in self.spread_pips_limit:
            return float(self.spread_pips_limit[instrument])
        if self.default_spread_limit is None:
            return None
        try:
            return float(self.default_spread_limit)
        except (TypeError, ValueError):
            return None

