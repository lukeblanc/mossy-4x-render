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
    base = Path(root) if root else Path("data")
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
    day_id_utc: Optional[str] = None
    day_start_equity: Optional[float] = None
    day_start_equity_utc: Optional[float] = None
    peak_equity_today: Optional[float] = None
    week_start_equity: Optional[float] = None
    daily_pl: float = 0.0
    drawdown_pct: float = 0.0
    daily_realized_pl: float = 0.0
    weekly_realized_pl: float = 0.0
    last_entry_ts_utc: Optional[datetime] = None
    last_entry_per_instrument: Dict[str, datetime] = field(default_factory=dict)
    cooldown_until: Dict[str, datetime] = field(default_factory=dict)
    peak_equity: Optional[float] = None
    last_trades: list = field(default_factory=list)
    has_hit_weekly_target: bool = False
    live_halted_on_equity_floor: bool = False
    max_drawdown_halt: bool = False
    daily_profit_cap_hit: bool = False
    daily_loss_cap_hit: bool = False
    daily_entry_count: int = 0

    def to_dict(self) -> Dict:
        return {
            "day_id": self.day_id,
            "week_id": self.week_id,
            "day_id_utc": self.day_id_utc,
            "day_start_equity": self.day_start_equity,
            "day_start_equity_utc": self.day_start_equity_utc,
            "peak_equity_today": self.peak_equity_today,
            "week_start_equity": self.week_start_equity,
            "daily_pl": self.daily_pl,
            "drawdown_pct": self.drawdown_pct,
            "daily_realized_pl": self.daily_realized_pl,
            "weekly_realized_pl": self.weekly_realized_pl,
            "last_entry_ts_utc": _iso(self.last_entry_ts_utc),
            "last_entry_per_instrument": {k: _iso(v) for k, v in self.last_entry_per_instrument.items()},
            "cooldown_until": {k: _iso(v) for k, v in self.cooldown_until.items()},
            "peak_equity": self.peak_equity,
            "last_trades": list(self.last_trades),
            "has_hit_weekly_target": self.has_hit_weekly_target,
            "live_halted_on_equity_floor": self.live_halted_on_equity_floor,
            "max_drawdown_halt": self.max_drawdown_halt,
            "daily_profit_cap_hit": self.daily_profit_cap_hit,
            "daily_loss_cap_hit": self.daily_loss_cap_hit,
            "daily_entry_count": self.daily_entry_count,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "RiskState":
        return cls(
            day_id=data.get("day_id"),
            week_id=data.get("week_id"),
            day_id_utc=data.get("day_id_utc"),
            day_start_equity=data.get("day_start_equity"),
            day_start_equity_utc=data.get("day_start_equity_utc"),
            peak_equity_today=_sanitize_equity(data.get("peak_equity_today")),
            week_start_equity=data.get("week_start_equity"),
            daily_pl=float(data.get("daily_pl", 0.0)),
            drawdown_pct=float(data.get("drawdown_pct", 0.0)),
            daily_realized_pl=float(data.get("daily_realized_pl", 0.0)),
            weekly_realized_pl=float(data.get("weekly_realized_pl", 0.0)),
            last_entry_ts_utc=_from_iso(data.get("last_entry_ts_utc")),
            last_entry_per_instrument={k: _from_iso(v) for k, v in (data.get("last_entry_per_instrument") or {}).items()},
            cooldown_until={k: _from_iso(v) for k, v in (data.get("cooldown_until") or {}).items()},
            peak_equity=_sanitize_equity(data.get("peak_equity")),
            last_trades=list(data.get("last_trades") or []),
            has_hit_weekly_target=bool(data.get("has_hit_weekly_target", False)),
            live_halted_on_equity_floor=bool(
                data.get("live_halted_on_equity_floor", False)
            ),
            max_drawdown_halt=bool(data.get("max_drawdown_halt", False)),
            daily_profit_cap_hit=bool(data.get("daily_profit_cap_hit", False)),
            daily_loss_cap_hit=bool(data.get("daily_loss_cap_hit", False)),
            daily_entry_count=int(data.get("daily_entry_count", 0) or 0),
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
    state_dir: Optional[Path] = None
    demo_mode: bool = False
    _last_equity_seen: Optional[float] = field(default=None, init=False, repr=False)
    _startup_reset_done: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self.mode = (self.mode or "paper").lower()
        self.demo_mode = bool(self.demo_mode)
        if self.state_dir:
            self.state_dir.mkdir(parents=True, exist_ok=True)
            self._state_file = (self.state_dir / "risk_state.json").resolve()
        else:
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
            self.config.get("risk_per_trade_pct", 0.005)
        )
        env_max_positions = os.getenv("MAX_CONCURRENT_POSITIONS") or os.getenv("MAX_OPEN_TRADES")
        max_positions_cfg = self.config.get("max_concurrent_positions", self.config.get("max_open_trades", 3))
        self.max_concurrent_positions = int(env_max_positions or max_positions_cfg or 3)
        self.cooldown_candles = int(self.config.get("cooldown_candles", 9))
        env_max_trades = os.getenv("MAX_TRADES_PER_DAY")
        configured_max_trades = self.config.get("max_trades_per_day", 0)
        soft_cap = int(os.getenv("MINI_RUN_MAX_TRADES_PER_DAY", 5))  # MINI-RUN safety: keep daily trades tight
        base_daily_trades = int(env_max_trades or configured_max_trades or soft_cap or 0)
        if env_max_trades is None:
            base_daily_trades = min(base_daily_trades, soft_cap)
        self.max_trades_per_day = int(base_daily_trades)
        self.daily_loss_cap_pct = float(
            self.config.get("daily_loss_cap_pct", 0.02)
        )
        self.daily_profit_target_usd = float(
            self.config.get("daily_profit_target_usd", 5.0)
        )
        self.weekly_loss_cap_pct = float(
            self.config.get("weekly_loss_cap_pct", 0.03)
        )
        self.sl_atr_mult = self._coerce_positive(
            self.config.get("sl_atr_mult", self.config.get("atr_stop_mult", 1.2)),
            cap=DEFAULT_ATR_STOP_MULT,
        )
        self.tp_atr_mult = self._coerce_positive(
            self.config.get("tp_atr_mult", self.config.get("tp_rr_multiple", 1.0)),
            cap=None,
        )
        self.atr_stop_mult = self.sl_atr_mult  # backwards compatibility attribute
        self.tp_rr_multiple = self.tp_atr_mult
        self.instrument_atr_multipliers: Dict[str, Dict[str, float]] = dict(
            self.config.get("instrument_atr_multipliers", {}) or {}
        )
        self.spread_pips_limit: Dict[str, float] = dict(
            self.config.get("spread_pips_limit", {}) or {}
        )
        self.default_spread_limit = self.config.get("default_spread_pips_limit")
        self.rollover_quiet = self.config.get("rollover_quiet_awst", {}) or {}
        self.rollover_start = _parse_time(self.rollover_quiet.get("start"))
        self.rollover_end = _parse_time(self.rollover_quiet.get("end"))
        self.timeframe = (self.config.get("timeframe") or "M5").upper()
        self.max_drawdown_cap_pct = float(self.config.get("max_drawdown_cap_pct", 0.10))
        equity_adjustment_pct_cfg = self.config.get(
            "equity_adjustment_pct", self.config.get("EQUITY_ADJUSTMENT_PCT", 0.05)
        )
        equity_adjustment_abs_cfg = self.config.get(
            "equity_adjustment_abs", self.config.get("EQUITY_ADJUSTMENT_ABS", 20.0)
        )
        self.equity_adjustment_pct = max(0.0, float(equity_adjustment_pct_cfg))
        self.equity_adjustment_abs = max(0.0, float(equity_adjustment_abs_cfg))

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
        self._update_peak_equity(equity)
        self._remember_equity(equity)
        if self._breached_max_drawdown(equity):
            self.state.max_drawdown_halt = True
            self._save_state()

        if self.state.max_drawdown_halt:
            return
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
        open_positions_count = len(open_positions) if open_positions is not None else None
        self._maybe_shift_baselines_for_adjustment(equity, open_positions_count)
        self._update_peak_equity(equity)

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

        if self._breached_daily_profit_cap(equity):
            print("[PROFIT-CAP] New entries blocked due to daily profit cap", flush=True)
            return False, "daily-profit-cap"

        if self.state.max_drawdown_halt:
            return False, "max-drawdown"

        if self.max_trades_per_day > 0 and self.state.daily_entry_count >= self.max_trades_per_day:
            return False, "daily-trade-cap"

        if self.max_concurrent_positions > 0 and len(open_positions) >= self.max_concurrent_positions:
            return False, "max-positions"

        cooldown_end = self.state.cooldown_until.get(instrument)
        if cooldown_end and now_utc < cooldown_end:
            return False, "cooldown"

        if self._in_rollover_window(now_utc):
            return False, "rollover-window"

        spread_limit = self._spread_limit_for(instrument)
        if spread_limit is not None and spread_pips is not None:
            if spread_pips > spread_limit:
                return False, "spread-too-wide"

        return True, "ok"

    def _atr_multipliers_for(self, instrument: Optional[str]) -> Tuple[float, float]:
        default_sl = self.sl_atr_mult if self.sl_atr_mult > 0 else DEFAULT_ATR_STOP_MULT
        default_tp = self.tp_atr_mult if self.tp_atr_mult > 0 else 1.0
        if not instrument:
            return default_sl, default_tp
        override = self.instrument_atr_multipliers.get(instrument, {})
        sl_mult = self._coerce_positive(override.get("sl_atr_mult", default_sl), cap=DEFAULT_ATR_STOP_MULT)
        tp_mult = self._coerce_positive(override.get("tp_atr_mult", default_tp), cap=None)
        return sl_mult, tp_mult

    def sl_distance_from_atr(self, atr_price_units: Optional[float], instrument: Optional[str] = None) -> float:
        if atr_price_units is None or atr_price_units <= 0:
            return 0.0
        sl_mult, _ = self._atr_multipliers_for(instrument)
        return max(0.0, atr_price_units * sl_mult)

    def register_entry(self, now_utc: datetime, instrument: str) -> None:
        self.state.last_entry_ts_utc = now_utc
        self.state.last_entry_per_instrument[instrument] = now_utc
        self.state.daily_entry_count = int(self.state.daily_entry_count or 0) + 1
        cooldown_minutes = self._cooldown_minutes()
        if cooldown_minutes > 0:
            self.state.cooldown_until[instrument] = now_utc + timedelta(minutes=cooldown_minutes)
        self._save_state()

    def register_exit(self, realized_pl: float) -> None:
        self.state.daily_realized_pl += float(realized_pl)
        self.state.weekly_realized_pl += float(realized_pl)
        self.state.last_trades.append(
            {"ts": datetime.now(timezone.utc).isoformat(), "pl": float(realized_pl)}
        )
        if len(self.state.last_trades) > 50:
            self.state.last_trades = self.state.last_trades[-50:]
        self._save_state()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _remember_equity(self, equity: Optional[float]) -> None:
        """Track the latest sanitized equity value for adjustment detection."""
        self._last_equity_seen = _sanitize_equity(equity)

    def startup_daily_reset(self, equity: Optional[float], *, open_positions_count: int = 0) -> None:
        """
        Reset daily baselines after startup equity retrieval.

        Demo: always apply.
        Live: apply only at startup (assumed no open trades) and only once.
        """

        if self._startup_reset_done:
            return

        if not self.demo_mode and self.mode != "live":
            # Only demo or live need the startup hygiene at present.
            return

        valid_equity = _sanitize_equity(equity)
        if valid_equity is None:
            return

        if self.mode == "live" and open_positions_count > 0 and not self.demo_mode:
            # Avoid altering live baselines mid-trade
            return

        self.state.day_start_equity = valid_equity
        self.state.day_start_equity_utc = valid_equity
        self.state.peak_equity_today = valid_equity
        self.state.daily_pl = 0.0
        self.state.drawdown_pct = 0.0
        self.state.daily_profit_cap_hit = False
        self.state.daily_loss_cap_hit = False
        self._last_equity_seen = valid_equity
        self._startup_reset_done = True
        self._save_state()

        mode_label = "demo" if self.demo_mode else "live"
        print(
            f"[STARTUP-RESET][WARN] mode={mode_label} equity={valid_equity:.2f}; daily baselines reset.",
            flush=True,
        )

    def _maybe_shift_baselines_for_adjustment(
        self, equity: float, open_positions_count: Optional[int]
    ) -> None:
        """
        Detect large balance adjustments (deposits/withdrawals/demo top-ups) when
        there are no open trades, and shift day/week baselines to keep caps honest.
        """

        valid_equity = _sanitize_equity(equity)
        if valid_equity is None:
            self._last_equity_seen = None
            return

        if self._last_equity_seen is None:
            self._last_equity_seen = valid_equity
            return

        if open_positions_count is None:
            self._last_equity_seen = valid_equity
            return

        delta = valid_equity - self._last_equity_seen
        ref = max(abs(self._last_equity_seen), 1e-9)
        looks_like_adjustment = (
            abs(delta) >= self.equity_adjustment_abs
            and (abs(delta) / ref) >= self.equity_adjustment_pct
        )

        if looks_like_adjustment and open_positions_count == 0:
            if self.state.day_start_equity is not None:
                self.state.day_start_equity += delta
            if self.state.day_start_equity_utc is not None:
                self.state.day_start_equity_utc += delta
            if self.state.week_start_equity is not None:
                self.state.week_start_equity += delta
            if self.state.peak_equity is not None:
                self.state.peak_equity = max(self.state.peak_equity + delta, valid_equity)
            else:
                self.state.peak_equity = valid_equity
            print(
                f"[EQUITY-ADJUST][WARN] Detected balance adjustment delta={delta:.2f}; shifted baselines.",
                flush=True,
            )
            self._last_equity_seen = valid_equity
            self._save_state()
            return

        self._last_equity_seen = valid_equity

    def _rollover(self, now_utc: datetime, equity: float) -> None:
        awst_now = now_utc.astimezone(AWST)
        day_id = awst_now.strftime("%Y-%m-%d")
        iso_cal = awst_now.isocalendar()
        week_id = f"{iso_cal.year}-W{iso_cal.week:02d}"
        utc_day_id = now_utc.strftime("%Y-%m-%d")
        changed = False
        valid_equity = _sanitize_equity(equity)

        # Track UTC day start equity for demo profit cap
        if valid_equity is None:
            if self.state.day_id_utc != utc_day_id:
                self.state.day_id_utc = utc_day_id
                self.state.day_start_equity_utc = None
                self.state.daily_profit_cap_hit = False
                changed = True
        else:
            if self.state.day_id_utc != utc_day_id:
                self.state.day_id_utc = utc_day_id
                self.state.day_start_equity_utc = valid_equity
                self.state.daily_profit_cap_hit = False
                print(
                    f"[PROFIT-CAP] UTC day start equity set to {valid_equity:.2f}",
                    flush=True,
                )
                changed = True
            elif self.state.day_start_equity_utc is None:
                self.state.day_start_equity_utc = valid_equity
                print(
                    f"[PROFIT-CAP] UTC day start equity set to {valid_equity:.2f}",
                    flush=True,
                )
                changed = True

        if valid_equity is None:
            prev_day_id = self.state.day_id
            if prev_day_id != day_id:
                self.state.day_id = day_id
                self.state.day_start_equity = None
                if prev_day_id is not None:
                    self.state.daily_entry_count = 0
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

        prev_day_id = self.state.day_id
        if prev_day_id != day_id:
            self.state.day_id = day_id
            self.state.day_start_equity = valid_equity
            self.state.daily_realized_pl = 0.0
            if prev_day_id is not None:
                self.state.daily_entry_count = 0
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

    def _breached_daily_profit_cap(self, equity: float) -> bool:
        if not self.demo_mode:
            return False
        if self.daily_profit_target_usd <= 0:
            return False
        if self.state.day_start_equity_utc is None:
            return False
        valid_equity = _sanitize_equity(equity)
        if valid_equity is None:
            return False
        if self.state.daily_profit_cap_hit:
            return True
        profit = valid_equity - self.state.day_start_equity_utc
        if profit >= self.daily_profit_target_usd:
            self.state.daily_profit_cap_hit = True
            print(
                f"[PROFIT-CAP] Daily profit target hit ({profit:.2f} >= {self.daily_profit_target_usd:.2f}); blocking new entries",
                flush=True,
            )
            self._save_state()
            return True
        return False

    def _breached_max_drawdown(self, equity: float) -> bool:
        if self.max_drawdown_cap_pct <= 0:
            return False
        if self.state.peak_equity is None:
            return False
        drawdown = max(self.state.peak_equity - equity, 0.0)
        return drawdown >= self.state.peak_equity * self.max_drawdown_cap_pct

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

    @staticmethod
    def _coerce_positive(value: object, *, cap: Optional[float] = None) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return 0.0
        if number < 0:
            number = 0.0
        if cap is not None:
            return min(number, cap)
        return number

    def _cooldown_minutes(self) -> int:
        minutes = self._granularity_minutes(self.timeframe)
        if minutes <= 0:
            return 0
        return max(0, int(self.cooldown_candles * minutes))

    @staticmethod
    def _granularity_minutes(timeframe: str) -> int:
        tf = (timeframe or "").upper()
        if tf.startswith("M"):
            try:
                return int(tf.replace("M", "", 1))
            except ValueError:
                return 0
        if tf.startswith("H"):
            try:
                return int(tf.replace("H", "", 1)) * 60
            except ValueError:
                return 0
        if tf.startswith("D"):
            return 24 * 60
        return 0

    def tp_distance_from_atr(self, atr_price_units: Optional[float], instrument: Optional[str] = None) -> float:
        if atr_price_units is None or atr_price_units <= 0:
            return 0.0
        _, tp_mult = self._atr_multipliers_for(instrument)
        return max(0.0, atr_price_units * tp_mult)

    def _update_peak_equity(self, equity: float) -> None:
        valid = _sanitize_equity(equity)
        if valid is None:
            return
        if self.state.peak_equity is None or valid > self.state.peak_equity:
            self.state.peak_equity = valid
            self._save_state()
