from __future__ import annotations

import asyncio
import json
import os
import sys
import math
import uuid
from flask import Flask, jsonify
import threading


def send_snapshot(user: str, equity: float) -> None:
    """
    Safe no-op snapshot sender.
    Prevents runtime crashes when external snapshot transport
    is not configured.
    """
    return


BOT_STATE = {
    "status": "starting",
    "equity": None,
    "drawdown_pct": None,
    "open_trades": 0,
    "last_heartbeat": None,
}





from datetime import datetime, timedelta, timezone
from typing import Dict, List
from pathlib import Path

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from app.broker import Broker
from app.health import watchdog
from src.decision_engine import DEFAULT_INSTRUMENTS, DecisionEngine, Evaluation
from src.risk_manager import RiskManager
import src.profit_protection as profit_protection
from src.profit_protection import ProfitProtection
from src import orb, session_filter
from src import position_sizer




from src.projector import project_market
from src.risk_setup import (
    build_profit_protection,
    build_risk_manager,
    resolve_state_dir,
)
from src.trade_journal import TradeJournal, default_journal_path

VERSION = "v1.6.1"

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "defaults.json"
DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR = resolve_state_dir(DEFAULT_DATA_DIR)
journal = TradeJournal(default_journal_path(DATA_DIR))
MINI_RUN_TAG = "MINI_RUN"


def load_config(path: Path = CONFIG_PATH) -> Dict:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        print(f"[CONFIG] Invalid JSON at {path}; using empty config", flush=True)
        return {}


def _parse_instruments(value: str | List[str] | None) -> List[str] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return [str(v).strip().upper() for v in value if str(v).strip()]
    tokens = [tok.strip().upper() for tok in str(value).replace(";", ",").split(",")]
    return [tok for tok in tokens if tok]


def _instrument_env_override() -> tuple[bool, str | List[str] | None]:
    for key in ("INSTRUMENTS", "INSTRUMENT"):
        if key in os.environ:
            return True, os.environ.get(key)
    return False, None


def _resolve_instruments_config(config: Dict) -> List[str]:
    has_env, env_value = _instrument_env_override()
    raw_value = env_value if has_env else config.get("instruments")
    parsed = _parse_instruments(raw_value)
    if parsed is None:
        return list(DEFAULT_INSTRUMENTS)
    return parsed


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


def _as_bool(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "y"}
    return bool(value)


def _coerce_float(value: object, fallback: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _build_trailing_config(config: Dict) -> Dict:
    trailing_config = config.get("trailing", {}) or {}
    trail_use_pips = False
    trail_arm_pips = _coerce_float(os.getenv("TRAIL_ARM_PIPS"), trailing_config.get("arm_pips", 0.0))
    trail_giveback_pips = _coerce_float(os.getenv("TRAIL_GIVEBACK_PIPS"), trailing_config.get("giveback_pips", 0.0))

    arm_default = trailing_config.get("arm_ccy", trailing_config.get("arm_usd", profit_protection.ARM_AT_CCY))
    giveback_default = trailing_config.get("giveback_ccy", trailing_config.get("giveback_usd", profit_protection.GIVEBACK_CCY))
    arm_env = os.getenv("TRAIL_ARM_CCY")
    if arm_env is None:
        arm_env = os.getenv("TRAIL_ARM_USD")
    giveback_env = os.getenv("TRAIL_GIVEBACK_CCY")
    if giveback_env is None:
        giveback_env = os.getenv("TRAIL_GIVEBACK_USD")
    trail_arm_ccy = _coerce_float(arm_env, arm_default)
    trail_giveback_ccy = _coerce_float(giveback_env, giveback_default)

    be_arm_pips = _coerce_float(os.getenv("BE_ARM_PIPS"), trailing_config.get("be_arm_pips", 0.0))
    be_offset_pips = _coerce_float(os.getenv("BE_OFFSET_PIPS"), trailing_config.get("be_offset_pips", 0.0))

    min_check_env = os.getenv("MIN_CHECK_INTERVAL_SEC")
    if min_check_env is None:
        min_check_env = os.getenv("TRAIL_MIN_CHECK_INTERVAL")
    min_check_interval_sec = _coerce_float(
        min_check_env,
        trailing_config.get("min_check_interval_sec", 0.0),
    )

    soft_scalp_source = os.getenv("SOFT_SCALP_MODE")
    if soft_scalp_source is None:
        soft_scalp_source = trailing_config.get("soft_scalp_mode", False)
    soft_scalp_mode = _as_bool(soft_scalp_source)

    resolved = {
        "arm_pips": trail_arm_pips,
        "giveback_pips": trail_giveback_pips,
        "arm_ccy": trail_arm_ccy,
        "giveback_ccy": trail_giveback_ccy,
        "use_pips": trail_use_pips,
        "be_arm_pips": be_arm_pips,
        "be_offset_pips": be_offset_pips,
        "min_check_interval_sec": min_check_interval_sec,
        "soft_scalp_mode": soft_scalp_mode,
    }

    warnings: list[str] = []
    if trail_arm_ccy <= 0:
        warnings.append("arm_ccy")
    if trail_giveback_ccy <= 0:
        warnings.append("giveback_ccy")
    for field in warnings:
        value = resolved.get(field)
        print(
            f"[TRAIL][WARN] Non-positive trailing threshold {field}={value}; check environment/config",
            flush=True,
        )

    return resolved


def _calc_exit_prices(signal: str, entry_price: float | None, sl_distance: float | None, tp_distance: float | None) -> tuple[float | None, float | None]:
    if entry_price is None:
        return None, None
    try:
        entry_val = float(entry_price)
    except (TypeError, ValueError):
        return None, None

    side = (signal or "").upper()
    sl_price = None
    tp_price = None

    if sl_distance is not None and sl_distance > 0:
        sl_price = entry_val - sl_distance if side == "BUY" else entry_val + sl_distance
    if tp_distance is not None and tp_distance > 0:
        tp_price = entry_val + tp_distance if side == "BUY" else entry_val - tp_distance
    return sl_price, tp_price


def _order_ticket(result: Dict) -> str | None:
    if not isinstance(result, dict):
        return None
    resp = result.get("response", {}) or {}
    tx_keys = (
        "orderCreateTransaction",
        "orderFillTransaction",
        "takeProfitOrderTransaction",
        "stopLossOrderTransaction",
    )
    for key in tx_keys:
        tx = resp.get(key) or {}
        for id_key in ("tradeOpenedID", "id", "orderID", "orderFillTransactionID"):
            if tx.get(id_key) is not None:
                return str(tx[id_key])
    last_id = resp.get("lastTransactionID") or result.get("order_id") or result.get("id")
    return str(last_id) if last_id is not None else None


config = load_config()
config["merge_default_instruments"] = _as_bool(
    os.getenv("MERGE_DEFAULT_INSTRUMENTS", config.get("merge_default_instruments", False))
)
config["instruments"] = _resolve_instruments_config(config)
config["timeframe"] = os.getenv("TIMEFRAME", config.get("timeframe", "M5"))
config["use_macd_confirmation"] = _as_bool(
    os.getenv("USE_MACD_CONFIRMATION", config.get("use_macd_confirmation", False))
)
config["session_mode"] = (os.getenv("SESSION_MODE") or config.get("session_mode") or "SOFT").upper()  # MINI-RUN: default to SOFT for boundary-friendly entries
config["session_off_session_vol_ratio"] = float(
    os.getenv("SESSION_OFF_SESSION_VOL_RATIO", config.get("session_off_session_vol_ratio", 1.25))
)
config["session_off_session_risk_scale"] = float(
    os.getenv("SESSION_OFF_SESSION_RISK_SCALE", config.get("session_off_session_risk_scale", 0.5))
)
config["xau_atr_guard_ratio"] = float(
    os.getenv("XAU_ATR_GUARD_RATIO", config.get("xau_atr_guard_ratio", 1.8))
)
config["xau_atr_guard_action"] = (os.getenv("XAU_ATR_GUARD_ACTION") or config.get("xau_atr_guard_action") or "skip").lower()
config["xau_atr_guard_size_scale"] = float(
    os.getenv("XAU_ATR_GUARD_SIZE_SCALE", config.get("xau_atr_guard_size_scale", 0.5))
)
mode_env = os.getenv("MODE", config.get("mode", "paper")).lower()
config["mode"] = "paper" if mode_env == "demo" else mode_env
aggressive_mode = _as_bool(os.getenv("AGGRESSIVE_MODE", config.get("aggressive_mode", False)))
risk_tf_minutes = _granularity_minutes(config["timeframe"])
risk_cooldown_candles = int(os.getenv("COOLDOWN_CANDLES", config.get("cooldown_candles", 9)))
risk_config = config.get("risk", {}) or {}
aggressive_max_hold_minutes = float(os.getenv("AGGRESSIVE_MAX_HOLD_MINUTES", config.get("aggressive_max_hold_minutes", 45)))
aggressive_max_loss_ccy = float(
    os.getenv(
        "AGGRESSIVE_MAX_LOSS_CCY",
        os.getenv("AGGRESSIVE_MAX_LOSS_USD", config.get("aggressive_max_loss_ccy", config.get("aggressive_max_loss_usd", 5.0))),
    )
)
aggressive_max_loss_atr_mult = float(os.getenv("AGGRESSIVE_MAX_LOSS_ATR_MULT", config.get("aggressive_max_loss_atr_mult", 1.2)))
trailing_config = _build_trailing_config(config)
config["trailing"] = trailing_config
print(f"[CONFIG] trailing resolved={trailing_config}", flush=True)
time_stop_cfg = config.get("time_stop", {}) or {}
time_stop_minutes = float(os.getenv("TIME_STOP_MINUTES", time_stop_cfg.get("minutes", 90)))
time_stop_min_pips = float(os.getenv("TIME_STOP_MIN_PIPS", time_stop_cfg.get("min_pips", 2.0)))
time_stop_xau_atr_mult = float(os.getenv("TIME_STOP_XAU_ATR_MULT", time_stop_cfg.get("xau_atr_mult", 0.35)))
config["time_stop"] = {
    "minutes": time_stop_minutes,
    "min_pips": time_stop_min_pips,
    "xau_atr_mult": time_stop_xau_atr_mult,
}

# Baseline risk defaults
risk_config.setdefault("risk_per_trade_pct", float(os.getenv("MAX_RISK_PER_TRADE", risk_config.get("risk_per_trade_pct", 0.005))))
sl_default = risk_config.get("sl_atr_mult", risk_config.get("atr_stop_mult", config.get("sl_atr_mult", 1.2)))
tp_default = risk_config.get("tp_atr_mult", risk_config.get("tp_rr_multiple", config.get("tp_atr_mult", 1.0)))
risk_config["sl_atr_mult"] = float(os.getenv("SL_ATR_MULT", os.getenv("ATR_STOP_MULT", sl_default)))
risk_config["tp_atr_mult"] = float(os.getenv("TP_ATR_MULT", tp_default))
risk_config.setdefault("instrument_atr_multipliers", risk_config.get("instrument_atr_multipliers", config.get("instrument_atr_multipliers", {})))
risk_config.setdefault("tp_rr_multiple", risk_config["tp_atr_mult"])
risk_config.setdefault("cooldown_candles", int(os.getenv("COOLDOWN_CANDLES", risk_config.get("cooldown_candles", 9))))
env_max_positions = os.getenv("MAX_CONCURRENT_POSITIONS") or os.getenv("MAX_OPEN_TRADES")
max_positions_default = risk_config.get("max_concurrent_positions", config.get("max_open_trades", 3))
risk_config["max_concurrent_positions"] = int(env_max_positions or max_positions_default or 3)
risk_config.setdefault("daily_loss_cap_pct", float(os.getenv("DAILY_LOSS_CAP_PCT", risk_config.get("daily_loss_cap_pct", 0.02))))
risk_config.setdefault("weekly_loss_cap_pct", float(os.getenv("WEEKLY_LOSS_CAP_PCT", risk_config.get("weekly_loss_cap_pct", 0.03))))
risk_config.setdefault("max_drawdown_cap_pct", float(os.getenv("MAX_DRAWDOWN_CAP_PCT", risk_config.get("max_drawdown_cap_pct", 0.10))))
risk_config.setdefault("daily_profit_target_usd", float(os.getenv("DAILY_PROFIT_TARGET_USD", risk_config.get("daily_profit_target_usd", 5.0))))
risk_config["max_trades_per_day"] = int(os.getenv("MAX_TRADES_PER_DAY", risk_config.get("max_trades_per_day", 0) or 0))

if aggressive_mode:
    # Loosen guardrails for higher throughput/risk
    risk_config["risk_per_trade_pct"] = float(os.getenv("AGGRESSIVE_RISK_PCT", risk_config.get("risk_per_trade_pct", 0.01)))
    risk_config["max_concurrent_positions"] = int(os.getenv("AGGRESSIVE_MAX_POSITIONS", risk_config.get("max_concurrent_positions", 3)))
    risk_config["cooldown_candles"] = int(os.getenv("AGGRESSIVE_COOLDOWN_CANDLES", 3))
    risk_config["daily_loss_cap_pct"] = float(os.getenv("AGGRESSIVE_DAILY_LOSS_CAP_PCT", 0.03))
    risk_config["weekly_loss_cap_pct"] = float(os.getenv("AGGRESSIVE_WEEKLY_LOSS_CAP_PCT", 0.06))
    risk_config["max_drawdown_cap_pct"] = float(os.getenv("AGGRESSIVE_MAX_DRAWDOWN_CAP_PCT", 0.20))
    risk_config["tp_rr_multiple"] = float(os.getenv("AGGRESSIVE_TP_RR", 2.0))
    # Remove profit cap in aggressive/demo and widen take-profit allowance
    risk_config["daily_profit_target_usd"] = float(os.getenv("AGGRESSIVE_DAILY_PROFIT_CAP", 0.0))
    risk_cooldown_candles = risk_config["cooldown_candles"]

config["cooldown_candles"] = risk_cooldown_candles
config["cooldown_minutes"] = risk_tf_minutes * risk_cooldown_candles if risk_tf_minutes else config.get("cooldown_minutes", 0)
config["max_open_trades"] = int(os.getenv("MAX_OPEN_TRADES", risk_config.get("max_concurrent_positions", config.get("max_open_trades", 3))))
risk_config["timeframe"] = config["timeframe"]
config["aggressive_mode"] = aggressive_mode
config["aggressive_max_hold_minutes"] = aggressive_max_hold_minutes
config["aggressive_max_loss_ccy"] = aggressive_max_loss_ccy
config["aggressive_max_loss_atr_mult"] = aggressive_max_loss_atr_mult
config["risk"] = risk_config

# Abort if live is requested (demo/practice only)
oanda_env = (os.getenv("OANDA_ENV") or "practice").lower()
if oanda_env == "live" or config["mode"] == "live":
    print("[STARTUP] Live mode is disabled for this deployment. Exiting.", flush=True)
    sys.exit(1)

broker = Broker()
engine = DecisionEngine(config)
risk = build_risk_manager(
    config,
    mode=config["mode"],
    demo_mode=(mode_env == "demo"),
    state_dir=DATA_DIR,
)

suppression_counters = {
    "signals_generated": 0,
    "signals_executed": 0,
    "blocked_off_session": 0,
    "blocked_risk": 0,
    "blocked_max_positions": 0,
    "blocked_spread": 0,
}


def _profit_guard_for_mode(mode: str, broker_instance: Broker) -> ProfitProtection:
    return build_profit_protection(
        mode,
        broker_instance,
        aggressive=aggressive_mode,
        trailing=trailing_config,
        time_stop=config["time_stop"],
        journal=journal,
    )


profit_guard = _profit_guard_for_mode(mode_env, broker)


def _startup_checks() -> None:
    broker.connectivity_check()
    try:
        equity = broker.account_equity()
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[STARTUP-RESET][WARN] Unable to fetch equity for reset: {exc}", flush=True)
        return

    try:
        open_trades = _open_trades_state()
        open_count = len(open_trades or [])
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[STARTUP-RESET][WARN] Unable to inspect open trades: {exc}", flush=True)
        open_count = 0

    risk.startup_daily_reset(equity, open_positions_count=open_count)


def _open_trades_state() -> List[Dict]:
    try:
        return broker.list_open_trades()
    except AttributeError:
        # Older broker implementations may not yet expose list_open_trades.
        return []
    except Exception as exc:
        print(f"[TRADE][WARN] Unable to refresh open trades: {exc}", flush=True)
        return []


def _trade_identifier(trade: Dict) -> str | None:
    try:
        return profit_guard._trade_id(trade)  # type: ignore[attr-defined]
    except Exception:
        return None


def _should_place_trade(open_trades: List[Dict], evaluation: Evaluation) -> tuple[bool, str | None]:
    if evaluation.signal not in {"BUY", "SELL"}:
        return False, "non-entry"
    if not evaluation.market_active:
        return False, "inactive-market"

    max_open = int(config.get("max_open_trades", 1))
    if len(open_trades) >= max_open:
        print(
            f"[TRADE] Skipping {evaluation.instrument} signal due to max open trades {max_open}",
            flush=True,
        )
        return False, "max-open"

    active_instruments = {trade.get("instrument") for trade in open_trades if isinstance(trade, dict)}
    if evaluation.instrument in active_instruments:
        print(
            f"[TRADE] Skipping {evaluation.instrument} signal; trade already open",
            flush=True,
        )
        return False, "duplicate"

    if evaluation.reason == "cooldown":
        print(
            f"[TRADE] Skipping {evaluation.instrument} signal; instrument cooling down",
            flush=True,
        )
        return False, "cooldown"

    return True, None


def _instrument_open_on_broker(instrument: str) -> bool:
    """Cross-check broker state to prevent duplicate exposure."""

    try:
        trades = broker.list_open_trades()
    except Exception as exc:
        print(f"[TRADE][WARN] Unable to verify broker positions for {instrument}: {exc}", flush=True)
        return False

    for trade in trades or []:
        if trade.get("instrument") == instrument:
            return True
    return False


def _trend_aligned(signal: str, diagnostics: Dict[str, float] | None) -> tuple[bool, float, float]:
    if diagnostics is None:
        return False, math.nan, math.nan
    ema_fast = diagnostics.get("ema_trend_fast", math.nan)
    ema_slow = diagnostics.get("ema_trend_slow", math.nan)
    if signal == "BUY":
        return ema_fast > ema_slow, ema_fast, ema_slow
    if signal == "SELL":
        return ema_fast < ema_slow, ema_fast, ema_slow
    return False, ema_fast, ema_slow


def _xau_blocked(signal: str, instrument: str, diagnostics: Dict[str, float] | None) -> tuple[bool, float, float]:
    if instrument != "XAU_USD" or diagnostics is None:
        return False, math.nan, math.nan
    rsi = diagnostics.get("rsi", math.nan)
    atr_current = diagnostics.get("atr", math.nan)
    atr_baseline = diagnostics.get("atr_baseline_50", math.nan)
    if atr_baseline and not math.isnan(atr_baseline) and atr_baseline > 0:
        atr_ratio = atr_current / atr_baseline
    else:
        atr_ratio = math.nan

    if signal == "SELL" and not math.isnan(rsi) and rsi < 18 and atr_ratio > 1.15:
        return True, rsi, atr_ratio
    if signal == "BUY" and not math.isnan(rsi) and rsi > 82 and atr_ratio > 1.15:
        return True, rsi, atr_ratio
    return False, rsi, atr_ratio


def _xau_guard(
    signal: str,
    instrument: str,
    diagnostics: Dict[str, float] | None,
    *,
    guard_ratio: float,
    guard_action: str,
    guard_scale: float,
) -> tuple[bool, float, str, float, float]:
    if instrument != "XAU_USD" or diagnostics is None:
        return False, 1.0, "n/a", math.nan, math.nan
    try:
        atr = float(diagnostics.get("atr", math.nan))
        atr_baseline = float(diagnostics.get("atr_baseline_50", math.nan))
    except (TypeError, ValueError):
        return False, 1.0, "invalid-atr", math.nan, math.nan

    if math.isnan(atr) or math.isnan(atr_baseline) or atr_baseline <= 0:
        return False, 1.0, "missing-atr", atr, math.nan
    atr_ratio = atr / atr_baseline
    if atr_ratio <= guard_ratio:
        return False, 1.0, "within-guard", atr, atr_ratio

    action = (guard_action or "skip").lower()
    if action == "scale":
        return False, max(guard_scale, 0.1), "scale", atr, atr_ratio
    return True, 0.0, "skip", atr, atr_ratio


def _macd_confirmation_enabled() -> bool:
    return _as_bool(config.get("use_macd_confirmation", False))


def _macd_confirms(signal: str, diagnostics: Dict[str, float] | None) -> tuple[bool, float, float, float]:
    if not _macd_confirmation_enabled():
        return True, math.nan, math.nan, math.nan
    if diagnostics is None:
        return False, math.nan, math.nan, math.nan

    def _coerce(value: object) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return math.nan

    macd_line = _coerce(diagnostics.get("macd_line"))
    macd_signal = _coerce(diagnostics.get("macd_signal"))
    macd_histogram = _coerce(diagnostics.get("macd_histogram"))
    macd_histogram_prev = _coerce(diagnostics.get("macd_histogram_prev"))

    if any(math.isnan(val) for val in (macd_line, macd_signal, macd_histogram)):
        return False, macd_line, macd_signal, macd_histogram

    hist_rising = False
    hist_falling = False
    if not math.isnan(macd_histogram_prev):
        hist_rising = macd_histogram > macd_histogram_prev
        hist_falling = macd_histogram < macd_histogram_prev

    if signal == "BUY":
        ok = macd_line > macd_signal and (macd_histogram >= 0 or hist_rising)
        return ok, macd_line, macd_signal, macd_histogram
    if signal == "SELL":
        ok = macd_line < macd_signal and (macd_histogram <= 0 or hist_falling)
        return ok, macd_line, macd_signal, macd_histogram
    return False, macd_line, macd_signal, macd_histogram


def _seed_opening_range_bounds(
    candles: List[Dict[str, float]],
    *,
    timeframe_minutes: int,
    session_start: datetime,
    now_utc: datetime,
    range_minutes: int = 15,
) -> tuple[float, float] | None:
    if timeframe_minutes <= 0 or not candles:
        return None
    elapsed_minutes = (now_utc - session_start).total_seconds() / 60.0
    if elapsed_minutes < 0:
        return None
    bars_since_start = int(elapsed_minutes // timeframe_minutes) + 1
    start_index = max(0, len(candles) - bars_since_start)
    opening_bars = max(1, math.ceil(range_minutes / timeframe_minutes))
    slice_end = start_index + opening_bars
    window = candles[start_index:slice_end]
    highs: List[float] = []
    lows: List[float] = []
    for candle in window:
        try:
            highs.append(float(candle["h"]))
            lows.append(float(candle["l"]))
        except (KeyError, TypeError, ValueError):
            continue
    if not highs or not lows:
        return None
    return max(highs), min(lows)


def _orb_filter(
    evaluation: Evaluation,
    session: session_filter.SessionSnapshot | None,
    now_utc: datetime,
) -> tuple[bool, str | None, orb.OpeningRange | None]:
    if session is None:
        return False, "off-session", None

    candles = evaluation.candles or []
    candle_high = candles[-1].get("h") if candles else evaluation.diagnostics.get("close") if evaluation.diagnostics else None  # type: ignore[index]
    candle_low = candles[-1].get("l") if candles else evaluation.diagnostics.get("close") if evaluation.diagnostics else None  # type: ignore[index]
    if candle_high is None or candle_low is None:
        return False, "missing-candle", None

    prev_range = orb.opening_range_for(evaluation.instrument, session)
    if prev_range is None and now_utc >= session.start_utc:
        timeframe_minutes = _granularity_minutes(config.get("timeframe", "M5"))
        seeded = _seed_opening_range_bounds(
            candles,
            timeframe_minutes=timeframe_minutes,
            session_start=session.start_utc,
            now_utc=now_utc,
        )
        if seeded:
            candle_high, candle_low = seeded

    range_before = None if prev_range is None else (prev_range.high, prev_range.low, prev_range.finalized)
    current_range = orb.update_opening_range(
        evaluation.instrument,
        session,
        candle_high=float(candle_high),
        candle_low=float(candle_low),
        now_utc=now_utc,
    )

    if range_before != (current_range.high, current_range.low, current_range.finalized):
        print(
            f"[ORB] {evaluation.instrument} session={session.session_id} "
            f"high={current_range.high:.5f} low={current_range.low:.5f} "
            f"finalized={'yes' if current_range.finalized else 'no'}",
            flush=True,
        )

    if now_utc < current_range.end_utc and not current_range.finalized:
        return False, "opening-range-forming", current_range

    close_price = evaluation.diagnostics.get("close") if evaluation.diagnostics else None
    if close_price is None:
        return False, "missing-close", current_range
    breakout_dir = orb.breakout_direction(float(close_price), current_range)
    if breakout_dir is None:
        return False, "inside-opening-range", current_range
    if breakout_dir != evaluation.signal:
        return False, f"breakout-mismatch-{breakout_dir.lower()}", current_range

    print(
        f"[ORB] Breakout {evaluation.instrument} dir={breakout_dir} close={float(close_price):.5f} "
        f"range=({current_range.low:.5f}..{current_range.high:.5f})",
        flush=True,
    )
    return True, None, current_range


def _projector_enabled() -> bool:
    return (os.getenv("ENABLE_PROJECTOR", "false") or "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
        "y",
    }


def _log_projector(evaluation: Evaluation, now_utc: datetime) -> None:
    if not _projector_enabled():
        return

    try:
        projection = project_market(
            evaluation.instrument,
            evaluation.candles or [],
            evaluation.diagnostics or {},
            now_utc,
        )
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[PROJECTOR] {evaluation.instrument} skipped error={exc}", flush=True)
        return

    ts = projection.get("timestamp") or now_utc
    if isinstance(ts, datetime):
        ts_dt = ts.astimezone(timezone.utc)
    else:
        ts_dt = now_utc
    ts_str = ts_dt.strftime("%H:%M")
    bias = projection.get("bias", "NEUTRAL")
    bias_score = projection.get("bias_score", 0.0) or 0.0
    volatility = projection.get("volatility", "NORMAL")
    confidence = projection.get("confidence", 0.0) or 0.0
    bias_score_fmt = f"{bias_score:.2f}"
    confidence_fmt = f"{int(round(confidence))}"

    range_info = projection.get("range") or {}
    low = range_info.get("low")
    high = range_info.get("high")
    range_fmt = "n/a"
    if isinstance(low, (int, float)) and isinstance(high, (int, float)):
        range_fmt = f"{low:.4f}..{high:.4f}"

    print(
        f"[PROJECTOR] {evaluation.instrument} ts={ts_str} bias={bias} "
        f"score={bias_score_fmt} conf={confidence_fmt} "
        f"vol={volatility} range={range_fmt}",
        flush=True,
    )


async def heartbeat() -> None:
    watchdog.last_heartbeat_ts = datetime.now(timezone.utc)
    ts_local = datetime.now(timezone.utc).astimezone().isoformat()
    equity = broker.account_equity()
    open_count = len(_open_trades_state())
    drawdown = None
    if risk.state.peak_equity:
        drawdown = risk.state.peak_equity - equity
    dd_pct = None
    if risk.state.peak_equity and risk.state.peak_equity > 0:
        dd_pct = drawdown / risk.state.peak_equity if drawdown is not None else None
    dd_pct_str = f"{(dd_pct * 100):.2f}" if dd_pct is not None else "n/a"

    BOT_STATE.update({
        "status": "running",
        "equity": float(equity),
        "drawdown_pct": (float(dd_pct) * 100.0) if dd_pct is not None else None,
        "open_trades": int(open_count),
        "last_heartbeat": datetime.now(timezone.utc).isoformat(),
    })

    })

    print(
        f"[HEARTBEAT] {ts_local} instruments={len(config.get('instruments', []))} "
        f"equity={equity:.2f} daily_pl={risk.state.daily_realized_pl:.2f} "
        f"drawdown_pct={dd_pct_str} open_trades={open_count}",
        flush=True,
    )
    print(
        "[SUMMARY] signals_generated={signals_generated} signals_executed={signals_executed} "
        "blocked_off_session={blocked_off_session} blocked_risk={blocked_risk} "
        "blocked_max_positions={blocked_max_positions} blocked_spread={blocked_spread}".format(
            **suppression_counters
        ),
        flush=True,
    )


async def decision_cycle() -> None:
    now_utc = datetime.now(timezone.utc)
    equity = broker.account_equity()
    try:
        risk.enforce_equity_floor(now_utc, equity, close_all_cb=broker.close_all_positions)
    except AttributeError:
        pass

    try:
        evaluations = engine.evaluate_all()
    except Exception as exc:  # pragma: no cover - defensive logging
        watchdog.record_error()
        ts = datetime.now(timezone.utc).astimezone().isoformat()
        print(f"[ERROR] {ts} decision-cycle failure error={exc}", flush=True)
        return
    else:
        open_trades = _open_trades_state()
        # --- Profit-protection rule ($3 trigger / $0.50 trail) ---
        closed_by_trail = profit_guard.process_open_trades(open_trades)
        if closed_by_trail:
            # Remove locally-tracked trades that were closed by the trailing rule
            open_trades = [
                trade
                for trade in open_trades
                if (trade.get("instrument") not in closed_by_trail)
                and (_trade_identifier(trade) not in closed_by_trail)
            ]

        session_mode = config.get("session_mode", "STRICT")
        session_snapshot = session_filter.current_session(now_utc, mode=session_mode)
        if session_snapshot:
            orb.reset_for_session(session_snapshot)

        for evaluation in evaluations:
            suppression_counters["signals_generated"] += 1
            _log_projector(evaluation, now_utc)
            diagnostics = evaluation.diagnostics or {}
            trend_ok, ema_trend_fast, ema_trend_slow = _trend_aligned(evaluation.signal, diagnostics)
            session_decision = session_filter.session_decision(
                now_utc,
                mode=session_mode,
                atr=diagnostics.get("atr"),
                atr_baseline=diagnostics.get("atr_baseline_50"),
                trend_aligned=trend_ok,
                max_off_session_vol_ratio=float(config.get("session_off_session_vol_ratio", 1.25)),
                off_session_risk_scale=float(config.get("session_off_session_risk_scale", 0.5)),
            )
            active_session = session_decision.session
            if session_decision.allowed and active_session is None:
                start_awst = now_utc.astimezone(session_filter.AWST) - timedelta(minutes=15)
                end_awst = start_awst + timedelta(minutes=15)
                active_session = session_filter.SessionSnapshot(
                    name="adhoc",
                    start_awst=start_awst,
                    end_awst=end_awst,
                    start_utc=start_awst.astimezone(timezone.utc),
                    end_utc=end_awst.astimezone(timezone.utc),
                )
                orb.reset_for_session(active_session)
            elif active_session:
                orb.reset_for_session(active_session)

            if not session_decision.allowed:
                suppression_counters["blocked_off_session"] += 1
                ts = now_utc.astimezone(timezone.utc).strftime("%H:%M")
                print(
                    f"[FILTER] Entries paused (off-session) now_utc={ts} mode={session_mode} reason={session_decision.reason}",
                    flush=True,
                )
                continue

            should_trade, skip_reason = _should_place_trade(open_trades, evaluation)
            if not should_trade:
                if skip_reason == "max-open":
                    suppression_counters["blocked_max_positions"] += 1
                continue

            # Final broker-side duplicate guard before risk checks or order submission.
            if _instrument_open_on_broker(evaluation.instrument):
                print(
                    f"[TRADE] Skipping {evaluation.instrument}; broker reports existing open position",
                    flush=True,
                )
                continue

            spread_pips = None
            try:
                spread_pips = broker.current_spread(evaluation.instrument)
            except AttributeError:
                spread_pips = None

            ok_to_open, risk_reason = risk.should_open(
                now_utc,
                equity,
                open_trades,
                evaluation.instrument,
                spread_pips,
            )
            if not ok_to_open:
                print(
                    f"[TRADE] Skipping {evaluation.instrument} due to {risk_reason}",
                    flush=True,
                )
                if risk_reason == "max-positions":
                    suppression_counters["blocked_max_positions"] += 1
                elif risk_reason == "spread-too-wide":
                    suppression_counters["blocked_spread"] += 1
                else:
                    suppression_counters["blocked_risk"] += 1
                continue

            if getattr(risk, "demo_mode", False) and now_utc.weekday() >= 5:
                print(
                    "[WEEKEND] Entry blocked - weekend lock active (UTC Saturday/Sunday)",
                    flush=True,
                )
                continue

            atr_val = diagnostics.get("atr")
            sl_distance = risk.sl_distance_from_atr(atr_val, instrument=evaluation.instrument)
            tp_distance = 0.0  # Disable standard TP to avoid interfering with account-currency profit protection
            entry_price = diagnostics.get("close")

            if not trend_ok:
                print(
                    f"[FILTER] Trend misaligned {evaluation.instrument} signal={evaluation.signal} "
                    f"ema50={ema_trend_fast:.5f} ema200={ema_trend_slow:.5f}",
                    flush=True,
                )
                continue

            xau_blocked, rsi_val, atr_ratio = _xau_blocked(
                evaluation.signal, evaluation.instrument, diagnostics
            )
            xau_guard_block, xau_risk_scale, xau_guard_reason, xau_atr_val, xau_atr_ratio = _xau_guard(
                evaluation.signal,
                evaluation.instrument,
                diagnostics,
                guard_ratio=float(config.get("xau_atr_guard_ratio", 1.8)),
                guard_action=str(config.get("xau_atr_guard_action", "skip")),
                guard_scale=float(config.get("xau_atr_guard_size_scale", 0.5)),
            )
            if xau_blocked or xau_guard_block:
                print(
                    f"[FILTER] XAU_USD blocked {evaluation.signal}: rsi={rsi_val:.2f} atr_ratio={atr_ratio:.2f} guard={xau_guard_reason} guard_atr={xau_atr_val:.5f} guard_ratio={xau_atr_ratio:.2f}",
                    flush=True,
                )
                suppression_counters["blocked_risk"] += 1
                continue

            macd_ok, macd_line, macd_signal_line, macd_histogram = _macd_confirms(
                evaluation.signal, diagnostics
            )
            if not macd_ok:
                print(
                    f"[FILTER] MACD veto {evaluation.instrument} macd={macd_line:.5f} "
                    f"signal={macd_signal_line:.5f} hist={macd_histogram:.5f}",
                    flush=True,
                )
                continue
            print(
                f"[MACD] Confirmed {evaluation.instrument} macd={macd_line:.5f} "
                f"signal={macd_signal_line:.5f} hist={macd_histogram:.5f}",
                flush=True,
            )

            orb_ok, orb_reason, current_range = _orb_filter(
                evaluation,
                active_session,
                now_utc,
            )
            if not orb_ok:
                print(
                    f"[FILTER] ORB block {evaluation.instrument} reason={orb_reason}",
                    flush=True,
                )
                continue

            units = position_sizer.units_for_risk(
                equity,
                entry_price or 0.0,
                sl_distance,
                risk.risk_per_trade_pct,
            )
            risk_scale = session_decision.risk_scale * xau_risk_scale
            if risk_scale < 1.0:
                units = int(units * max(risk_scale, 0.1))
            if units <= 0:
                print(
                    f"[TRADE] Skipping {evaluation.instrument} due to zero position size",
                    flush=True,
                )
                continue

            result = broker.place_order(
                evaluation.instrument,
                evaluation.signal,
                units,
                sl_distance=sl_distance,
                tp_distance=tp_distance,
                entry_price=entry_price,
            )
            if result.get("status") == "SENT":
                sl_price, tp_price = _calc_exit_prices(evaluation.signal, entry_price, sl_distance, tp_distance)
                ticket = _order_ticket(result) or f"local-{uuid.uuid4().hex}"
                session_id_label = "OFF_SESSION"
                if active_session:
                    session_id_label = (active_session.name or "SESSION").upper()
                elif session_decision.in_session:
                    session_id_label = "SESSION"
                indicators_snapshot = {
                    "rsi": diagnostics.get("rsi"),
                    "atr": diagnostics.get("atr"),
                    "ema50": diagnostics.get("ema_trend_fast"),
                    "ema200": diagnostics.get("ema_trend_slow"),
                    "ema_fast": diagnostics.get("ema_fast"),
                    "ema_slow": diagnostics.get("ema_slow"),
                }
                gating_flags = {
                    "session_ok": session_decision.allowed,
                    "spread_ok": risk_reason != "spread-too-wide",
                    "risk_ok": ok_to_open,
                    "trend_ok": trend_ok,
                    "xau_guard_ok": not (xau_blocked or xau_guard_block),
                }
                try:
                    journal.record_entry(
                        trade_id=ticket,
                        timestamp_utc=now_utc,
                        instrument=evaluation.instrument,
                        side=evaluation.signal,
                        units=units,
                        entry_price=entry_price,
                        stop_loss_price=sl_price,
                        take_profit_price=tp_price,
                        spread_at_entry=spread_pips,
                        session_id=session_id_label,
                        session_mode=session_decision.mode,
                        run_tag=MINI_RUN_TAG,
                        gating_flags=gating_flags,
                        indicators_snapshot=indicators_snapshot,
                    )
                except Exception:
                    # Journal failures must not block live execution.
                    pass
                engine.mark_trade(evaluation.instrument)
                open_trades.append({"instrument": evaluation.instrument, "id": ticket})
                risk.register_entry(now_utc, evaluation.instrument)
                suppression_counters["signals_executed"] += 1
                print(
                    f"[ORDER] ticket={ticket or 'n/a'} instrument={evaluation.instrument} "
                    f"sl={ 'n/a' if sl_price is None else f'{sl_price:.5f}'} "
                    f"tp={ 'n/a' if tp_price is None else f'{tp_price:.5f}'}",
                    flush=True,
                )
            else:
                print(
                    f"[TRADE] Order failed instrument={evaluation.instrument} signal={evaluation.signal}"
                    f" response={result}",
                    flush=True,
                )
    finally:
        watchdog.last_decision_ts = datetime.now(timezone.utc)


async def runner() -> None:
    _startup_checks()

    equity = broker.account_equity()
    send_snapshot("luke", equity)

    scheduler = AsyncIOScheduler()
    scheduler.add_job(heartbeat, "interval", minutes=1)
    scheduler.add_job(decision_cycle, "interval", minutes=1)
    scheduler.start()
    asyncio.create_task(watchdog.run())
    await heartbeat()
    await decision_cycle()
    while True:
        await asyncio.sleep(3600)

if __name__ == "__main__":
    asyncio.run(runner())
