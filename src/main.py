from __future__ import annotations

import asyncio
import json
import os
import sys
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from app.broker import Broker
from app.health import watchdog
from src.decision_engine import DEFAULT_INSTRUMENTS, DecisionEngine, Evaluation
from src.risk_manager import RiskManager
from src import session_filter
from src import position_sizer
from src.projector import project_market
from src.risk_setup import (
    build_profit_protection,
    build_risk_manager,
    resolve_state_dir,
)

VERSION = "v1.6.1"

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "defaults.json"
DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR = resolve_state_dir(DEFAULT_DATA_DIR)


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
mode_env = os.getenv("MODE", config.get("mode", "paper")).lower()
config["mode"] = "paper" if mode_env == "demo" else mode_env
aggressive_mode = _as_bool(os.getenv("AGGRESSIVE_MODE", config.get("aggressive_mode", False)))
risk_tf_minutes = _granularity_minutes(config["timeframe"])
risk_cooldown_candles = int(os.getenv("COOLDOWN_CANDLES", config.get("cooldown_candles", 9)))
risk_config = config.get("risk", {}) or {}
aggressive_max_hold_minutes = float(os.getenv("AGGRESSIVE_MAX_HOLD_MINUTES", config.get("aggressive_max_hold_minutes", 45)))
aggressive_max_loss_usd = float(os.getenv("AGGRESSIVE_MAX_LOSS_USD", config.get("aggressive_max_loss_usd", 5.0)))
aggressive_max_loss_atr_mult = float(os.getenv("AGGRESSIVE_MAX_LOSS_ATR_MULT", config.get("aggressive_max_loss_atr_mult", 1.2)))
trailing_config = config.get("trailing", {}) or {}
trail_use_pips = _as_bool(os.getenv("TRAIL_USE_PIPS", trailing_config.get("use_pips", True)))
trail_arm_pips = float(os.getenv("TRAIL_ARM_PIPS", trailing_config.get("arm_pips", 8.0)))
trail_giveback_pips = float(os.getenv("TRAIL_GIVEBACK_PIPS", trailing_config.get("giveback_pips", 4.0)))
trail_arm_usd = float(os.getenv("TRAIL_ARM_USD", trailing_config.get("arm_usd", 3.0)))
trail_giveback_usd = float(os.getenv("TRAIL_GIVEBACK_USD", trailing_config.get("giveback_usd", 0.5)))
be_arm_pips = float(os.getenv("BE_ARM_PIPS", trailing_config.get("be_arm_pips", 6.0)))
be_offset_pips = float(os.getenv("BE_OFFSET_PIPS", trailing_config.get("be_offset_pips", 1.0)))
min_check_interval_sec = float(os.getenv("MIN_CHECK_INTERVAL_SEC", trailing_config.get("min_check_interval_sec", 0.0)))
trailing_config = {
    "arm_pips": trail_arm_pips,
    "giveback_pips": trail_giveback_pips,
    "arm_usd": trail_arm_usd,
    "giveback_usd": trail_giveback_usd,
    "use_pips": trail_use_pips,
    "be_arm_pips": be_arm_pips,
    "be_offset_pips": be_offset_pips,
    "min_check_interval_sec": min_check_interval_sec,
}
config["trailing"] = trailing_config
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
config["aggressive_max_loss_usd"] = aggressive_max_loss_usd
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


profit_guard = build_profit_protection(
    mode_env,
    broker,
    aggressive_mode,
    trailing=trailing_config,
    time_stop=config["time_stop"],
)


def _startup_checks() -> None:
    broker.connectivity_check()


def _open_trades_state() -> List[Dict]:
    try:
        return broker.list_open_trades()
    except AttributeError:
        # Older broker implementations may not yet expose list_open_trades.
        return []


def _trade_identifier(trade: Dict) -> str | None:
    try:
        return profit_guard._trade_id(trade)  # type: ignore[attr-defined]
    except Exception:
        return None


def _should_place_trade(open_trades: List[Dict], evaluation: Evaluation) -> bool:
    if evaluation.signal not in {"BUY", "SELL"}:
        return False
    if not evaluation.market_active:
        return False

    max_open = int(config.get("max_open_trades", 1))
    if len(open_trades) >= max_open:
        print(
            f"[TRADE] Skipping {evaluation.instrument} signal due to max open trades {max_open}",
            flush=True,
        )
        return False

    active_instruments = {trade.get("instrument") for trade in open_trades if isinstance(trade, dict)}
    if evaluation.instrument in active_instruments:
        print(
            f"[TRADE] Skipping {evaluation.instrument} signal; trade already open",
            flush=True,
        )
        return False

    if evaluation.reason == "cooldown":
        print(
            f"[TRADE] Skipping {evaluation.instrument} signal; instrument cooling down",
            flush=True,
        )
        return False

    return True


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
    print(
        f"[HEARTBEAT] {ts_local} instruments={len(config.get('instruments', []))} "
        f"equity={equity:.2f} daily_pl={risk.state.daily_realized_pl:.2f} "
        f"drawdown_pct={dd_pct_str} open_trades={open_count}",
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

        session_mode = "demo" if getattr(risk, "demo_mode", False) else mode_env
        entries_allowed = session_filter.is_entry_session(now_utc, mode=session_mode)
        if not entries_allowed:
            ts = now_utc.astimezone(timezone.utc).strftime("%H:%M")
            print(f"[FILTER] Entries paused (off-session) now_utc={ts}", flush=True)

        for evaluation in evaluations:
            _log_projector(evaluation, now_utc)
            if not _should_place_trade(open_trades, evaluation):
                continue

            diagnostics = evaluation.diagnostics or {}
            spread_pips = None
            try:
                spread_pips = broker.current_spread(evaluation.instrument)
            except AttributeError:
                spread_pips = None

            ok_to_open, reason = risk.should_open(
                now_utc,
                equity,
                open_trades,
                evaluation.instrument,
                spread_pips,
            )
            if not ok_to_open:
                print(
                    f"[TRADE] Skipping {evaluation.instrument} due to {reason}",
                    flush=True,
                )
                continue

            if not entries_allowed:
                continue

            if getattr(risk, "demo_mode", False) and now_utc.weekday() >= 5:
                print(
                    "[WEEKEND] Entry blocked - weekend lock active (UTC Saturday/Sunday)",
                    flush=True,
                )
                continue

            atr_val = diagnostics.get("atr")
            sl_distance = risk.sl_distance_from_atr(atr_val, instrument=evaluation.instrument)
            tp_distance = risk.tp_distance_from_atr(atr_val, instrument=evaluation.instrument)
            entry_price = diagnostics.get("close")

            trend_ok, ema_fast, ema_slow = _trend_aligned(evaluation.signal, diagnostics)
            if not trend_ok:
                print(
                    f"[FILTER] Trend misaligned {evaluation.instrument} signal={evaluation.signal} "
                    f"ema50={ema_fast:.5f} ema200={ema_slow:.5f}",
                    flush=True,
                )
                continue

            xau_blocked, rsi_val, atr_ratio = _xau_blocked(
                evaluation.signal, evaluation.instrument, diagnostics
            )
            if xau_blocked:
                print(
                    f"[FILTER] XAU_USD blocked {evaluation.signal}: rsi={rsi_val:.2f} atr_ratio={atr_ratio:.2f}",
                    flush=True,
                )
                continue

            units = position_sizer.units_for_risk(
                equity,
                entry_price or 0.0,
                sl_distance,
                risk.risk_per_trade_pct,
            )
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
                engine.mark_trade(evaluation.instrument)
                open_trades.append({"instrument": evaluation.instrument})
                risk.register_entry(now_utc, evaluation.instrument)
                sl_price, tp_price = _calc_exit_prices(evaluation.signal, entry_price, sl_distance, tp_distance)
                ticket = _order_ticket(result)
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
