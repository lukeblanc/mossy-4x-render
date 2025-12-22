from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from app.broker import Broker
from app.health import watchdog
from src.decision_engine import DecisionEngine, Evaluation
from src.risk_manager import RiskManager
from src.profit_protection import ProfitProtection
from src import position_sizer

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "defaults.json"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_config(path: Path = CONFIG_PATH) -> Dict:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        print(f"[CONFIG] Invalid JSON at {path}; using empty config", flush=True)
        return {}


def _parse_instruments(value: str | List[str] | None) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip().upper() for v in value if str(v).strip()]
    tokens = [tok.strip().upper() for tok in str(value).replace(";", ",").split(",")]
    return [tok for tok in tokens if tok]


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


config = load_config()
env_instruments = os.getenv("INSTRUMENTS") or os.getenv("INSTRUMENT")
resolved_instruments = _parse_instruments(env_instruments) or config.get("instruments") or ["EUR_USD"]
config["instruments"] = resolved_instruments
config["max_open_trades"] = int(os.getenv("MAX_OPEN_TRADES", config.get("max_open_trades", 2)))
config["timeframe"] = os.getenv("TIMEFRAME", config.get("timeframe", "M5"))
mode_env = os.getenv("MODE", config.get("mode", "paper")).lower()
config["mode"] = "paper" if mode_env == "demo" else mode_env
risk_tf_minutes = _granularity_minutes(config["timeframe"])
risk_cooldown_candles = int(os.getenv("COOLDOWN_CANDLES", config.get("cooldown_candles", 9)))
config["cooldown_candles"] = risk_cooldown_candles
config["cooldown_minutes"] = risk_tf_minutes * risk_cooldown_candles if risk_tf_minutes else config.get("cooldown_minutes", 0)
risk_config = config.get("risk", {}) or {}
risk_config.setdefault("risk_per_trade_pct", float(os.getenv("MAX_RISK_PER_TRADE", risk_config.get("risk_per_trade_pct", 0.005))))
risk_config.setdefault("atr_stop_mult", float(os.getenv("ATR_STOP_MULT", risk_config.get("atr_stop_mult", 1.8))))
risk_config.setdefault("tp_rr_multiple", float(os.getenv("TP_RR_MULTIPLE", risk_config.get("tp_rr_multiple", 1.5))))
risk_config.setdefault("cooldown_candles", int(os.getenv("COOLDOWN_CANDLES", risk_config.get("cooldown_candles", 9))))
risk_config.setdefault("max_concurrent_positions", int(os.getenv("MAX_CONCURRENT_POSITIONS", risk_config.get("max_concurrent_positions", 2))))
risk_config.setdefault("daily_loss_cap_pct", float(os.getenv("DAILY_LOSS_CAP_PCT", risk_config.get("daily_loss_cap_pct", 0.02))))
risk_config.setdefault("max_drawdown_cap_pct", float(os.getenv("MAX_DRAWDOWN_CAP_PCT", risk_config.get("max_drawdown_cap_pct", 0.10))))
risk_config["timeframe"] = config["timeframe"]
config["risk"] = risk_config

# Abort if live is requested (demo/practice only)
oanda_env = (os.getenv("OANDA_ENV") or "practice").lower()
if oanda_env == "live" or config["mode"] == "live":
    print("[STARTUP] Live mode is disabled for this deployment. Exiting.", flush=True)
    sys.exit(1)

broker = Broker()
engine = DecisionEngine(config)
risk = RiskManager(config.get("risk", {}), mode=config["mode"], state_dir=DATA_DIR)
profit_guard = ProfitProtection(broker)


def _startup_checks() -> None:
    broker.connectivity_check()


def _open_trades_state() -> List[Dict]:
    try:
        return broker.list_open_trades()
    except AttributeError:
        # Older broker implementations may not yet expose list_open_trades.
        return []


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
                if trade.get("instrument") not in closed_by_trail
            ]
        for evaluation in evaluations:
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

            atr_val = diagnostics.get("atr")
            sl_distance = risk.sl_distance_from_atr(atr_val)
            tp_distance = risk.tp_distance_from_atr(atr_val)
            entry_price = diagnostics.get("close")
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
            )
            if result.get("status") == "SENT":
                engine.mark_trade(evaluation.instrument)
                open_trades.append({"instrument": evaluation.instrument})
                risk.register_entry(now_utc, evaluation.instrument)
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
