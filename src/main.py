from __future__ import annotations

import asyncio
import json
import os
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


def load_config(path: Path = CONFIG_PATH) -> Dict:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        print(f"[CONFIG] Invalid JSON at {path}; using empty config", flush=True)
        return {}


config = load_config()
broker = Broker()
engine = DecisionEngine(config)
risk_mode = os.getenv("MODE", config.get("mode", "paper")).lower()
risk = RiskManager(config.get("risk", {}), mode=risk_mode)
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
    print(
        f"[HEARTBEAT] {ts_local} instruments={len(config.get('instruments', []))}",
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
            )
            if result.get("status") == "SENT":
                engine.mark_trade(evaluation.instrument)
                open_trades.append({"instrument": evaluation.instrument})
                risk.register_entry(now_utc)
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
