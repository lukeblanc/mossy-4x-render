from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from app.broker import Broker
from app.health import watchdog
from src.decision_engine import DecisionEngine, Evaluation

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "defaults.json"
LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
TRADE_LOG_PATH = LOG_DIR / "trade_activity.log"


def load_config(path: Path = CONFIG_PATH) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


config = load_config()
broker = Broker()
engine = DecisionEngine(config)


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
        message = (
            f"[TRADE] Skipping {evaluation.instrument} signal due to max open trades {max_open}"
        )
        print(message, flush=True)
        _record_trade_log(message)
        return False

    active_instruments = {trade.get("instrument") for trade in open_trades if isinstance(trade, dict)}
    if evaluation.instrument in active_instruments:
        message = f"[TRADE] Skipping {evaluation.instrument} signal; trade already open"
        print(message, flush=True)
        _record_trade_log(message)
        return False

    if evaluation.reason == "cooldown":
        message = f"[TRADE] Skipping {evaluation.instrument} signal; instrument cooling down"
        print(message, flush=True)
        _record_trade_log(message)
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
    instruments = getattr(engine, "instruments", [])
    if instruments:
        instrument_list = ", ".join(instruments)
        print(
            f"[CYCLE] Running decision cycle for {len(instruments)} instruments: {instrument_list}",
            flush=True,
        )
    else:
        print("[CYCLE] Running decision cycle", flush=True)
    try:
        evaluations = engine.evaluate_all()
    except Exception as exc:  # pragma: no cover - defensive logging
        watchdog.record_error()
        ts = datetime.now(timezone.utc).astimezone().isoformat()
        print(f"[ERROR] {ts} decision-cycle failure error={exc}", flush=True)
        return
    else:
        open_trades = _open_trades_state()
        if open_trades:
            summaries = []
            for trade in open_trades:
                instrument = trade.get("instrument", "?")
                units = trade.get("currentUnits") or trade.get("units")
                summaries.append(f"{instrument}:{units}")
            print(
                "[STATUS] Open trades=" + ", ".join(summaries),
                flush=True,
            )
        else:
            print("[STATUS] No open trades", flush=True)
        for evaluation in evaluations:
            if not _should_place_trade(open_trades, evaluation):
                continue

            diagnostics = evaluation.diagnostics or {}
            units = engine.position_size(evaluation.instrument, diagnostics)
            result = broker.place_order(
                evaluation.instrument, evaluation.signal, units
            )
            if result.get("status") == "SENT":
                engine.mark_trade(evaluation.instrument)
                open_trades.append({"instrument": evaluation.instrument})
                order_details = result.get("response", {}) if isinstance(result, dict) else {}
                order_fill = order_details.get("orderFillTransaction", {})
                order_create = order_details.get("orderCreateTransaction", {})
                order_id = (
                    order_fill.get("id")
                    or order_create.get("id")
                    or order_details.get("lastTransactionID")
                )
                trade_opened = None
                trade_open_payload = order_fill.get("tradeOpened")
                if isinstance(trade_open_payload, dict):
                    trade_opened = trade_open_payload.get("tradeID")
                fill_price = (
                    order_fill.get("price")
                    or order_fill.get("averagePrice")
                    or order_create.get("price")
                )
                message = (
                    f"[TRADE] Placed {evaluation.signal} on {evaluation.instrument} "
                    f"units={units} order_id={order_id} trade_id={trade_opened} price={fill_price}"
                )
                print(message, flush=True)
                _record_trade_log(message)
            else:
                message = (
                    f"[TRADE] Order failed instrument={evaluation.instrument} signal={evaluation.signal}"
                    f" response={result}"
                )
                print(message, flush=True)
                _record_trade_log(message)
        print(
            f"[CYCLE] Completed decision cycle for {len(evaluations)} instruments", flush=True
        )
    finally:
        watchdog.last_decision_ts = datetime.now(timezone.utc)


def _record_trade_log(message: str) -> None:
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).astimezone().isoformat()
        with TRADE_LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(f"{timestamp} {message}\n")
    except Exception:
        # Logging should never interrupt trading. Fail silently.
        pass


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
