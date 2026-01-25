from __future__ import annotations

import asyncio
import atexit
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Set

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from app.broker import Broker
from app.config import settings
from app.health import watchdog
from app.observability import EventLogger, get_event_logger, health_report_payload
from app.strategy import decide
from app.dashboard import send_bot_status
from src import profit_protection
from src.risk_setup import (
    build_profit_protection,
    build_risk_manager,
    resolve_state_dir,
)
from src.trade_journal import TradeJournal, default_journal_path

broker = Broker()
STATE_DIR = resolve_state_dir(Path(__file__).resolve().parent.parent / "data")
METRICS_FILE = STATE_DIR / "render_decisions.jsonl"
SUMMARY_INTERVAL = max(1, int(settings.METRIC_SUMMARY_INTERVAL))
EVENT_LOGGER: EventLogger = get_event_logger(STATE_DIR)
JOURNAL = TradeJournal(default_journal_path(STATE_DIR))
START_TS = datetime.now(timezone.utc)
MAX_DRAWDOWN_PCT: Optional[float] = None
CAP_HIT_TODAY = False
COOLDOWN_RELEASE_LOGGED: Set[str] = set()
CAP_DAY = START_TS.date()

trail_arm_pips = float(os.getenv("TRAIL_ARM_PIPS", "0.0"))
trail_giveback_pips = float(os.getenv("TRAIL_GIVEBACK_PIPS", "0.0"))
trail_arm_ccy = float(os.getenv("TRAIL_ARM_CCY", os.getenv("TRAIL_ARM_USD", str(profit_protection.ARM_AT_CCY))))
trail_giveback_ccy = float(
    os.getenv("TRAIL_GIVEBACK_CCY", os.getenv("TRAIL_GIVEBACK_USD", str(profit_protection.GIVEBACK_CCY)))
)
be_arm_pips = float(os.getenv("BE_ARM_PIPS", "6.0"))
be_offset_pips = float(os.getenv("BE_OFFSET_PIPS", "1.0"))
min_check_interval_sec = float(os.getenv("TRAIL_MIN_CHECK_INTERVAL", "0.0"))
trailing_config = {
    "arm_pips": trail_arm_pips,
    "giveback_pips": trail_giveback_pips,
    "arm_ccy": trail_arm_ccy,
    "giveback_ccy": trail_giveback_ccy,
    "use_pips": False,
    "be_arm_pips": be_arm_pips,
    "be_offset_pips": be_offset_pips,
    "min_check_interval_sec": min_check_interval_sec,
}
time_stop_config = {
    "minutes": float(os.getenv("TIME_STOP_MINUTES", "90")),
    "min_pips": float(os.getenv("TIME_STOP_MIN_PIPS", "2.0")),
    "xau_atr_mult": float(os.getenv("TIME_STOP_XAU_ATR_MULT", "0.35")),
}
risk_config = {
    "risk_per_trade_pct": settings.MAX_RISK_PER_TRADE,
    "cooldown_candles": settings.STRAT_COOLDOWN_BARS,
    "sl_atr_mult": settings.SL_ATR_MULT,
    "tp_atr_mult": settings.TP_ATR_MULT,
    "instrument_atr_multipliers": settings.INSTRUMENT_ATR_MULTIPLIERS,
    "timeframe": settings.STRAT_TIMEFRAME,
}

risk = build_risk_manager(
    {"risk": risk_config},
    mode=settings.MODE,
    demo_mode=settings.MODE.lower() == "demo",
    state_dir=STATE_DIR,
)
profit_guard = build_profit_protection(
    settings.MODE,
    broker,
    aggressive=False,
    trailing=trailing_config,
    time_stop=time_stop_config,
    journal=JOURNAL,
)


class DecisionMetrics:
    def __init__(self, path: Path, summary_interval: int) -> None:
        self.path = path
        self.summary_interval = max(1, summary_interval)
        self.decisions = 0
        self.orders = 0
        self.wins = 0
        self.losses = 0
        self.total_r = 0.0
        self.r_samples = 0
        self.max_adverse: Optional[float] = None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._load_existing()

    def _load_existing(self) -> None:
        if not self.path.exists():
            return
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    self._accumulate(payload)
        except Exception:
            return

    def _accumulate(self, record: Dict[str, Any]) -> None:
        self.decisions += 1
        if record.get("order_status") == "SENT":
            self.orders += 1
        pl = record.get("pl")
        if isinstance(pl, (int, float)):
            if pl > 0:
                self.wins += 1
            elif pl < 0:
                self.losses += 1
        expected_r = record.get("expected_r")
        if isinstance(expected_r, (int, float)) and expected_r > 0:
            self.total_r += expected_r
            self.r_samples += 1
        adverse = record.get("adverse_pips")
        if isinstance(adverse, (int, float)):
            if self.max_adverse is None or adverse > self.max_adverse:
                self.max_adverse = adverse

    def record(self, record: Dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        self._accumulate(record)

    def summary(self) -> Dict[str, Any]:
        avg_r = self.total_r / self.r_samples if self.r_samples else None
        total_outcomes = self.wins + self.losses
        win_rate = (self.wins / total_outcomes) if total_outcomes else None
        return {
            "decisions": self.decisions,
            "orders": self.orders,
            "win_rate": win_rate,
            "avg_r": avg_r,
            "max_adverse": self.max_adverse,
        }


metrics = DecisionMetrics(METRICS_FILE, SUMMARY_INTERVAL)


# Startup connectivity checks
def _startup_checks():
    """Connectivity checks to verify credentials if provided."""
    EVENT_LOGGER.log(
        "startup",
        {"mode": settings.MODE, "instrument": settings.INSTRUMENT, "timeframe": settings.STRAT_TIMEFRAME},
    )
    broker.connectivity_check()


def _entry_price_from_diag(diag: Dict[str, Any]) -> Optional[float]:
    if not diag:
        return None
    for key in ("close",):
        value = diag.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    closes = diag.get("closes") or []
    if closes:
        try:
            return float(closes[-1])
        except (TypeError, ValueError):
            return None
    return None


def _indicator_integrity(diag: Dict[str, Any]) -> tuple[bool, list[str]]:
    issues = []
    warmup_complete = bool(diag.get("warmup_complete"))
    if not warmup_complete:
        issues.append("warmup_incomplete")
    for key in ("ema_fast", "ema_slow", "rsi", "atr"):
        value = diag.get(key)
        if value is None:
            issues.append(f"{key}_missing")
            continue
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            issues.append(f"{key}_invalid")
    return len(issues) == 0, issues


def _log_indicator_snapshot(diag: Dict[str, Any]) -> None:
    EVENT_LOGGER.log(
        "indicator_snapshot",
        {
            "symbol": settings.INSTRUMENT,
            "timeframe": settings.STRAT_TIMEFRAME,
            "ema12": diag.get("ema_fast"),
            "ema26": diag.get("ema_slow"),
            "rsi14": diag.get("rsi"),
            "atr14": diag.get("atr"),
            "candle_close": diag.get("close"),
            "warmup_complete": bool(diag.get("warmup_complete")),
        },
    )


def _log_account_snapshot(stage: str, *, equity_override: Optional[float] = None) -> Dict[str, Any]:
    global MAX_DRAWDOWN_PCT
    snapshot = broker.account_snapshot()
    equity = equity_override if equity_override is not None else snapshot.get("equity")
    peak = risk.state.peak_equity or equity
    drawdown = None
    drawdown_pct = None
    if peak and equity is not None:
        drawdown = max(float(peak) - float(equity), 0.0)
        drawdown_pct = (drawdown / float(peak)) * 100.0 if peak else 0.0
        if MAX_DRAWDOWN_PCT is None or drawdown_pct > MAX_DRAWDOWN_PCT:
            MAX_DRAWDOWN_PCT = drawdown_pct
    EVENT_LOGGER.log(
        "account_snapshot",
        {
            "stage": stage,
            "balance": snapshot.get("balance"),
            "equity": equity,
            "used_margin": snapshot.get("used_margin"),
            "free_margin": snapshot.get("free_margin"),
            "margin_level": snapshot.get("margin_level"),
            "open_positions": snapshot.get("open_positions"),
            "drawdown": drawdown,
            "drawdown_pct": drawdown_pct,
        },
    )
    return snapshot


def _order_pl(result: Dict[str, Any]) -> Optional[float]:
    if not isinstance(result, dict):
        return None
    response = result.get("response") or {}
    transactions = [
        response.get("orderFillTransaction", {}),
        response.get("orderCancelTransaction", {}),
        response.get("orderCreateTransaction", {}),
    ]
    for tx in transactions:
        if not isinstance(tx, dict):
            continue
        pl = tx.get("pl") or tx.get("profit")
        if pl is not None:
            try:
                return float(pl)
            except (TypeError, ValueError):
                continue
    return None


def _order_ticket(result: Dict[str, Any]) -> Optional[str]:
    if not isinstance(result, dict):
        return None
    response = result.get("response") or {}
    transactions = [
        response.get("orderCreateTransaction", {}),
        response.get("orderFillTransaction", {}),
        response.get("orderCancelTransaction", {}),
    ]
    for tx in transactions:
        if not isinstance(tx, dict):
            continue
        for id_key in ("id", "orderID", "orderFillTransactionID", "tradeOpenedID"):
            value = tx.get(id_key)
            if value:
                return str(value)
    last_id = response.get("lastTransactionID") or result.get("order_id")
    return str(last_id) if last_id else None


def _filter_closed_trades(open_trades: list[Dict[str, Any]], closed_ids: list[str]) -> list[Dict[str, Any]]:
    if not closed_ids:
        return open_trades
    closed_set = set(closed_ids)
    remaining: list[Dict[str, Any]] = []
    for trade in open_trades:
        tid = profit_guard._trade_id(trade)  # type: ignore[attr-defined]
        instrument = trade.get("instrument")
        if tid in closed_set or instrument in closed_set:
            continue
        remaining.append(trade)
    return remaining


async def heartbeat():
    """Send a heartbeat and update watchdog timestamp."""
    watchdog.last_heartbeat_ts = datetime.now(timezone.utc)
    ts_local = datetime.now(timezone.utc).astimezone().isoformat()
    print(f"[HEARTBEAT] {ts_local} tz={settings.TZ} mode={settings.MODE}", flush=True)


async def decision_tick():
    """Run the strategy decision, log diagnostics, and place demo orders."""
    global CAP_HIT_TODAY, CAP_DAY
    now_utc = datetime.now(timezone.utc)
    if now_utc.date() != CAP_DAY:
        CAP_DAY = now_utc.date()
        CAP_HIT_TODAY = False
        COOLDOWN_RELEASE_LOGGED.clear()
    equity = broker.account_equity()
    try:
        risk.enforce_equity_floor(now_utc, equity, close_all_cb=broker.close_all_positions)
    except AttributeError:
        pass

    open_trades = broker.list_open_trades()
    closed_by_trail = profit_guard.process_open_trades(open_trades)
    if closed_by_trail:
        open_trades = _filter_closed_trades(open_trades, closed_by_trail)

    spread_pips = broker.current_spread(settings.INSTRUMENT)
    ts_local = now_utc.astimezone()

    try:
        signal, reason, diag = decide()  # decide() is synchronous
    except Exception as e:
        watchdog.record_error()
        err_ts = datetime.now(timezone.utc).astimezone().isoformat()
        EVENT_LOGGER.log("runtime_error", {"error": str(e)})
        print(f"[ERROR] {err_ts} error={e}", flush=True)
        metrics.record(
            {
                "ts": ts_local.isoformat(),
                "instrument": settings.INSTRUMENT,
                "signal": "ERROR",
                "reason": str(e),
                "order_status": "ERROR",
            }
        )
        return

    watchdog.last_decision_ts = datetime.now(timezone.utc)
    diag = diag or {}
    _log_indicator_snapshot(diag)
    size_multiplier = float(diag.get("size_multiplier", 1.0))
    size_multiplier = max(size_multiplier, 0.0)
    computed_size = 0
    if size_multiplier > 0:
        computed_size = max(1, int(settings.ORDER_SIZE * size_multiplier))

    atr_val = diag.get("atr")
    sl_distance = risk.sl_distance_from_atr(atr_val, instrument=settings.INSTRUMENT)
    tp_distance = 0.0  # Disable standard TP to avoid interfering with USD-based profit protection
    tp_distance_audit = risk.tp_distance_from_atr(atr_val, instrument=settings.INSTRUMENT)
    expected_r = tp_distance_audit / sl_distance if sl_distance > 0 else None
    entry_price = _entry_price_from_diag(diag)
    pip_size = diag.get("pip_size") or broker._pip_size(settings.INSTRUMENT)  # type: ignore[attr-defined]
    sl_price = None
    tp_price = None
    if entry_price is not None and sl_distance > 0:
        sl_price = entry_price - sl_distance if signal == "BUY" else entry_price + sl_distance
    if entry_price is not None and tp_distance_audit > 0:
        tp_price = entry_price + tp_distance_audit if signal == "BUY" else entry_price - tp_distance_audit

    log = {
        "ts": ts_local.isoformat(),
        "instrument": settings.INSTRUMENT,
        "ema_fast": diag.get("ema_fast"),
        "ema_slow": diag.get("ema_slow"),
        "rsi": diag.get("rsi"),
        "atr": atr_val,
        "adx": diag.get("adx"),
        "session": diag.get("session"),
        "size_multiplier": size_multiplier,
        "signal": signal,
        "reason": reason,
        "mode": settings.MODE,
        "size": computed_size,
        "sl_distance": sl_distance,
        "tp_distance": tp_distance,
        "tp_distance_audit": tp_distance_audit,
    }
    print(f"[DECISION] {log}", flush=True)
    print(
        f"[RISK] instrument={settings.INSTRUMENT} atr={atr_val} sl_dist={sl_distance:.5f} tp_dist={tp_distance:.5f}",
        flush=True,
    )

    order_status = "SKIPPED"
    broker_response: Dict[str, Any] | None = None
    indicator_ok, indicator_issues = _indicator_integrity(diag)
    if signal in ("BUY", "SELL"):
        trade_count_today = int(risk.state.daily_entry_count or 0)
        EVENT_LOGGER.log(
            "trade_cap_check",
            {
                "trade_count_today": trade_count_today,
                "max_trades_per_day": risk.max_trades_per_day,
            },
        )
        if risk.max_trades_per_day > 0 and trade_count_today >= risk.max_trades_per_day:
            EVENT_LOGGER.log(
                "trade_cap_reached",
                {
                    "trade_count_today": trade_count_today,
                    "max_trades_per_day": risk.max_trades_per_day,
                },
            )
            EVENT_LOGGER.log(
                "trading_halted",
                {"reason": "daily-trade-cap", "trade_count_today": trade_count_today},
            )
            global CAP_HIT_TODAY
            CAP_HIT_TODAY = True
        if not indicator_ok:
            EVENT_LOGGER.log(
                "indicator_block",
                {"issues": indicator_issues, "symbol": settings.INSTRUMENT},
            )
            order_status = "BLOCKED"
        elif computed_size <= 0:
            print(
                f"[BROKER] Skipping order {signal} for {settings.INSTRUMENT} due to zero size",
                flush=True,
            )
            order_status = "BLOCKED"
        else:
            ok_to_open, risk_reason = risk.should_open(
                now_utc,
                equity,
                open_trades,
                settings.INSTRUMENT,
                spread_pips,
            )
            if not ok_to_open:
                EVENT_LOGGER.log(
                    "risk_block",
                    {
                        "reason": risk_reason,
                        "symbol": settings.INSTRUMENT,
                    },
                )
                if risk_reason == "daily-trade-cap":
                    EVENT_LOGGER.log(
                        "trading_halted",
                        {"reason": "daily-trade-cap", "trade_count_today": trade_count_today},
                    )
                    CAP_HIT_TODAY = True
                if risk_reason == "cooldown":
                    cooldown_until = risk.state.cooldown_until.get(settings.INSTRUMENT)
                    EVENT_LOGGER.log(
                        "cooldown_block",
                        {
                            "symbol": settings.INSTRUMENT,
                            "cooldown_until": cooldown_until.isoformat() if cooldown_until else None,
                        },
                    )
                print(
                    f"[TRADE] Skipping {settings.INSTRUMENT} due to {risk_reason}",
                    flush=True,
                )
                order_status = "BLOCKED"
            else:
                _log_account_snapshot("pre_trade", equity_override=equity)
                broker_response = broker.place_order(
                    settings.INSTRUMENT,
                    signal,
                    computed_size,
                    sl_distance=sl_distance,
                    tp_distance=tp_distance,
                    entry_price=entry_price,
                )
                order_status = broker_response.get("status", "UNKNOWN")
                if order_status == "SENT":
                    order_id = _order_ticket(broker_response) or broker_response.get("order_id")
                    JOURNAL.record_entry(
                        trade_id=str(order_id or f"{settings.INSTRUMENT}-{int(time.time())}"),
                        timestamp_utc=now_utc,
                        instrument=settings.INSTRUMENT,
                        side=signal,
                        units=computed_size,
                        entry_price=entry_price,
                        stop_loss_price=sl_price,
                        take_profit_price=tp_price,
                        spread_at_entry=spread_pips,
                        session_id=str(diag.get("session") or "UNKNOWN"),
                        session_mode=str(diag.get("session") or "UNKNOWN"),
                        run_tag=settings.STRATEGY_TAG,
                        strategy_tag=settings.STRATEGY_TAG,
                        gating_flags={"indicator_ok": indicator_ok, "risk_ok": True},
                        indicators_snapshot=diag,
                    )
                    risk.register_entry(now_utc, settings.INSTRUMENT)
                    cooldown_until = risk.state.cooldown_until.get(settings.INSTRUMENT)
                    if cooldown_until:
                        EVENT_LOGGER.log(
                            "cooldown_start",
                            {
                                "symbol": settings.INSTRUMENT,
                                "cooldown_until": cooldown_until.isoformat(),
                            },
                        )
                _log_account_snapshot("post_trade")
    adverse_pips = spread_pips if spread_pips is not None else None
    pl = _order_pl(broker_response or {})
    metrics_payload = {
        "ts": ts_local.isoformat(),
        "instrument": settings.INSTRUMENT,
        "signal": signal,
        "reason": reason,
        "size": computed_size,
        "size_multiplier": size_multiplier,
        "atr": atr_val,
        "sl_distance": sl_distance,
        "tp_distance": tp_distance,
        "expected_r": expected_r,
        "entry_price": entry_price,
        "spread_pips": spread_pips,
        "adverse_pips": adverse_pips,
        "pip_size": pip_size,
        "order_status": order_status,
        "response": broker_response,
        "pl": pl,
    }
    metrics.record(metrics_payload)

    if metrics.decisions % SUMMARY_INTERVAL == 0:
        snapshot = metrics.summary()
        win_rate_pct = f"{snapshot['win_rate'] * 100:.1f}%" if snapshot["win_rate"] is not None else "n/a"
        avg_r_fmt = f"{snapshot['avg_r']:.2f}" if snapshot["avg_r"] is not None else "n/a"
        adverse_fmt = f"{snapshot['max_adverse']:.2f}" if snapshot["max_adverse"] is not None else "n/a"
        print(
            f"[METRICS] decisions={snapshot['decisions']} orders={snapshot['orders']} "
            f"win_rate={win_rate_pct} avg_R={avg_r_fmt} max_adverse_pips={adverse_fmt}",
            flush=True,
        )

    cooldown_until = risk.state.cooldown_until.get(settings.INSTRUMENT)
    if cooldown_until and now_utc >= cooldown_until and settings.INSTRUMENT not in COOLDOWN_RELEASE_LOGGED:
        EVENT_LOGGER.log(
            "cooldown_release",
            {"symbol": settings.INSTRUMENT, "cooldown_until": cooldown_until.isoformat()},
        )
        COOLDOWN_RELEASE_LOGGED.add(settings.INSTRUMENT)


async def periodic_snapshot():
    _log_account_snapshot("periodic")


async def daily_health_report():
    payload = health_report_payload(
        logger=EVENT_LOGGER,
        uptime_seconds=(datetime.now(timezone.utc) - START_TS).total_seconds(),
        errors_count=watchdog.total_errors,
        max_trade_cap_hit=CAP_HIT_TODAY,
        max_drawdown_pct=MAX_DRAWDOWN_PCT,
    )
    EVENT_LOGGER.log("daily_health_report", payload)


async def runner():
    """Main runner scheduling heartbeat and decision tasks and running watchdog."""
    _startup_checks()
    send_bot_status("RUNNING")

    scheduler = AsyncIOScheduler()
    scheduler.add_job(heartbeat, "interval", seconds=settings.HEARTBEAT_SECONDS)
    scheduler.add_job(decision_tick, "interval", seconds=settings.DECISION_SECONDS)
    scheduler.add_job(periodic_snapshot, "interval", minutes=settings.OBS_SNAPSHOT_MINUTES)
    scheduler.add_job(daily_health_report, "interval", minutes=settings.HEALTH_REPORT_MINUTES)
    scheduler.start()
    asyncio.create_task(watchdog.run())
    await heartbeat()
    await decision_tick()
    while True:
        await asyncio.sleep(3600)


if __name__ == "__main__":
    def _shutdown_log() -> None:
        EVENT_LOGGER.log(
            "shutdown",
            {
                "uptime_seconds": (datetime.now(timezone.utc) - START_TS).total_seconds(),
            },
        )

    atexit.register(_shutdown_log)
    try:
        asyncio.run(runner())
    except Exception as exc:
        EVENT_LOGGER.log("crash", {"error": str(exc)})
        raise
