import asyncio
from datetime import datetime, timezone
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from app.config import settings
from app.strategy import decide
from app.broker import Broker
from app.health import watchdog

broker = Broker()

# Startup connectivity checks
def _startup_checks():
    """Connectivity checks to verify credentials if provided."""
    broker.connectivity_check()

async def heartbeat():
    """Send a heartbeat and update watchdog timestamp."""
    # update heartbeat timestamp (UTC)
    watchdog.last_heartbeat_ts = datetime.now(timezone.utc)
    ts_local = datetime.now(timezone.utc).astimezone().isoformat()
    print(f"[HEARTBEAT] {ts_local} tz={settings.TZ} mode={settings.MODE}", flush=True)

async def decision_tick():
    """Run the strategy decision, log diagnostics, and place demo orders."""
    ts_local = datetime.now(timezone.utc).astimezone()
    try:
        signal, reason, diag = decide()  # decide() is synchronous
    except Exception as e:
        watchdog.record_error()
        err_ts = datetime.now(timezone.utc).astimezone().isoformat()
        print(f"[ERROR] {err_ts} error={e}", flush=True)
        return
    # update last decision timestamp
    watchdog.last_decision_ts = datetime.now(timezone.utc)
    diag = diag or {}
    size_multiplier = float(diag.get("size_multiplier", 1.0))
    size_multiplier = max(size_multiplier, 0.0)
    computed_size = 0
    if size_multiplier > 0:
        computed_size = max(1, int(settings.ORDER_SIZE * size_multiplier))

    log = {
        "ts": ts_local.isoformat(),
        "instrument": settings.INSTRUMENT,
        "ema_fast": diag.get("ema_fast"),
        "ema_slow": diag.get("ema_slow"),
        "rsi": diag.get("rsi"),
        "atr": diag.get("atr"),
        "adx": diag.get("adx"),
        "session": diag.get("session"),
        "size_multiplier": size_multiplier,
        "signal": signal,
        "reason": reason,
        "mode": settings.MODE,
        "size": computed_size,
    }
    print(f"[DECISION] {log}", flush=True)
    if signal in ("BUY", "SELL"):
        if computed_size <= 0:
            print(
                f"[BROKER] Skipping order {signal} for {settings.INSTRUMENT} due to zero size",
                flush=True,
            )
            return
        broker.place_order(settings.INSTRUMENT, signal, computed_size)

async def runner():
    """Main runner scheduling heartbeat and decision tasks and running watchdog."""
    _startup_checks()
    scheduler = AsyncIOScheduler()
    scheduler.add_job(heartbeat, "interval", seconds=settings.HEARTBEAT_SECONDS)
    scheduler.add_job(decision_tick, "interval", seconds=settings.DECISION_SECONDS)
    scheduler.start()
    asyncio.create_task(watchdog.run())
    await heartbeat()
    await decision_tick()
    while True:
        await asyncio.sleep(3600)

if __name__ == "__main__":
    asyncio.run(runner())
