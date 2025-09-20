import asyncio
from datetime import datetime
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from app.config import settings
from app.strategy import decide
from app.broker import Broker

broker = Broker()

# Startup connectivity check
def _startup_checks():
    # harmless in demo; proves keys if provided
    broker.connectivity_check()

async def heartbeat():
    print(
        f"[HEARTBEAT] {datetime.now().isoformat()} tz={settings.TZ} mode={settings.MODE}",
        flush=True,
    )

async def decision_tick():
    sig = decide()
    print(
        f"[DECISION] {datetime.now().isoformat()} signal={sig}",
        flush=True,
    )
if sig in ("BUY", "SELL"):
        broker.place_order(sig, size=1.0)

async def runner():
    scheduler = AsyncIOScheduler(timezone=settings.TZ)
    scheduler.add_job(heartbeat, "interval", seconds=settings.HEARTBEAT_SECONDS, id="heartbeat")
    scheduler.add_job(decision_tick, "interval", seconds=settings.DECISION_SECONDS, id="decision")
    scheduler.start()
    try:
        while True:
            await asyncio.sleep(3600)
    except (KeyboardInterrupt, asyncio.CancelledError):
        scheduler.shutdown()

def main():
    try:
        import uvloop  # optional; ignored if not supported
        uvloop.install()
    except Exception:
        pass
    _startup_checks()  # call this before starting the scheduler
    asyncio.run(runner())

if __name__ == "__main__":
    main()
