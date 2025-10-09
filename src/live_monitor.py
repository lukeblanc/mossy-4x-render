from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timezone
from typing import Dict, Iterable, Optional, Tuple

from app.broker import Broker


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def _extract_trade_id(trade: Dict) -> Optional[str]:
    for key in ("id", "tradeID", "tradeId"):
        value = trade.get(key)
        if value is not None:
            return str(value)
    return None


def _format_trade(trade: Dict) -> str:
    instrument = trade.get("instrument", "?")
    units_raw = trade.get("currentUnits") or trade.get("units") or "0"
    try:
        units_val = float(units_raw)
    except (TypeError, ValueError):
        units_val = 0.0
    direction = "BUY" if units_val > 0 else "SELL" if units_val < 0 else "FLAT"
    units_display = trade.get("currentUnits") or trade.get("units") or units_raw
    price = trade.get("price") or trade.get("averagePrice") or "n/a"
    pnl = trade.get("unrealizedPL") or trade.get("unrealizedPl")
    if pnl is None:
        pnl_display = "n/a"
    else:
        pnl_display = str(pnl)
    trade_id = _extract_trade_id(trade) or "?"
    return (
        f"id={trade_id} {instrument} direction={direction} units={units_display} "
        f"price={price} unrealizedPL={pnl_display}"
    )


def _diff_trades(
    previous: Dict[str, Dict], current: Dict[str, Dict]
) -> Tuple[Iterable[Dict], Iterable[Dict]]:
    prev_ids = set(previous.keys())
    curr_ids = set(current.keys())
    opened = [current[tid] for tid in curr_ids - prev_ids]
    closed = [previous[tid] for tid in prev_ids - curr_ids]
    return opened, closed


async def monitor(interval_seconds: int = 15) -> None:
    broker = Broker()
    if not (broker.key and broker.account):
        print(
            "[MONITOR] OANDA credentials missing. Set OANDA_API_KEY and OANDA_ACCOUNT_ID to track live trades.",
            flush=True,
        )
        return

    print(
        (
            f"[MONITOR] Starting OANDA {broker.mode.upper()} monitor for account {broker.account} "
            f"poll_interval={interval_seconds}s"
        ),
        flush=True,
    )

    previous: Dict[str, Dict] = {}

    while True:
        trades = broker.list_open_trades() or []
        snapshot: Dict[str, Dict] = {}
        for trade in trades:
            trade_id = _extract_trade_id(trade)
            if trade_id is None:
                continue
            snapshot[trade_id] = trade

        opened, closed = _diff_trades(previous, snapshot)

        if opened:
            for trade in opened:
                print(
                    f"[MONITOR] {_now_iso()} NEW { _format_trade(trade) }",
                    flush=True,
                )
        if closed:
            for trade in closed:
                print(
                    f"[MONITOR] {_now_iso()} CLOSED { _format_trade(trade) }",
                    flush=True,
                )

        if snapshot:
            summary = ", ".join(_format_trade(trade) for trade in snapshot.values())
            print(
                f"[MONITOR] {_now_iso()} ACTIVE {len(snapshot)} -> {summary}",
                flush=True,
            )
        else:
            print(f"[MONITOR] {_now_iso()} No open trades", flush=True)

        previous = snapshot
        await asyncio.sleep(interval_seconds)


def _parse_interval_argument() -> int:
    if len(sys.argv) < 2:
        return 15
    try:
        value = int(sys.argv[1])
        return value if value > 0 else 15
    except (TypeError, ValueError):
        return 15


def main() -> None:
    interval = _parse_interval_argument()
    asyncio.run(monitor(interval))


if __name__ == "__main__":
    main()
