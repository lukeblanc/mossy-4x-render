"""Compare recorded gold fills with an hourly dollar-and-yields filter."""

import argparse
import csv
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path


def utc(value):
    timestamp = datetime.fromisoformat(value)
    if timestamp.utcoffset() is None:
        raise ValueError(f"Timestamp must include a UTC offset: {value}")
    return timestamp.astimezone(timezone.utc)


def load_bars(path):
    bars = {}
    with path.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            close_time = utc(row["close_time"])
            available_at = utc(row["available_at"])
            value = float(row["value"])
            if (
                close_time != close_time.replace(minute=0, second=0, microsecond=0)
                or available_at < close_time
                or not math.isfinite(value)
                or close_time in bars
            ):
                raise ValueError(f"{path}: invalid or duplicate bar at {close_time}")
            bars[close_time] = (available_at, value)
    return bars


def direction(bars, entry):
    hour = entry.replace(minute=0, second=0, microsecond=0)
    current = bars.get(hour)
    previous = bars.get(hour - timedelta(hours=1))
    if current is None or previous is None:
        return None
    if max(current[0], previous[0]) > entry:
        return None
    return (current[1] > previous[1]) - (current[1] < previous[1])


def metrics(results):
    equity = peak = drawdown = 0.0
    for _, pnl in sorted(results):
        equity += pnl
        peak = max(peak, equity)
        drawdown = max(drawdown, peak - equity)
    return {
        "trades": len(results),
        "positive_trades": sum(pnl > 0 for _, pnl in results),
        "recorded_pnl": round(equity, 8),
        "closed_trade_drawdown": round(drawdown, 8),
    }


def replay(trades, dollar, yields, holdout_start):
    groups = {
        name: {"recorded": [], "comparable": [], "filtered": []}
        for name in ("development", "holdout")
    }
    seen = set()
    for trade in trades:
        if (
            trade["instrument"] != "XAU_USD"
            or trade["journal_status"] != "JOURNAL_CLOSED_CONFIRMED"
            or trade["broker_confirmed"] != 1
        ):
            continue
        trade_id = trade["trade_id"]
        if not trade_id or trade_id in seen:
            raise ValueError(f"Missing or duplicate trade ID: {trade_id}")
        seen.add(trade_id)
        entry = utc(trade["timestamp_utc"])
        exit_time = utc(trade["exit_timestamp_utc"])
        pnl = float(trade["realized_pnl_ccy"])
        side = {"BUY": -1, "SELL": 1}[trade["side"]]
        if not math.isfinite(pnl) or exit_time < entry:
            raise ValueError(f"Invalid P&L or timestamps for trade {trade_id}")
        group = groups["holdout" if entry >= holdout_start else "development"]
        result = (exit_time, pnl)
        group["recorded"].append(result)
        dollar_move, yield_move = direction(dollar, entry), direction(yields, entry)
        if dollar_move is None or yield_move is None:
            continue
        group["comparable"].append(result)
        if dollar_move == side == yield_move:
            group["filtered"].append(result)
    return {
        name: {
            "missing_market_data": len(group["recorded"]) - len(group["comparable"]),
            "trades_skipped_by_filter": len(group["comparable"]) - len(group["filtered"]),
            "recorded_baseline": metrics(group["recorded"]),
            "comparable_baseline": metrics(group["comparable"]) if group["comparable"] else None,
            "filtered": metrics(group["filtered"]) if group["comparable"] else None,
        }
        for name, group in groups.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("journal", type=Path, help="Sanitised journal JSON export")
    parser.add_argument("dxy", type=Path, help="Hourly DXY CSV: close_time,available_at,value")
    parser.add_argument("us2y", type=Path, help="Hourly US two-year yield CSV, same format")
    parser.add_argument("--holdout-start", required=True, type=utc)
    args = parser.parse_args()
    trades = json.loads(args.journal.read_text(encoding="utf-8"))["trades"]
    result = replay(trades, load_bars(args.dxy), load_bars(args.us2y), args.holdout_start)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
