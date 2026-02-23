from __future__ import annotations

import csv
import json
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "data"
DB = STATE / "trade_journal.db"
EVENTS = STATE / "audit_events.jsonl"
CSV_24H = STATE / "trade_history_last24h.csv"


@dataclass
class Metrics:
    trades: int
    net_pnl: float
    win_rate: float
    avg_win: float
    avg_loss: float
    max_drawdown: float
    profit_factor: float


def compute_metrics(rows: list[dict[str, Any]]) -> Metrics:
    pnl = [float(r.get("realized_pnl") or 0.0) for r in rows]
    wins = [x for x in pnl if x > 0]
    losses = [x for x in pnl if x < 0]
    net = sum(pnl)
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    equity = 0.0
    peak = 0.0
    mdd = 0.0
    for p in pnl:
        equity += p
        peak = max(peak, equity)
        dd = peak - equity
        mdd = max(mdd, dd)
    return Metrics(
        trades=len(pnl),
        net_pnl=net,
        win_rate=(len(wins) / len(pnl) * 100.0) if pnl else 0.0,
        avg_win=(sum(wins) / len(wins)) if wins else 0.0,
        avg_loss=(sum(losses) / len(losses)) if losses else 0.0,
        max_drawdown=mdd,
        profit_factor=(gross_win / gross_loss) if gross_loss > 0 else (999.0 if gross_win > 0 else 0.0),
    )


def load_30d_trades() -> list[dict[str, Any]]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=30)
    if not DB.exists():
        return []
    with sqlite3.connect(DB) as conn:
        rows = conn.execute(
            """
            SELECT timestamp_open, timestamp_close, symbol, side, size,
                   entry_price, exit_price, stop_loss, take_profit,
                   close_reason, realized_pnl, order_id, strategy_tag
            FROM audit_trades
            WHERE timestamp_open >= ?
            ORDER BY timestamp_open ASC
            """,
            (cutoff.isoformat(),),
        ).fetchall()
    keys = [
        "timestamp_open", "timestamp_close", "symbol", "side", "size",
        "entry_price", "exit_price", "stop_loss", "take_profit",
        "close_reason", "realized_pnl", "order_id", "strategy_tag",
    ]
    return [dict(zip(keys, r)) for r in rows]


def load_event_patterns() -> dict[str, Counter]:
    out: dict[str, Counter] = defaultdict(Counter)
    if not EVENTS.exists():
        return out
    cutoff = datetime.now(timezone.utc) - timedelta(days=30)
    with EVENTS.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts_raw = event.get("ts")
            if not ts_raw:
                continue
            try:
                ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
            except ValueError:
                continue
            if ts < cutoff:
                continue
            ev = event.get("event") or "unknown"
            out["event"][ev] += 1
            if ev == "risk_block":
                out["risk_reason"][str(event.get("reason") or "unknown")] += 1
            if ev == "indicator_snapshot":
                out["timeframe"][str(event.get("timeframe") or "unknown")] += 1
    return out


def main() -> None:
    trades = load_30d_trades()
    metrics = compute_metrics(trades)
    patterns = load_event_patterns()

    print("=== Mossy 4X 30-Day Performance ===")
    print(f"trades={metrics.trades}")
    print(f"net_pnl={metrics.net_pnl:.4f}")
    print(f"win_rate={metrics.win_rate:.2f}%")
    print(f"avg_win={metrics.avg_win:.4f}")
    print(f"avg_loss={metrics.avg_loss:.4f}")
    print(f"max_drawdown={metrics.max_drawdown:.4f}")
    print(f"profit_factor={metrics.profit_factor:.4f}")

    if not trades:
        print("NOTE: No 30-day trades found in data/trade_journal.db (audit_trades).")

    if CSV_24H.exists():
        with CSV_24H.open("r", encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
        print(f"trade_history_last24h_rows={max(len(rows) - 1, 0)}")
    else:
        print("trade_history_last24h.csv missing")

    if patterns:
        print("event_counts:", dict(patterns.get("event", {})))
        if patterns.get("risk_reason"):
            print("risk_block_reasons:", dict(patterns["risk_reason"]))


if __name__ == "__main__":
    main()
