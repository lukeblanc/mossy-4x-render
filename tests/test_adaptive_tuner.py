from __future__ import annotations

import sqlite3
from pathlib import Path

from src.adaptive_tuner import AdaptiveTuner


def _make_db(path: Path, trade_pnl: list[float], event_pnl: list[float] | None = None) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            """
            CREATE TABLE trades (
                trade_id TEXT,
                exit_timestamp_utc TEXT,
                realized_pnl_ccy REAL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE trade_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                instrument TEXT,
                direction TEXT,
                entry_price REAL,
                exit_price REAL,
                profit REAL,
                reason TEXT,
                equity_after REAL
            )
            """
        )
        for idx, pnl in enumerate(trade_pnl, start=1):
            conn.execute(
                "INSERT INTO trades(trade_id, exit_timestamp_utc, realized_pnl_ccy) VALUES (?, datetime('now'), ?)",
                (str(idx), pnl),
            )
        for idx, pnl in enumerate(event_pnl or [], start=1):
            conn.execute(
                "INSERT INTO trade_events(timestamp, instrument, direction, entry_price, exit_price, profit, reason, equity_after) "
                "VALUES (datetime('now'), 'EUR_USD', 'BUY', 1.1, 1.2, ?, 'TRAIL', 1000)",
                (pnl,),
            )
        conn.commit()
    finally:
        conn.close()


def test_adaptive_tuner_reduces_risk_on_loss_streak(tmp_path):
    db = tmp_path / "journal.db"
    _make_db(db, [-1.0, -2.0, -0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    snap = AdaptiveTuner(db, lookback=10, min_sample=8).snapshot()
    assert snap.closed_trades == 10
    assert snap.loss_streak == 3
    assert snap.risk_multiplier == 0.6
    assert snap.source == "trades"


def test_adaptive_tuner_uses_conservative_mode_with_small_sample(tmp_path):
    db = tmp_path / "journal.db"
    _make_db(db, [1.0, -1.0, 1.0])

    snap = AdaptiveTuner(db, lookback=40, min_sample=8).snapshot()
    assert snap.closed_trades == 3
    assert snap.risk_multiplier == 0.85
    assert snap.source == "trades"


def test_adaptive_tuner_falls_back_to_trade_events_when_trades_unavailable(tmp_path):
    db = tmp_path / "journal.db"
    _make_db(db, [], event_pnl=[-2.0, -1.0, 1.5, 0.5, -0.2])

    snap = AdaptiveTuner(db, lookback=10, min_sample=8).snapshot()
    assert snap.closed_trades == 5
    assert snap.source == "trade_events"
