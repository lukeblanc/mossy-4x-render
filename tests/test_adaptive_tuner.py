from __future__ import annotations

import sqlite3
from pathlib import Path

from src.adaptive_tuner import AdaptiveTuner


def _make_db(path: Path, pnl_values: list[float]) -> None:
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
        for idx, pnl in enumerate(pnl_values, start=1):
            conn.execute(
                "INSERT INTO trades(trade_id, exit_timestamp_utc, realized_pnl_ccy) VALUES (?, datetime('now'), ?)",
                (str(idx), pnl),
            )
        conn.commit()
    finally:
        conn.close()


def test_adaptive_tuner_reduces_risk_on_loss_streak(tmp_path):
    db = tmp_path / "journal.db"
    _make_db(db, [-1.0, -2.0, -0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    snap = AdaptiveTuner(db, lookback=10).snapshot()
    assert snap.closed_trades == 10
    assert snap.loss_streak == 3
    assert snap.risk_multiplier == 0.6


def test_adaptive_tuner_defaults_to_one_when_small_sample(tmp_path):
    db = tmp_path / "journal.db"
    _make_db(db, [1.0, -1.0, 1.0])

    snap = AdaptiveTuner(db, lookback=40).snapshot()
    assert snap.closed_trades == 3
    assert snap.risk_multiplier == 1.0
