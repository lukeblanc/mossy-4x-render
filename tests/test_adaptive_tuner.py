from __future__ import annotations

import sqlite3
from pathlib import Path

from src.adaptive_tuner import AdaptiveTuner
from src.learning_cohort import (
    DEFAULT_LEARNING_COHORT_START_UTC,
    DEFAULT_LEARNING_RUN_TAG,
)


def _make_db(path: Path, trade_pnl: list[float], event_pnl: list[float] | None = None) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            """
            CREATE TABLE trades (
                trade_id TEXT,
                timestamp_utc TEXT,
                exit_timestamp_utc TEXT,
                realized_pnl_ccy REAL,
                broker_confirmed INTEGER,
                run_tag TEXT
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
                """
                INSERT INTO trades(
                    trade_id, timestamp_utc, exit_timestamp_utc,
                    realized_pnl_ccy, broker_confirmed, run_tag
                ) VALUES (?, '2026-07-13T12:50:00+00:00', ?, ?, 1, 'MINI_RUN')
                """,
                (str(idx), f"2026-07-13T13:{60 - idx:02d}:00+00:00", pnl),
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
    assert snap.lifetime_closed_trades == 10
    assert snap.session_closed_trades == 10
    assert snap.loss_streak == 3
    assert snap.risk_multiplier == 0.6
    assert snap.reason_code == "loss_streak"
    assert snap.source == "trades"


def test_adaptive_tuner_uses_conservative_mode_with_small_sample(tmp_path):
    db = tmp_path / "journal.db"
    _make_db(db, [1.0, -1.0, 1.0])

    snap = AdaptiveTuner(db, lookback=40, min_sample=8).snapshot()
    assert snap.lifetime_closed_trades == 3
    assert snap.session_closed_trades == 3
    assert snap.risk_multiplier == 0.85
    assert snap.reason_code == "small_sample"
    assert snap.source == "trades"


def test_adaptive_tuner_never_uses_legacy_trade_events(tmp_path):
    db = tmp_path / "journal.db"
    _make_db(db, [], event_pnl=[-2.0, -1.0, 1.5, 0.5, -0.2])

    snap = AdaptiveTuner(db, lookback=10, min_sample=8).snapshot()
    assert snap.lifetime_closed_trades == 0
    assert snap.session_closed_trades == 0
    assert snap.risk_multiplier == 0.85
    assert snap.reason_code == "small_sample"
    assert snap.source == "trades"


def test_adaptive_tuner_normalizes_blank_cohort_filters(tmp_path):
    db = tmp_path / "journal.db"
    _make_db(db, [1.0, -0.5, 0.25])

    tuner = AdaptiveTuner(
        db,
        lookback=10,
        min_sample=8,
        run_tag="   ",
        window_start_utc="",
    )
    snap = tuner.snapshot()

    assert tuner.run_tag == DEFAULT_LEARNING_RUN_TAG
    assert tuner.window_start_utc == DEFAULT_LEARNING_COHORT_START_UTC
    assert snap.filter_run_tag == DEFAULT_LEARNING_RUN_TAG
    assert snap.filter_window_start_utc == DEFAULT_LEARNING_COHORT_START_UTC
    assert snap.session_closed_trades == 3
    assert snap.reason_code == "small_sample"


def test_adaptive_tuner_reports_negative_expectancy_reducer(tmp_path):
    db = tmp_path / "journal.db"
    _make_db(db, [1.0, -4.0, 1.0, -4.0, 1.0, -4.0, 1.0, -4.0])

    snap = AdaptiveTuner(db, lookback=10, min_sample=8).snapshot()

    assert snap.loss_streak == 0
    assert snap.risk_multiplier == 0.65
    assert snap.reason_code == "negative_expectancy"


def test_adaptive_tuner_reports_profit_factor_reducer(tmp_path):
    db = tmp_path / "journal.db"
    # Recent wins make recency-weighted expectancy positive, while aggregate
    # gross profit remains below aggregate gross loss.
    _make_db(db, [2.0, 2.0, 2.0, 2.0, -3.0, -3.0, -3.0, -3.0])

    snap = AdaptiveTuner(db, lookback=10, min_sample=8).snapshot()

    assert snap.loss_streak == 0
    assert snap.risk_multiplier == 0.8
    assert snap.reason_code == "profit_factor_below_one"


def test_adaptive_tuner_reports_low_win_rate_reducer(tmp_path):
    db = tmp_path / "journal.db"
    # Three large wins keep expectancy and profit factor positive, while the
    # setup still wins fewer than four of ten observations.
    _make_db(db, [10.0, -1.0, 10.0, -1.0, 10.0, -1.0, -1.0, -1.0, -1.0, -1.0])

    snap = AdaptiveTuner(db, lookback=10, min_sample=8).snapshot()

    assert snap.loss_streak == 0
    assert snap.wins == 3
    assert snap.risk_multiplier == 0.8
    assert snap.reason_code == "low_win_rate"
