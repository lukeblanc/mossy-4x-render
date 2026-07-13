from __future__ import annotations

import sqlite3
from pathlib import Path

from src import adaptive_tuner as adaptive_module
from src.adaptive_tuner import AdaptiveTuner


def _db(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE trades (
                trade_id TEXT,
                exit_timestamp_utc TEXT,
                realized_pnl_ccy REAL,
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
        for index, pnl in enumerate((1.0, -0.5, 0.4), start=1):
            conn.execute(
                """
                INSERT INTO trades(trade_id, exit_timestamp_utc, realized_pnl_ccy, run_tag)
                VALUES (?, '2026-07-13T13:00:00+00:00', ?, 'MINI_RUN')
                """,
                (str(index), pnl),
            )
        for pnl in (-5.0, -4.0, -3.0, -2.0, -1.0, 1.0, 1.0, 1.0):
            conn.execute(
                """
                INSERT INTO trade_events(timestamp, instrument, direction, entry_price,
                                         exit_price, profit, reason, equity_after)
                VALUES ('2026-07-01T00:00:00+00:00', 'USD_JPY', 'SELL',
                        1.0, 1.0, ?, 'SL', 1000.0)
                """,
                (pnl,),
            )


def test_clean_cohort_never_falls_back_to_unfiltered_events(tmp_path, monkeypatch):
    monkeypatch.setenv("SHADOW_LEARNING_ENABLED", "false")
    path = tmp_path / "journal.db"
    _db(path)
    tuner = AdaptiveTuner(
        path,
        lookback=80,
        min_sample=8,
        run_tag="MINI_RUN",
        window_start_utc="2026-07-13T12:47:00+00:00",
    )

    snapshot = tuner.snapshot()

    assert snapshot.source == "trades"
    assert snapshot.session_closed_trades == 3
    assert snapshot.wins == 2
    assert snapshot.losses == 1
    assert snapshot.risk_multiplier == 0.85


def test_shadow_analysis_runs_on_first_snapshot_even_with_young_monotonic_clock(
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "journal.db"
    _db(path)
    calls: list[Path] = []
    sentinel = object()

    monkeypatch.setenv("SHADOW_LEARNING_ENABLED", "true")
    monkeypatch.setenv("SHADOW_INTERVAL_SECONDS", "3600")
    monkeypatch.setattr(adaptive_module.time, "monotonic", lambda: 10.0)
    monkeypatch.setattr(
        adaptive_module,
        "run_shadow_analysis",
        lambda db_path: calls.append(Path(db_path)) or sentinel,
    )

    tuner = AdaptiveTuner(path, min_sample=8)
    tuner.snapshot()
    tuner.snapshot()

    assert calls == [path]
    assert tuner.shadow_report is sentinel
