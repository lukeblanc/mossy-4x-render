from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

from src.weekly_ops_report import build_weekly_report, render_markdown, save_report


def _make_db(path) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE trades (
                trade_id TEXT PRIMARY KEY,
                exit_timestamp_utc TEXT,
                realized_pnl_ccy REAL,
                instrument TEXT,
                side TEXT,
                session_id TEXT,
                exit_reason TEXT,
                duration_seconds REAL,
                spread_at_entry REAL,
                spread_at_exit REAL,
                max_profit_ccy REAL,
                broker_confirmed INTEGER
            )
            """
        )


def test_weekly_report_calculates_verified_performance(tmp_path) -> None:
    db = tmp_path / "trade_journal.db"
    _make_db(db)
    now = datetime(2026, 8, 2, 12, 0, tzinfo=timezone.utc)
    rows = [
        ("T1", now - timedelta(days=1), 2.0, "AUD_USD", "BUY", "LONDON", "TP"),
        ("T2", now - timedelta(days=2), -1.0, "AUD_USD", "SELL", "LONDON", "SL"),
        ("T3", now - timedelta(days=3), 1.5, "GBP_USD", "BUY", "NEWYORK", "TRAIL"),
    ]
    with sqlite3.connect(db) as conn:
        for trade_id, closed, pnl, instrument, side, session, reason in rows:
            conn.execute(
                """
                INSERT INTO trades(
                    trade_id, exit_timestamp_utc, realized_pnl_ccy, instrument,
                    side, session_id, exit_reason, duration_seconds,
                    spread_at_entry, spread_at_exit, max_profit_ccy, broker_confirmed
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 600, 0.8, 0.9, ?, 1)
                """,
                (trade_id, closed.isoformat(), pnl, instrument, side, session, reason, max(pnl, 0.0)),
            )

    report = build_weekly_report(db, now_utc=now)

    assert report.total.trades == 3
    assert report.total.wins == 2
    assert report.total.losses == 1
    assert report.total.net_pnl == 2.5
    assert report.total.profit_factor == 3.5
    assert report.best_instrument == "GBP_USD"
    assert report.worst_instrument == "AUD_USD"
    assert report.status in {"GO", "CAUTION"}
    assert report.alerts == ()


def test_weekly_report_flags_missing_or_empty_journal(tmp_path) -> None:
    missing = tmp_path / "missing.db"
    report = build_weekly_report(
        missing,
        now_utc=datetime(2026, 8, 2, 12, 0, tzinfo=timezone.utc),
    )
    assert report.status == "STAND DOWN"
    assert "trade journal missing" in report.alerts

    db = tmp_path / "trade_journal.db"
    _make_db(db)
    empty = build_weekly_report(
        db,
        now_utc=datetime(2026, 8, 2, 12, 0, tzinfo=timezone.utc),
    )
    assert empty.total.trades == 0
    assert any("no broker-confirmed closed trades" in alert for alert in empty.alerts)


def test_report_saves_markdown_json_and_latest(tmp_path) -> None:
    db = tmp_path / "trade_journal.db"
    _make_db(db)
    now = datetime(2026, 8, 2, 12, 0, tzinfo=timezone.utc)
    report = build_weekly_report(db, now_utc=now)

    markdown_path, json_path, latest_path = save_report(report, tmp_path / "reports")

    assert markdown_path.exists()
    assert json_path.exists()
    assert latest_path.exists()
    markdown = render_markdown(report)
    assert "Mossy 4X Weekly ALGO Operations Report" in markdown
    assert "Champion entry and execution rules were not changed" in markdown
    assert latest_path.read_text(encoding="utf-8") == markdown
