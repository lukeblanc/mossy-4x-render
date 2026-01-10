import json
import sqlite3
from datetime import datetime, timedelta, timezone

from src.trade_journal import TradeJournal


def _row_for(conn: sqlite3.Connection, trade_id: str):
    return conn.execute(
        "SELECT timestamp_utc, instrument, spread_at_entry, gating_flags, indicators_snapshot, "
        "exit_timestamp_utc, max_profit_ccy, realized_pnl_ccy, exit_reason, duration_seconds, broker_confirmed, run_tag "
        "FROM trades WHERE trade_id=?",
        (trade_id,),
    ).fetchone()


def test_trade_journal_entry_and_exit(tmp_path):
    db_path = tmp_path / "journal.db"
    journal = TradeJournal(db_path)
    entry_ts = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
    journal.record_entry(
        trade_id="T-1",
        timestamp_utc=entry_ts,
        instrument="EUR_USD",
        side="BUY",
        units=1000,
        entry_price=1.2345,
        stop_loss_price=1.2300,
        take_profit_price=1.2400,
        spread_at_entry=0.12,
        session_id="LONDON",
        session_mode="STRICT",
        run_tag="MINI_RUN",
        gating_flags={"session_ok": True, "risk_ok": True, "spread_ok": True},
        indicators_snapshot={"rsi": 55.5, "atr": 0.0007},
    )

    with sqlite3.connect(db_path) as conn:
        row = _row_for(conn, "T-1")
        assert row[0].startswith("2024-01-01T12:00")
        assert row[1] == "EUR_USD"
        assert row[2] == 0.12
        gating = json.loads(row[3])
        assert gating["session_ok"] is True
        indicators = json.loads(row[4])
        assert indicators["rsi"] == 55.5
        assert row[11] == "MINI_RUN"

    exit_ts = entry_ts + timedelta(minutes=15)
    journal.record_exit(
        trade_id="T-1",
        exit_timestamp_utc=exit_ts,
        exit_price=1.2360,
        spread_at_exit=0.15,
        max_profit_ccy=1.25,
        realized_pnl_ccy=1.10,
        exit_reason="TRAIL",
        duration_seconds=900,
        broker_confirmed=True,
        run_tag="MINI_RUN",
    )

    with sqlite3.connect(db_path) as conn:
        row = _row_for(conn, "T-1")
        assert row[5].startswith("2024-01-01T12:15")
        assert row[6] == 1.25
        assert row[7] == 1.10
        assert row[8] == "TRAIL"
        assert row[9] == 900
        assert row[10] == 1
