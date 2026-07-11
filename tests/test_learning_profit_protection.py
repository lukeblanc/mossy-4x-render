from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from src.learning_profit_protection import LearningProfitProtection
from src.trade_journal import TradeJournal


class BrokerWithCloseFill:
    def __init__(self) -> None:
        self.profits = [1.2, 0.6]
        self.trades = [
            {
                "id": "broker-trade-99",
                "instrument": "AUD_USD",
                "currentUnits": "1000",
                "price": "0.66000",
            }
        ]

    def get_unrealized_profit(self, instrument: str):
        return self.profits.pop(0)

    def list_open_trades(self):
        return list(self.trades)

    def position_snapshot(self, instrument: str):
        if not self.trades:
            return {"instrument": instrument, "longUnits": "0", "shortUnits": "0"}
        return {"instrument": instrument, "longUnits": "1000", "shortUnits": "0"}

    def close_position_side(self, instrument: str, long_units: float, short_units: float):
        self.trades = []
        return {
            "status": "CLOSED",
            "response": {
                "longOrderFillTransaction": {
                    "pl": "0.55",
                    "price": "0.66010",
                }
            },
        }

    def current_spread(self, instrument: str):
        return 0.8

    def account_equity(self):
        return 1500.55

    @staticmethod
    def _pip_size(instrument: str):
        return 0.0001


def test_broker_trade_id_reconciles_to_entry_and_saves_realised_pnl(tmp_path):
    journal = TradeJournal(tmp_path / "trade_journal.db")
    opened = datetime(2026, 7, 11, 7, 0, tzinfo=timezone.utc)
    journal.record_entry(
        trade_id="order-1",
        timestamp_utc=opened,
        instrument="AUD_USD",
        side="BUY",
        units=1000,
        entry_price=0.66,
        stop_loss_price=0.659,
        take_profit_price=0.661,
        spread_at_entry=0.8,
        session_id="LONDON-2026-07-11-1500",
        session_mode="SOFT",
        run_tag="MINI_RUN",
        gating_flags={"trend_ok": True},
        indicators_snapshot={"rsi": 57.0, "ema_fast": 1.2, "ema_slow": 1.1},
        equity_after=1500.0,
    )

    broker = BrokerWithCloseFill()
    guard = LearningProfitProtection(
        broker,
        arm_ccy=1.0,
        giveback_ccy=0.5,
        journal=journal,
    )

    trade = dict(broker.trades[0])
    trade["openTime"] = opened.isoformat()
    assert guard.process_open_trades([trade], now_utc=opened + timedelta(minutes=5)) == []

    trade = dict(broker.trades[0])
    trade["openTime"] = opened.isoformat()
    closed = guard.process_open_trades([trade], now_utc=opened + timedelta(minutes=10))

    assert closed == ["broker-trade-99"]
    with sqlite3.connect(journal.path) as conn:
        row = conn.execute(
            """
            SELECT exit_timestamp_utc, exit_price, realized_pnl_ccy,
                   exit_reason, broker_confirmed, side, entry_price
            FROM trades
            WHERE trade_id = 'order-1'
            """
        ).fetchone()
        orphan = conn.execute(
            "SELECT COUNT(*) FROM trades WHERE trade_id = 'broker-trade-99'"
        ).fetchone()[0]

    assert row is not None
    assert row[0] is not None
    assert row[1] == pytest.approx(0.66010)
    assert row[2] == pytest.approx(0.55)
    assert row[3] == "TRAIL"
    assert row[4] == 1
    assert row[5] == "BUY"
    assert row[6] == pytest.approx(0.66)
    assert orphan == 0
