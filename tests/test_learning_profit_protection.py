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
                    "time": "2026-07-11T07:10:00Z",
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


class BrokerWithExternalClose(BrokerWithCloseFill):
    def __init__(self) -> None:
        super().__init__()
        self.profits = [0.2]
        self.closed_details = None

    def trade_details(self, trade_id: str):
        return self.closed_details


def _record_entry(journal: TradeJournal, opened: datetime) -> None:
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


def _read_result(journal: TradeJournal):
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
    return row, orphan


def test_broker_trade_id_reconciles_to_entry_and_saves_realised_pnl(tmp_path):
    journal = TradeJournal(tmp_path / "trade_journal.db")
    opened = datetime(2026, 7, 11, 7, 0, tzinfo=timezone.utc)
    _record_entry(journal, opened)

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
    row, orphan = _read_result(journal)
    assert row is not None
    assert row[0] == "2026-07-11T07:10:00+00:00"
    assert row[1] == pytest.approx(0.66010)
    assert row[2] == pytest.approx(0.55)
    assert row[3] == "TRAIL"
    assert row[4] == 1
    assert row[5] == "BUY"
    assert row[6] == pytest.approx(0.66)
    assert orphan == 0


def test_external_stop_or_take_profit_is_not_forgotten(tmp_path):
    journal = TradeJournal(tmp_path / "trade_journal.db")
    opened = datetime(2026, 7, 11, 7, 0, tzinfo=timezone.utc)
    _record_entry(journal, opened)

    broker = BrokerWithExternalClose()
    guard = LearningProfitProtection(
        broker,
        arm_ccy=1.0,
        giveback_ccy=0.5,
        journal=journal,
    )

    trade = dict(broker.trades[0])
    trade["openTime"] = opened.isoformat()
    assert guard.process_open_trades([trade], now_utc=opened + timedelta(minutes=5)) == []

    broker.trades = []
    broker.closed_details = {
        "id": "broker-trade-99",
        "instrument": "AUD_USD",
        "state": "CLOSED",
        "currentUnits": "0",
        "realizedPL": "-1.25",
        "averageClosePrice": "0.65900",
        "closeTime": "2026-07-11T07:08:00Z",
    }
    closed = guard.process_open_trades([], now_utc=opened + timedelta(minutes=10))

    assert closed == ["broker-trade-99"]
    row, orphan = _read_result(journal)
    assert row is not None
    assert row[0] == "2026-07-11T07:08:00+00:00"
    assert row[1] == pytest.approx(0.65900)
    assert row[2] == pytest.approx(-1.25)
    assert row[3] == "BROKER_CLOSED"
    assert row[4] == 1
    assert orphan == 0
