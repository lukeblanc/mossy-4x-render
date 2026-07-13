from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

import pytest

from src.journal_reconciler import JournalReconcilerProfitProtection
from src.risk_setup import resolve_state_dir
from src.trade_journal import TradeJournal


class FakeResponse:
    def __init__(self, status_code: int, payload: dict) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> dict:
        return self._payload


class FakeClient:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def get(self, path: str, params=None):
        if path.endswith("/orders/order-1"):
            return FakeResponse(
                200,
                {"order": {"id": "order-1", "fillingTransactionID": "fill-1"}},
            )
        if path.endswith("/transactions/fill-1"):
            return FakeResponse(
                200,
                {"transaction": {"id": "fill-1", "tradeOpened": {"tradeID": "trade-99"}}},
            )
        if path.endswith("/transactions/sinceid"):
            return FakeResponse(200, {"transactions": []})
        return FakeResponse(404, {})


class ClosedTradeBroker:
    account = "practice-account"

    def __init__(self, details_by_id: dict[str, dict]) -> None:
        self.details_by_id = details_by_id

    def _client(self):
        return FakeClient()

    def trade_details(self, trade_id: str):
        return self.details_by_id.get(str(trade_id))

    def current_spread(self, instrument: str):
        return 0.8

    def account_equity(self):
        return 1498.75

    def get_unrealized_profit(self, instrument: str):
        return 0.0

    def position_snapshot(self, instrument: str):
        return {"instrument": instrument, "longUnits": "0", "shortUnits": "0"}

    def close_position_side(self, instrument: str, long_units: float, short_units: float):
        return {"status": "CLOSED"}

    @staticmethod
    def _pip_size(instrument: str):
        return 0.0001


def _record_entry(journal: TradeJournal, trade_id: str) -> datetime:
    opened = datetime(2026, 7, 13, 1, 0, tzinfo=timezone.utc)
    journal.record_entry(
        trade_id=trade_id,
        timestamp_utc=opened,
        instrument="AUD_USD",
        side="BUY",
        units=1000,
        entry_price=0.66,
        stop_loss_price=0.659,
        take_profit_price=0.661,
        spread_at_entry=0.8,
        session_id="LONDON",
        session_mode="SOFT",
        run_tag="MINI_RUN",
        gating_flags={"trend_ok": True},
        indicators_snapshot={
            "rsi": 57.0,
            "ema_fast": 1.2,
            "ema_slow": 1.1,
            "ema50": 1.15,
            "ema200": 1.0,
        },
        equity_after=1500.0,
    )
    return opened


def _read_exit(journal: TradeJournal, trade_id: str):
    with sqlite3.connect(journal.path) as conn:
        return conn.execute(
            """
            SELECT exit_timestamp_utc, exit_price, realized_pnl_ccy,
                   exit_reason, broker_confirmed
            FROM trades
            WHERE trade_id = ?
            """,
            (trade_id,),
        ).fetchone()


def test_untracked_fast_close_is_recovered_from_journal(tmp_path):
    journal = TradeJournal(tmp_path / "trade_journal.db")
    _record_entry(journal, "trade-42")
    broker = ClosedTradeBroker(
        {
            "trade-42": {
                "id": "trade-42",
                "instrument": "AUD_USD",
                "state": "CLOSED",
                "currentUnits": "0",
                "realizedPL": "-1.25",
                "averageClosePrice": "0.65900",
                "closeTime": "2026-07-13T01:00:20Z",
            }
        }
    )
    guard = JournalReconcilerProfitProtection(
        broker,
        arm_ccy=1.0,
        giveback_ccy=0.5,
        journal=journal,
    )

    closed = guard.process_open_trades([])

    assert closed == ["trade-42"]
    row = _read_exit(journal, "trade-42")
    assert row is not None
    assert row[0] == "2026-07-13T01:00:20+00:00"
    assert row[1] == pytest.approx(0.659)
    assert row[2] == pytest.approx(-1.25)
    assert row[3] == "BROKER_CLOSED"
    assert row[4] == 1


def test_legacy_order_id_maps_to_broker_trade_before_recovery(tmp_path):
    journal = TradeJournal(tmp_path / "trade_journal.db")
    _record_entry(journal, "order-1")
    broker = ClosedTradeBroker(
        {
            "trade-99": {
                "id": "trade-99",
                "instrument": "AUD_USD",
                "state": "CLOSED",
                "currentUnits": "0",
                "realizedPL": "0.85",
                "averageClosePrice": "0.66100",
                "closeTime": "2026-07-13T01:00:15Z",
            }
        }
    )
    guard = JournalReconcilerProfitProtection(
        broker,
        arm_ccy=1.0,
        giveback_ccy=0.5,
        journal=journal,
    )

    closed = guard.process_open_trades([])

    assert closed == ["trade-99"]
    row = _read_exit(journal, "order-1")
    assert row is not None
    assert row[0] == "2026-07-13T01:00:15+00:00"
    assert row[1] == pytest.approx(0.661)
    assert row[2] == pytest.approx(0.85)
    assert row[3] == "BROKER_CLOSED"
    with sqlite3.connect(journal.path) as conn:
        orphan = conn.execute(
            "SELECT COUNT(*) FROM trades WHERE trade_id = 'trade-99'"
        ).fetchone()[0]
    assert orphan == 0


def test_state_dir_prefers_mounted_render_disk(tmp_path, monkeypatch):
    monkeypatch.delenv("MOSSY_STATE_PATH", raising=False)
    explicit = tmp_path / "explicit-state"
    monkeypatch.setenv("MOSSY_STATE_PATH", str(explicit))
    assert resolve_state_dir(tmp_path / "fallback") == explicit
