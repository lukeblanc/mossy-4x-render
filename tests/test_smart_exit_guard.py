from __future__ import annotations

from src.smart_exit_guard import SmartExitGuard


class DummyBroker:
    def __init__(self, profits):
        self.profits = list(profits)
        self.closed = []

    def get_unrealized_profit(self, instrument: str):
        if not self.profits:
            return 0.0
        return self.profits.pop(0)

    def close_position_side(self, instrument: str, long_units: float, short_units: float):
        self.closed.append(instrument)
        return {"status": "SIMULATED"}

    def close_trade(self, trade_id: str, instrument: str | None = None):
        self.closed.append(instrument or trade_id)
        return {"status": "SIMULATED"}

    def current_spread(self, instrument: str):
        return 0.2

    def _pip_size(self, instrument: str):
        return 0.0001

    def list_open_trades(self):
        return None

    def position_snapshot(self, instrument: str):
        return None


def _trade(profit: float):
    return {
        "id": "T1",
        "instrument": "AUD_USD",
        "currentUnits": 1000,
        "unrealizedPL": profit,
    }


def test_hard_cash_loss_floor_is_active_without_aggressive_mode(monkeypatch):
    monkeypatch.setenv("HARD_MAX_LOSS_CCY", "1.25")
    broker = DummyBroker([-1.26])
    guard = SmartExitGuard(broker, aggressive=False)

    open_trades = [_trade(-1.26)]
    closed = guard.process_open_trades(open_trades)

    assert closed == ["T1"]
    assert broker.closed == ["AUD_USD"]


def test_winner_protection_retains_share_of_early_peak(monkeypatch):
    monkeypatch.setenv("PROFIT_PROTECT_TRIGGER_CCY", "2.00")
    monkeypatch.setenv("PROFIT_PROTECT_FLOOR_CCY", "0.25")
    monkeypatch.setenv("PROFIT_PROTECT_CAPTURE_RATIO", "0.35")
    monkeypatch.setenv("PROFIT_TRAIL_ARM_CCY", "3.00")
    broker = DummyBroker([2.00, 0.69])
    guard = SmartExitGuard(broker, aggressive=False)

    assert guard.process_open_trades([_trade(2.00)]) == []
    closed = guard.process_open_trades([_trade(0.69)])

    assert closed == ["T1"]


def test_winner_protection_does_not_clip_trade_before_trigger(monkeypatch):
    monkeypatch.setenv("PROFIT_PROTECT_TRIGGER_CCY", "2.00")
    monkeypatch.setenv("PROFIT_PROTECT_CAPTURE_RATIO", "0.35")
    broker = DummyBroker([1.90, 0.20])
    guard = SmartExitGuard(broker, aggressive=False)

    assert guard.process_open_trades([_trade(1.90)]) == []
    assert guard.process_open_trades([_trade(0.20)]) == []
    assert broker.closed == []


def test_profit_trail_uses_fixed_and_percentage_capture_floors(monkeypatch):
    monkeypatch.setenv("PROFIT_TRAIL_ARM_CCY", "3.00")
    monkeypatch.setenv("PROFIT_TRAIL_GIVEBACK_CCY", "0.75")
    monkeypatch.setenv("PROFIT_TRAIL_MIN_CAPTURE_RATIO", "0.65")
    broker = DummyBroker([3.00, 4.00, 3.24])
    guard = SmartExitGuard(broker, aggressive=False)

    assert guard.process_open_trades([_trade(3.00)]) == []
    assert guard.process_open_trades([_trade(4.00)]) == []
    closed = guard.process_open_trades([_trade(3.24)])

    assert closed == ["T1"]
