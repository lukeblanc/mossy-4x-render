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
    monkeypatch.setenv("HARD_MAX_LOSS_CCY", "1.50")
    broker = DummyBroker([-1.51])
    guard = SmartExitGuard(broker, aggressive=False)

    open_trades = [_trade(-1.51)]
    closed = guard.process_open_trades(open_trades)

    assert closed == ["T1"]
    assert broker.closed == ["AUD_USD"]


def test_break_even_protection_after_early_profit(monkeypatch):
    monkeypatch.setenv("PROFIT_PROTECT_TRIGGER_CCY", "1.50")
    monkeypatch.setenv("PROFIT_PROTECT_FLOOR_CCY", "0.00")
    monkeypatch.setenv("PROFIT_TRAIL_ARM_CCY", "3.00")
    broker = DummyBroker([1.50, -0.01])
    guard = SmartExitGuard(broker, aggressive=False)

    assert guard.process_open_trades([_trade(1.50)]) == []
    closed = guard.process_open_trades([_trade(-0.01)])

    assert closed == ["T1"]


def test_profit_trail_banks_after_fifty_cent_giveback(monkeypatch):
    monkeypatch.setenv("PROFIT_TRAIL_ARM_CCY", "3.00")
    monkeypatch.setenv("PROFIT_TRAIL_GIVEBACK_CCY", "0.50")
    broker = DummyBroker([3.00, 4.00, 3.49])
    guard = SmartExitGuard(broker, aggressive=False)

    assert guard.process_open_trades([_trade(3.00)]) == []
    assert guard.process_open_trades([_trade(4.00)]) == []
    closed = guard.process_open_trades([_trade(3.49)])

    assert closed == ["T1"]
