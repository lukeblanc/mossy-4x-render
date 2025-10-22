from __future__ import annotations

import pytest

from src.profit_protection import ProfitProtection


class DummyBroker:
    def __init__(self, profits):
        self.profits = {k: list(v) for k, v in profits.items()}
        self.closed = []

    def get_unrealized_profit(self, instrument: str):
        values = self.profits.get(instrument)
        if not values:
            return 0.0
        return values.pop(0)

    def close_position(self, instrument: str):
        self.closed.append(instrument)
        return {"status": "SIMULATED"}


def test_trailing_rule_closes_on_drawdown(capsys):
    broker = DummyBroker({"EUR_USD": [3.25, 2.6]})
    guard = ProfitProtection(broker)
    trades = [{"instrument": "EUR_USD", "currentUnits": "100"}]

    # First pass should establish the high-water mark without closing
    closed = guard.process_open_trades(trades)
    assert closed == []
    assert broker.closed == []

    # Second pass drops $0.65 from the high-water mark and should close
    trades = [{"instrument": "EUR_USD", "currentUnits": "100"}]
    closed = guard.process_open_trades(trades)
    assert closed == ["EUR_USD"]
    assert broker.closed == ["EUR_USD"]

    captured = capsys.readouterr().out
    assert "[TRAIL] Closed EUR_USD" in captured
    assert "fell $0.65" in captured


def test_state_clears_when_positions_exit():
    broker = DummyBroker({"GBP_USD": [3.5]})
    guard = ProfitProtection(broker)
    trades = [{"instrument": "GBP_USD", "currentUnits": "100"}]

    guard.process_open_trades(trades)
    assert guard.snapshot()["GBP_USD"] == pytest.approx(3.5)

    # No open trades -> cleanup should remove the high-water mark
    guard.process_open_trades([])
    assert guard.snapshot() == {}
