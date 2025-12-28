from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.profit_protection import ProfitProtection
from src import main as main_mod


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


def test_demo_trailing_thresholds():
    broker = DummyBroker({})

    demo_guard = main_mod._profit_guard_for_mode("demo", broker)
    live_guard = main_mod._profit_guard_for_mode("live", broker)
    paper_guard = main_mod._profit_guard_for_mode("paper", broker)

    assert demo_guard.trigger == pytest.approx(1.0)
    assert demo_guard.trail == pytest.approx(0.5)
    assert live_guard.trigger == pytest.approx(3.0)
    assert live_guard.trail == pytest.approx(0.5)
    assert paper_guard.trigger == pytest.approx(3.0)
    assert paper_guard.trail == pytest.approx(0.5)


def test_time_exit_only_in_aggressive_mode(capsys):
    open_time = (datetime.now(timezone.utc) - timedelta(minutes=90)).isoformat()
    trades = [{"instrument": "EUR_USD", "currentUnits": "100", "openTime": open_time}]

    aggressive_broker = DummyBroker({"EUR_USD": [-1.0]})
    aggressive_guard = ProfitProtection(
        aggressive_broker,
        aggressive=True,
        aggressive_max_hold_minutes=45,
    )
    closed = aggressive_guard.process_open_trades(trades)
    assert closed == ["EUR_USD"]
    assert aggressive_broker.closed == ["EUR_USD"]

    normal_broker = DummyBroker({"EUR_USD": [-1.0]})
    normal_guard = ProfitProtection(
        normal_broker,
        aggressive=False,
        aggressive_max_hold_minutes=45,
    )
    closed = normal_guard.process_open_trades(trades)
    assert closed == []
    assert normal_broker.closed == []

    output = capsys.readouterr().out
    assert "[TIME-EXIT] Closing EUR_USD" in output
    assert output.count("[TIME-EXIT]") == 1


def test_loss_floor_only_in_aggressive_mode(capsys):
    trades = [
        {"instrument": "GBP_USD", "currentUnits": "100", "openTime": datetime.now(timezone.utc).isoformat()},
        {
            "instrument": "AUD_USD",
            "currentUnits": "100",
            "openTime": datetime.now(timezone.utc).isoformat(),
            "atr": 2.0,
        },
    ]

    aggressive_broker = DummyBroker({"GBP_USD": [-6.0], "AUD_USD": [-2.5]})
    aggressive_guard = ProfitProtection(
        aggressive_broker,
        aggressive=True,
        aggressive_max_loss_usd=5.0,
        aggressive_max_loss_atr_mult=1.0,
    )
    closed = aggressive_guard.process_open_trades(trades)
    assert set(closed) == {"GBP_USD", "AUD_USD"}
    assert aggressive_broker.closed == ["GBP_USD", "AUD_USD"]

    normal_broker = DummyBroker({"GBP_USD": [-6.0], "AUD_USD": [-2.5]})
    normal_guard = ProfitProtection(
        normal_broker,
        aggressive=False,
        aggressive_max_loss_usd=5.0,
        aggressive_max_loss_atr_mult=1.0,
    )
    closed = normal_guard.process_open_trades(trades)
    assert closed == []
    assert normal_broker.closed == []

    output = capsys.readouterr().out
    assert "[LOSS-FLOOR] Closing GBP_USD" in output
    assert "[LOSS-FLOOR] Closing AUD_USD" in output
    assert output.count("[LOSS-FLOOR]") == 2
