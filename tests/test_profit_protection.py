from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.profit_protection import ProfitProtection
from src import main as main_mod


class DummyBroker:
    def __init__(self, profits=None, pips=None):
        self.profits = {k: list(v) for k, v in (profits or {}).items()}
        self.pips = {k: list(v) for k, v in (pips or {}).items()}
        self.closed = []

    def get_unrealized_profit(self, instrument: str):
        values = self.profits.get(instrument)
        if not values:
            return 0.0
        return values.pop(0)

    def close_position(self, instrument: str):
        self.closed.append({"instrument": instrument})
        return {"status": "SIMULATED"}

    def close_trade(self, trade_id: str, instrument: str | None = None):
        self.closed.append({"trade_id": trade_id, "instrument": instrument})
        return {"status": "SIMULATED"}

    def current_spread(self, instrument: str):
        return 0.2

    def _pip_size(self, instrument: str) -> float:
        return 0.0001


def _trade(trade_id: str, instrument: str, units: float, pips: float | None = None, profit: float | None = None):
    payload = {
        "id": trade_id,
        "instrument": instrument,
        "currentUnits": units,
    }
    if pips is not None:
        payload["unrealizedPips"] = pips
    if profit is not None:
        payload["unrealizedPL"] = profit
    return payload


def test_trailing_giveback_closes_at_floor(capsys):
    broker = DummyBroker()
    guard = ProfitProtection(
        broker,
        arm_pips=8,
        giveback_pips=4,
        use_pips=True,
        be_arm_pips=20,  # keep break-even out of the way
    )

    trade = _trade("T1", "EUR_USD", 1000, pips=0)
    guard.process_open_trades([trade])

    trade = _trade("T1", "EUR_USD", 1000, pips=10)
    guard.process_open_trades([trade])

    trade = _trade("T1", "EUR_USD", 1000, pips=12)
    guard.process_open_trades([trade])

    trade = _trade("T1", "EUR_USD", 1000, pips=7)
    closed = guard.process_open_trades([trade])

    assert closed == ["T1"]
    assert broker.closed == [{"trade_id": "T1", "instrument": "EUR_USD"}]

    out = capsys.readouterr().out
    assert "[TRAIL] armed ticket=T1 profit_pips=10.00" in out
    assert "[TRAIL] close ticket=T1 current_pips=7.00 floor=8.00 high_water=12.00 reason=TRAIL_GIVEBACK" in out


def test_multiple_positions_do_not_share_state():
    broker = DummyBroker()
    guard = ProfitProtection(broker, arm_pips=5, giveback_pips=2, use_pips=True)

    t1 = _trade("T1", "EUR_USD", 1000, pips=6)
    t2 = _trade("T2", "EUR_USD", -1000, pips=3)
    guard.process_open_trades([t1, t2])

    # Only T1 should be armed and have a high-water update
    snap = guard.snapshot()
    assert "T1" in snap and snap["T1"].high_water_pips == pytest.approx(6)
    assert "T2" in snap and snap["T2"].high_water_pips == pytest.approx(3)
    assert snap["T1"].trail_active is True
    assert snap["T2"].trail_active is False

    # Drop T1 to trigger close while T2 climbs without being capped by T1's state
    t1_drop = _trade("T1", "EUR_USD", 1000, pips=3)
    t2_rise = _trade("T2", "EUR_USD", -1000, pips=6)
    closed = guard.process_open_trades([t1_drop, t2_rise])

    assert "T1" in closed
    assert "T2" not in closed
    assert any(entry.get("trade_id") == "T1" for entry in broker.closed)
    # T2 should retain its state
    assert guard.snapshot()["T2"].high_water_pips == pytest.approx(6)


def test_daily_profit_cap_does_not_block_trailing(monkeypatch):
    class DummyGuard:
        def __init__(self):
            self.calls = 0
        def process_open_trades(self, trades, **_):
            self.calls += 1
            return ["EUR_USD"]

    class DummyRisk:
        risk_per_trade_pct = 0.001
        demo_mode = False
        def __init__(self):
            self.registered = []
        def enforce_equity_floor(self, *args, **kwargs):
            pass
        def should_open(self, *args, **kwargs):
            return False, "daily_profit_cap"
        def sl_distance_from_atr(self, atr):
            return 0.5
        def tp_distance_from_atr(self, atr):
            return 1.0
        def register_entry(self, now_utc, instrument: str):
            self.registered.append(instrument)
        def register_exit(self, *args, **kwargs):
            pass

    class DummyEngine:
        def __init__(self):
            self.marked = []
        def evaluate_all(self):
            return []
        def mark_trade(self, instrument):
            self.marked.append(instrument)

    class DummyBroker:
        def __init__(self):
            self.calls = []
        def account_equity(self):
            return 10000.0
        def close_all_positions(self):
            pass

    dummy_guard = DummyGuard()
    dummy_risk = DummyRisk()
    dummy_engine = DummyEngine()
    dummy_broker = DummyBroker()

    monkeypatch.setattr(main_mod, "profit_guard", dummy_guard)
    monkeypatch.setattr(main_mod, "risk", dummy_risk)
    monkeypatch.setattr(main_mod, "engine", dummy_engine)
    monkeypatch.setattr(main_mod, "broker", dummy_broker)
    monkeypatch.setattr(main_mod, "_open_trades_state", lambda: [{"instrument": "EUR_USD"}])

    main_mod.asyncio.run(main_mod.decision_cycle())

    assert dummy_guard.calls == 1  # trailing ran
    assert dummy_engine.marked == []  # no new entries
    assert dummy_risk.registered == []  # no entries registered


def test_time_exit_only_in_aggressive_mode(capsys):
    open_time = (datetime.now(timezone.utc) - timedelta(minutes=90)).isoformat()
    trades = [{"id": "T-EXIT", "instrument": "EUR_USD", "currentUnits": "100", "openTime": open_time, "unrealizedPips": -1.0}]

    aggressive_broker = DummyBroker()
    aggressive_guard = ProfitProtection(
        aggressive_broker,
        aggressive=True,
        aggressive_max_hold_minutes=45,
    )
    closed = aggressive_guard.process_open_trades(trades)
    assert closed == ["T-EXIT"]
    assert aggressive_broker.closed[0]["trade_id"] == "T-EXIT"

    normal_broker = DummyBroker()
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


def test_demo_trailing_thresholds():
    broker = DummyBroker()

    demo_guard = main_mod._profit_guard_for_mode("demo", broker)
    live_guard = main_mod._profit_guard_for_mode("live", broker)
    paper_guard = main_mod._profit_guard_for_mode("paper", broker)

    assert demo_guard.trigger == pytest.approx(1.0)
    assert demo_guard.trail == pytest.approx(0.5)
    assert live_guard.trigger == pytest.approx(3.0)
    assert live_guard.trail == pytest.approx(0.5)
    assert paper_guard.trigger == pytest.approx(3.0)
    assert paper_guard.trail == pytest.approx(0.5)
