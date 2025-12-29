from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.profit_protection import ProfitProtection
from src.decision_engine import Evaluation
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
        if instrument.endswith("JPY"):
            return 0.01
        if instrument.startswith("XAU"):
            return 0.1
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
        def sl_distance_from_atr(self, atr, instrument=None):
            return 0.5
        def tp_distance_from_atr(self, atr, instrument=None):
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
        time_stop_minutes=9999,
        time_stop_min_pips=9999,
    )
    closed = aggressive_guard.process_open_trades(trades)
    assert closed == ["T-EXIT"]
    assert aggressive_broker.closed[0]["trade_id"] == "T-EXIT"

    normal_broker = DummyBroker()
    normal_guard = ProfitProtection(
        normal_broker,
        aggressive=False,
        aggressive_max_hold_minutes=45,
        time_stop_minutes=9999,
        time_stop_min_pips=9999,
    )
    closed = normal_guard.process_open_trades(trades)
    assert closed == []
    assert normal_broker.closed == []

    output = capsys.readouterr().out
    assert "[TIME-EXIT] Closing EUR_USD" in output
    assert output.count("[TIME-EXIT]") == 1


def test_time_stop_uses_atr_fraction_for_xau(capsys):
    opened_at = datetime.now(timezone.utc) - timedelta(minutes=120)
    trade = {
        "id": "XAU-TS",
        "instrument": "XAU_USD",
        "currentUnits": 100,
        "openTime": opened_at.isoformat(),
        "unrealizedPips": 0.5,
        "atr": 0.8,
    }
    broker = DummyBroker()
    guard = ProfitProtection(
        broker,
        time_stop_minutes=60,
        time_stop_min_pips=2.0,
        time_stop_xau_atr_mult=0.5,
    )

    closed = guard.process_open_trades([trade], now_utc=opened_at + timedelta(minutes=120))

    assert closed == ["XAU-TS"]
    assert broker.closed == [{"trade_id": "XAU-TS", "instrument": "XAU_USD"}]
    output = capsys.readouterr().out
    assert "[TIME-STOP] TIME_STOP XAU_USD" in output


def test_time_stop_runs_when_entries_blocked(monkeypatch):
    class TimeStopBroker(DummyBroker):
        def account_equity(self):
            return 10_000.0
        def close_all_positions(self):
            pass

    class DummyRisk:
        risk_per_trade_pct = 0.001
        demo_mode = False
        def enforce_equity_floor(self, *args, **kwargs):
            pass
        def should_open(self, *args, **kwargs):
            return False, "daily-profit-cap"
        def sl_distance_from_atr(self, atr, instrument=None):
            return 0.0
        def tp_distance_from_atr(self, atr, instrument=None):
            return 0.0
        def register_entry(self, *args, **kwargs):
            pass
        def register_exit(self, *args, **kwargs):
            pass

    class DummyEngine:
        def __init__(self):
            self.marked = []
        def evaluate_all(self):
            return [
                Evaluation(
                    instrument="GBP_USD",
                    signal="BUY",
                    diagnostics={
                        "atr": 0.01,
                        "atr_baseline_50": 0.01,
                        "close": 1.2345,
                        "ema_trend_fast": 1.25,
                        "ema_trend_slow": 1.2,
                    },
                    reason="trend",
                    market_active=True,
                )
            ]
        def mark_trade(self, instrument):
            self.marked.append(instrument)

    stale_open = (datetime.now(timezone.utc) - timedelta(minutes=120)).isoformat()
    open_trade = {
        "id": "STALE",
        "instrument": "EUR_USD",
        "currentUnits": 1000,
        "openTime": stale_open,
        "unrealizedPips": 1.0,
    }
    broker = TimeStopBroker()
    guard = ProfitProtection(broker, time_stop_minutes=1, time_stop_min_pips=2.0)
    monkeypatch.setattr(main_mod, "profit_guard", guard)
    monkeypatch.setattr(main_mod, "broker", broker)
    monkeypatch.setattr(main_mod, "risk", DummyRisk())
    monkeypatch.setattr(main_mod, "engine", DummyEngine())
    monkeypatch.setattr(main_mod, "_open_trades_state", lambda: [open_trade])
    monkeypatch.setattr(main_mod.session_filter, "is_entry_session", lambda *args, **kwargs: True)
    monkeypatch.setattr(main_mod.position_sizer, "units_for_risk", lambda *args, **kwargs: 0)

    main_mod.asyncio.run(main_mod.decision_cycle())

    assert {"trade_id": "STALE", "instrument": "EUR_USD"} in broker.closed


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
