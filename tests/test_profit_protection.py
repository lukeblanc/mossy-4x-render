from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

import src.profit_protection as profit_protection
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


def _trade(trade_id: str, instrument: str, units: float, profit: float | None = None, pips: float | None = None):
    payload = {
        "id": trade_id,
        "instrument": instrument,
        "currentUnits": units,
    }
    if profit is not None:
        payload["unrealizedPL"] = profit
    if pips is not None:
        payload["unrealizedPips"] = pips
    return payload


def test_trailing_giveback_closes_at_floor(capsys):
    broker = DummyBroker()
    guard = ProfitProtection(
        broker,
        arm_ccy=0.75,
        giveback_ccy=0.5,
    )

    trade = _trade("T1", "EUR_USD", 1000, profit=0.0)
    guard.process_open_trades([trade])

    trade = _trade("T1", "EUR_USD", 1000, profit=0.8)
    guard.process_open_trades([trade])

    trade = _trade("T1", "EUR_USD", 1000, profit=1.20)
    guard.process_open_trades([trade])

    trade = _trade("T1", "EUR_USD", 1000, profit=0.6)
    closed = guard.process_open_trades([trade])

    assert closed == ["T1"]
    assert broker.closed == [{"trade_id": "T1", "instrument": "EUR_USD"}]

    out = capsys.readouterr().out
    assert "[TRAIL] armed ticket=T1 profit_ccy=0.80" in out
    assert "[TRAIL] close ticket=T1 current_profit=0.60 floor=0.70 high_water=1.20 reason=pnl_profit_protection" in out


def test_multiple_positions_do_not_share_state():
    broker = DummyBroker()
    guard = ProfitProtection(broker, arm_ccy=0.5, giveback_ccy=0.25)

    t1 = _trade("T1", "EUR_USD", 1000, profit=0.6)
    t2 = _trade("T2", "EUR_USD", -1000, profit=0.3)
    guard.process_open_trades([t1, t2])

    # Only T1 should be armed and have a high-water update
    snap = guard.snapshot()
    assert "T1" in snap and snap["T1"].max_profit_ccy == pytest.approx(0.6)
    assert "T2" in snap and snap["T2"].max_profit_ccy == pytest.approx(0.3)
    assert snap["T1"].armed is True
    assert snap["T2"].armed is False

    # Drop T1 to trigger close while T2 climbs without being capped by T1's state
    t1_drop = _trade("T1", "EUR_USD", 1000, profit=0.2)
    t2_rise = _trade("T2", "EUR_USD", -1000, profit=0.6)
    closed = guard.process_open_trades([t1_drop, t2_rise])

    assert "T1" in closed
    assert "T2" not in closed
    assert any(entry.get("trade_id") == "T1" for entry in broker.closed)
    # T2 should retain its state
    assert guard.snapshot()["T2"].max_profit_ccy == pytest.approx(0.6)


def test_closeout_missing_treated_as_closed_when_gone(capsys):
    class CloseoutBroker(DummyBroker):
        def __init__(self):
            super().__init__()
            self.trades = []

        def close_trade(self, trade_id: str, instrument: str | None = None):
            return {"status": "ERROR", "code": 400, "errorCode": "CLOSEOUT_POSITION_DOESNT_EXIST"}

        def list_open_trades(self):
            return []

    broker = CloseoutBroker()
    guard = ProfitProtection(broker, arm_ccy=0.5, giveback_ccy=0.25)

    armed = _trade("T-MISS", "GBP_USD", 1000, profit=0.8)
    guard.process_open_trades([armed])
    drop = _trade("T-MISS", "GBP_USD", 1000, profit=0.2)
    closed = guard.process_open_trades([drop])

    assert closed == ["T-MISS"]
    out = capsys.readouterr().out
    assert "[TRAIL][INFO] Broker missing position – treated as closed ticket=T-MISS instrument=GBP_USD" in out


def test_closeout_missing_warns_when_position_still_open(capsys):
    class StickyBroker(DummyBroker):
        def __init__(self):
            super().__init__()
            self.trades = [{"id": "T-STICK", "instrument": "GBP_USD"}]

        def close_trade(self, trade_id: str, instrument: str | None = None):
            return {"status": "ERROR", "code": 400, "errorCode": "CLOSEOUT_POSITION_DOESNT_EXIST"}

        def list_open_trades(self):
            return list(self.trades)

    broker = StickyBroker()
    guard = ProfitProtection(broker, arm_ccy=0.5, giveback_ccy=0.25)

    armed = _trade("T-STICK", "GBP_USD", 1000, profit=0.8)
    guard.process_open_trades([armed])
    drop = _trade("T-STICK", "GBP_USD", 1000, profit=0.2)
    closed = guard.process_open_trades([drop])

    assert closed == ["T-STICK"]
    out = capsys.readouterr().out
    assert "Broker missing position – treated as closed ticket=T-STICK instrument=GBP_USD" in out


def test_payload_with_missing_position_marks_closed(capsys):
    class PayloadBroker(DummyBroker):
        def __init__(self):
            super().__init__()
            self.trades = [{"id": "T-PAY", "instrument": "EUR_USD"}]

        def close_trade(self, trade_id: str, instrument: str | None = None):
            payload = {
                "longOrderRejectTransaction": {
                    "rejectReason": "CLOSEOUT_POSITION_DOESNT_EXIST",
                    "instrument": instrument,
                },
                "errorCode": "CLOSEOUT_POSITION_DOESNT_EXIST",
                "errorMessage": "The Position requested to be closed out does not exist",
            }
            return {"status": "ERROR", "code": 400, "text": json.dumps(payload)}

    broker = PayloadBroker()
    guard = ProfitProtection(broker, arm_ccy=0.5, giveback_ccy=0.25)

    armed = _trade("T-PAY", "EUR_USD", 1000, profit=0.8)
    guard.process_open_trades([armed])
    drop = _trade("T-PAY", "EUR_USD", 1000, profit=0.2)
    closed = guard.process_open_trades([drop])

    assert closed == ["T-PAY"]
    out = capsys.readouterr().out
    assert "[TRAIL][INFO] Broker missing position – treated as closed ticket=T-PAY instrument=EUR_USD" in out


def test_zero_units_snapshot_treated_as_closed(capsys):
    class ZeroUnitsBroker(DummyBroker):
        def __init__(self):
            super().__init__()
            self.trades = [{"id": "T-ZERO", "instrument": "EUR_USD", "units": 0}]

        def close_trade(self, trade_id: str, instrument: str | None = None):
            return {"status": "ERROR", "code": 400, "errorCode": "CLOSEOUT_POSITION_DOESNT_EXIST"}

        def list_open_trades(self):
            return list(self.trades)

    broker = ZeroUnitsBroker()
    guard = ProfitProtection(broker, arm_ccy=0.5, giveback_ccy=0.25)

    armed = _trade("T-ZERO", "EUR_USD", 1000, profit=0.8)
    guard.process_open_trades([armed])
    drop = _trade("T-ZERO", "EUR_USD", 1000, profit=0.2)
    closed = guard.process_open_trades([drop])

    assert closed == ["T-ZERO"]
    out = capsys.readouterr().out
    assert "[TRAIL][INFO] Broker missing position – treated as closed ticket=T-ZERO instrument=EUR_USD" in out


def test_missing_position_reconciles_without_warn(capsys):
    class MissingPositionBroker(DummyBroker):
        def __init__(self):
            super().__init__()
            self.trades = []

        def close_trade(self, trade_id: str, instrument: str | None = None):
            return {
                "status": "ERROR",
                "code": 400,
                "errorCode": "CLOSEOUT_POSITION_DOESNT_EXIST",
                "longOrderRejectTransaction": {"rejectReason": "POSITION_CLOSEOUT_DOESNT_EXIST"},
            }

        def list_open_trades(self):
            return list(self.trades)

    broker = MissingPositionBroker()
    guard = ProfitProtection(broker, arm_ccy=0.5, giveback_ccy=0.25)

    armed = _trade("T-404", "USD_JPY", 1000, profit=0.8)
    open_trades = [armed]
    guard.process_open_trades(open_trades)

    drop = _trade("T-404", "USD_JPY", 1000, profit=0.1)
    open_trades = [drop]
    closed = guard.process_open_trades(open_trades)

    assert closed == ["T-404"]
    assert open_trades == []
    assert guard.snapshot() == {}

    out = capsys.readouterr().out
    assert "[TRAIL][INFO] Broker missing position – treated as closed ticket=T-404 instrument=USD_JPY" in out
    assert "[WARN]" not in out


def test_missing_position_not_retried(capsys):
    class NoRetryBroker(DummyBroker):
        def __init__(self):
            super().__init__()
            self.calls = 0
        def close_trade(self, trade_id: str, instrument: str | None = None):
            self.calls += 1
            return {"status": "ERROR", "code": 400, "errorCode": "CLOSEOUT_POSITION_DOESNT_EXIST"}
        def list_open_trades(self):
            return []

    broker = NoRetryBroker()
    guard = ProfitProtection(broker, arm_ccy=0.5, giveback_ccy=0.25)

    armed = _trade("T-NORETRY", "EUR_USD", 1000, profit=0.8)
    guard.process_open_trades([armed])
    drop = _trade("T-NORETRY", "EUR_USD", 1000, profit=0.1)
    guard.process_open_trades([drop])

    # Second call with same trade should be ignored because it's locally closed.
    guard.process_open_trades([drop])

    assert broker.calls == 1
    out = capsys.readouterr().out
    assert out.count("Broker missing position – treated as closed ticket=T-NORETRY instrument=EUR_USD") == 1


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
    monkeypatch.setattr(
        main_mod.session_filter,
        "session_decision",
        lambda *args, **kwargs: main_mod.session_filter.SessionDecision(True, True, None, "STRICT"),
    )
    monkeypatch.setattr(main_mod.position_sizer, "units_for_risk", lambda *args, **kwargs: 0)

    main_mod.asyncio.run(main_mod.decision_cycle())

    assert {"trade_id": "STALE", "instrument": "EUR_USD"} in broker.closed


def test_demo_trailing_thresholds():
    broker = DummyBroker()

    demo_guard = main_mod._profit_guard_for_mode("demo", broker)
    live_guard = main_mod._profit_guard_for_mode("live", broker)
    paper_guard = main_mod._profit_guard_for_mode("paper", broker)

    assert demo_guard.trigger == pytest.approx(profit_protection.ARM_AT_CCY)
    assert demo_guard.trail == pytest.approx(profit_protection.GIVEBACK_CCY)
    assert live_guard.trigger == pytest.approx(profit_protection.ARM_AT_CCY)
    assert live_guard.trail == pytest.approx(profit_protection.GIVEBACK_CCY)
    assert paper_guard.trigger == pytest.approx(profit_protection.ARM_AT_CCY)
    assert paper_guard.trail == pytest.approx(profit_protection.GIVEBACK_CCY)
