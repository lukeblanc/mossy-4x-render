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
    def __init__(self, profits=None, pips=None, open_trades=None):
        self.profits = {k: list(v) for k, v in (profits or {}).items()}
        self.pips = {k: list(v) for k, v in (pips or {}).items()}
        self.closed = []
        self.trades = list(open_trades) if open_trades is not None else None
        self.side_payloads = []

    def get_unrealized_profit(self, instrument: str):
        values = self.profits.get(instrument)
        if not values:
            return 0.0
        return values.pop(0)

    def close_position_side(self, instrument: str, long_units: float, short_units: float):
        payload = {}
        if long_units > 0 and short_units == 0:
            payload = {"longUnits": "ALL"}
        elif short_units < 0 and long_units == 0:
            payload = {"shortUnits": "ALL"}
        elif long_units != 0 or short_units != 0:
            payload = {"longUnits": "ALL", "shortUnits": "ALL"}
        else:
            payload = {"longUnits": "0", "shortUnits": "0"}
        self.side_payloads.append(payload)
        self.closed.append({"instrument": instrument, "payload": payload})
        return {"status": "SIMULATED"}

    def close_position(self, instrument: str, *, long_units="ALL", short_units="ALL", trade_id=None):
        try:
            long_val = float(long_units)
        except (TypeError, ValueError):
            long_val = 0.0
        try:
            short_val = float(short_units)
        except (TypeError, ValueError):
            short_val = 0.0
        return self.close_position_side(instrument, long_val, short_val)

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

    def list_open_trades(self):
        if self.trades is None:
            return None
        return list(self.trades)

    def position_snapshot(self, instrument: str):
        if self.trades is None:
            return None
        for trade in self.trades:
            if trade.get("instrument") != instrument:
                continue
            raw_units = trade.get("currentUnits") or trade.get("units")
            try:
                units = float(raw_units)
            except (TypeError, ValueError):
                units = 0.0
            if units > 0:
                return {"instrument": instrument, "longUnits": str(units), "shortUnits": "0"}
            if units < 0:
                return {"instrument": instrument, "longUnits": "0", "shortUnits": str(units)}
            return {"instrument": instrument, "longUnits": "0", "shortUnits": "0"}
        return None


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


def test_close_trade_handles_missing_snapshot_without_units():
    broker = DummyBroker()
    guard = ProfitProtection(broker)

    result = guard._close_trade(  # noqa: SLF001
        "T-no-snapshot",
        "EUR_USD",
        profit=0.0,
        pips=None,
        floor=0.0,
        high_water=0.0,
        spread_pips=None,
        log_prefix="[TEST]",
        reason="MISSING_SNAPSHOT",
        summary=None,
        open_trades=None,
        state=None,
        units=None,
    )

    assert result is True
    assert broker.closed[-1]["payload"] == {"longUnits": "0", "shortUnits": "0"}


def test_trailing_giveback_closes_at_floor(capsys):
    broker = DummyBroker(profits={"EUR_USD": [0.0, 0.8, 1.2, 0.6]})
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
    assert broker.closed == [{"instrument": "EUR_USD", "payload": {"longUnits": "ALL"}}]

    out = capsys.readouterr().out
    assert "[TRAIL][INFO] armed ticket=T1 profit=0.80" in out
    assert "[TRAIL] close ticket=T1 current_profit=0.60 floor=0.70 high_water=1.20 reason=pnl_profit_protection" in out


def test_giveback_triggers_without_meeting_arm_threshold(capsys):
    broker = DummyBroker(profits={"EUR_USD": [0.4, 0.8, 0.25]})
    guard = ProfitProtection(broker, arm_ccy=1.0, giveback_ccy=0.5)

    trade = _trade("T-RETRACE", "EUR_USD", 1000, profit=0.4)
    guard.process_open_trades([trade])

    trade = _trade("T-RETRACE", "EUR_USD", 1000, profit=0.8)
    guard.process_open_trades([trade])

    trade = _trade("T-RETRACE", "EUR_USD", 1000, profit=0.25)
    closed = guard.process_open_trades([trade])

    assert closed == ["T-RETRACE"]
    assert broker.closed == [{"instrument": "EUR_USD", "payload": {"longUnits": "ALL"}}]

    out = capsys.readouterr().out
    assert "[TRAIL][INFO] giveback_met ticket=T-RETRACE instrument=EUR_USD profit=0.25" in out
    assert "[TRAIL][DEBUG] profit=0.80 high=0.80 floor=0.30 armed=False" in out


def test_trade_summary_emitted_once_with_profits(capsys):
    start = datetime.now(timezone.utc) - timedelta(seconds=30)
    broker = DummyBroker(profits={"EUR_USD": [1.2, 0.6]})
    guard = ProfitProtection(broker, arm_ccy=1.0, giveback_ccy=0.5)

    first = _trade("T-SUMMARY", "EUR_USD", 1000, profit=1.2)
    first["openTime"] = start.isoformat()
    guard.process_open_trades([first], now_utc=start)

    second = _trade("T-SUMMARY", "EUR_USD", 1000, profit=0.6)
    second["openTime"] = start.isoformat()
    closed = guard.process_open_trades([second], now_utc=start + timedelta(seconds=30))

    assert closed == ["T-SUMMARY"]
    out = capsys.readouterr().out
    assert out.count("[TRADE-SUMMARY]") == 1
    assert "max_profit_ccy=1.20" in out
    assert "final_profit_ccy=0.60" in out


def test_close_position_side_payload_has_no_units_key():
    broker = DummyBroker(profits={"EUR_USD": [1.2, 0.6]})
    guard = ProfitProtection(broker, arm_ccy=1.0, giveback_ccy=0.5)

    armed = [_trade("T-PAYLOAD", "EUR_USD", 1000, profit=1.2)]
    guard.process_open_trades(armed)

    giveback = [_trade("T-PAYLOAD", "EUR_USD", 1000, profit=0.6)]
    guard.process_open_trades(giveback)

    assert broker.side_payloads[0] == {"longUnits": "ALL"}
    assert "units" not in broker.side_payloads[0]


def test_multiple_positions_do_not_share_state():
    broker = DummyBroker(profits={"EUR_USD": [0.6, 0.3, 0.2, 0.6]})
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
    assert any(entry.get("instrument") == "EUR_USD" for entry in broker.closed)
    # T2 should retain its state
    assert guard.snapshot()["T2"].max_profit_ccy == pytest.approx(0.6)


def test_closeout_missing_treated_as_closed_when_gone(capsys):
    class CloseoutBroker(DummyBroker):
        def __init__(self):
            super().__init__(profits={"GBP_USD": [0.8, 0.2]}, open_trades=[{"id": "T-MISS", "instrument": "GBP_USD", "currentUnits": 1000}])

        def close_position_side(self, instrument: str, long_units: float, short_units: float):
            super().close_position_side(instrument, long_units, short_units)
            self.trades = []
            return {"status": "ERROR", "code": 400, "errorCode": "CLOSEOUT_POSITION_DOESNT_EXIST"}

        def list_open_trades(self):
            return list(self.trades)

    broker = CloseoutBroker()
    guard = ProfitProtection(broker, arm_ccy=0.5, giveback_ccy=0.25)

    armed = _trade("T-MISS", "GBP_USD", 1000, profit=0.8)
    guard.process_open_trades([armed])
    drop = _trade("T-MISS", "GBP_USD", 1000, profit=0.2)
    closed = guard.process_open_trades([drop])

    assert closed == ["T-MISS"]
    assert len(broker.side_payloads) == 1
    # subsequent cycles should not re-attempt once treated as closed
    guard.process_open_trades([drop])
    assert len(broker.side_payloads) == 1
    out = capsys.readouterr().out
    assert "treated_as_closed_after_missing_position ticket=T-MISS instrument=GBP_USD" in out


def test_closeout_missing_warns_when_position_still_open(capsys):
    class StickyBroker(DummyBroker):
        def __init__(self):
            super().__init__(profits={"GBP_USD": [0.8, 0.2]}, open_trades=[{"id": "T-STICK", "instrument": "GBP_USD", "currentUnits": 1000}])

        def close_position_side(self, instrument: str, long_units: float, short_units: float):
            super().close_position_side(instrument, long_units, short_units)
            return {"status": "ERROR", "code": 400, "errorCode": "CLOSEOUT_POSITION_DOESNT_EXIST"}

        def list_open_trades(self):
            return list(self.trades)

    broker = StickyBroker()
    guard = ProfitProtection(broker, arm_ccy=0.5, giveback_ccy=0.25)

    armed = _trade("T-STICK", "GBP_USD", 1000, profit=0.8)
    guard.process_open_trades([armed])
    drop = _trade("T-STICK", "GBP_USD", 1000, profit=0.2)
    closed = guard.process_open_trades([drop])

    assert closed == []
    assert len(broker.side_payloads) == 2  # initial attempt + retry only
    out = capsys.readouterr().out
    assert "missing_position_but_snapshot_still_open ticket=T-STICK" in out


def test_payload_with_missing_position_marks_closed(capsys):
    class PayloadBroker(DummyBroker):
        def __init__(self):
            super().__init__(profits={"EUR_USD": [0.8, 0.2]}, open_trades=[{"id": "T-PAY", "instrument": "EUR_USD", "currentUnits": 1000}])

        def close_position_side(self, instrument: str, long_units: float, short_units: float):
            super().close_position_side(instrument, long_units, short_units)
            self.trades = []
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
    assert "treated_as_closed_after_missing_position ticket=T-PAY instrument=EUR_USD" in out


def test_zero_units_snapshot_treated_as_closed(capsys):
    class ZeroUnitsBroker(DummyBroker):
        def __init__(self):
            super().__init__(profits={"EUR_USD": [0.8, 0.2]})
            self.trades = [{"id": "T-ZERO", "instrument": "EUR_USD", "units": 0}]

        def close_position_side(self, instrument: str, long_units: float, short_units: float):
            super().close_position_side(instrument, long_units, short_units)
            return {"status": "ERROR", "code": 400, "errorCode": "CLOSEOUT_POSITION_DOESNT_EXIST"}

        def list_open_trades(self):
            return list(self.trades)

    broker = ZeroUnitsBroker()
    guard = ProfitProtection(broker, arm_ccy=0.5, giveback_ccy=0.25)

    armed = _trade("T-ZERO", "EUR_USD", 1000, profit=0.8)
    first_closed = guard.process_open_trades([armed])
    drop = _trade("T-ZERO", "EUR_USD", 1000, profit=0.2)
    closed = guard.process_open_trades([drop])

    assert first_closed == ["T-ZERO"]
    assert closed == []
    out = capsys.readouterr().out
    assert "reconcile_missing_position ticket=T-ZERO instrument=EUR_USD action=mark_closed" in out


def test_missing_position_reconciles_without_warn(capsys):
    class MissingPositionBroker(DummyBroker):
        def __init__(self):
            super().__init__(profits={"USD_JPY": [0.8, 0.1]}, open_trades=[{"id": "T-404", "instrument": "USD_JPY", "currentUnits": 1000}])

        def close_position_side(self, instrument: str, long_units: float, short_units: float):
            super().close_position_side(instrument, long_units, short_units)
            self.trades = []
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
    assert "treated_as_closed_after_missing_position ticket=T-404 instrument=USD_JPY" in out
    assert "[WARN]" not in out


def test_missing_position_not_retried(capsys):
    class NoRetryBroker(DummyBroker):
        def __init__(self):
            super().__init__(profits={"EUR_USD": [0.8, 0.1, 0.1]}, open_trades=[{"id": "T-NORETRY", "instrument": "EUR_USD", "currentUnits": 1000}])
            self.calls = 0
        def close_position_side(self, instrument: str, long_units: float, short_units: float):
            self.calls += 1
            super().close_position_side(instrument, long_units, short_units)
            self.trades = []
            return {"status": "ERROR", "code": 400, "errorCode": "CLOSEOUT_POSITION_DOESNT_EXIST"}
        def list_open_trades(self):
            return list(self.trades)

    broker = NoRetryBroker()
    guard = ProfitProtection(broker, arm_ccy=0.5, giveback_ccy=0.25)

    armed = _trade("T-NORETRY", "EUR_USD", 1000, profit=0.8)
    guard.process_open_trades([armed])
    drop = _trade("T-NORETRY", "EUR_USD", 1000, profit=0.1)
    guard.process_open_trades([drop])

    # Second call with same trade should be ignored because it's locally closed.
    guard.process_open_trades([drop])

    assert broker.calls == 1
    assert len(broker.side_payloads) == 1
    out = capsys.readouterr().out
    assert out.count("treated_as_closed_after_missing_position ticket=T-NORETRY instrument=EUR_USD") == 1
    assert out.count("[TRADE-SUMMARY]") == 1


def test_trail_arms_at_one_and_closes_after_half_giveback(capsys):
    broker = DummyBroker(profits={"EUR_USD": [0.90, 1.00, 0.40]})
    guard = ProfitProtection(broker)  # defaults arm at 1.00, giveback 0.50

    open_trades = [_trade("T-ARM", "EUR_USD", 1000, profit=0.90)]
    closed = guard.process_open_trades(open_trades)
    assert closed == []

    out = capsys.readouterr().out
    assert "armed ticket=T-ARM" not in out
    assert any("[TRAIL][DEBUG]" in line for line in out.splitlines())

    open_trades = [_trade("T-ARM", "EUR_USD", 1000, profit=1.00)]
    closed = guard.process_open_trades(open_trades)
    assert closed == []
    out = capsys.readouterr().out
    assert "[TRAIL][INFO] armed ticket=T-ARM profit=1.00" in out

    open_trades = [_trade("T-ARM", "EUR_USD", 1000, profit=0.40)]
    closed = guard.process_open_trades(open_trades)
    assert closed == ["T-ARM"]
    out = capsys.readouterr().out
    assert "[TRAIL][DEBUG] profit=0.40" in out


def test_trailing_close_not_blocked_by_interval(capsys):
    class CountingBroker(DummyBroker):
        def __init__(self):
            super().__init__(profits={"EUR_USD": [1.0, 0.5]})
            self.calls = 0

        def close_position_side(self, instrument: str, long_units: float, short_units: float):
            self.calls += 1
            return super().close_position_side(instrument, long_units, short_units)

        def list_open_trades(self):
            return [{"id": "T-INTERVAL", "instrument": "EUR_USD", "currentUnits": 1000}]

    broker = CountingBroker()
    guard = ProfitProtection(broker, arm_ccy=0.5, giveback_ccy=0.5, min_check_interval_sec=60)

    now = datetime.now(timezone.utc)
    armed_trade = [_trade("T-INTERVAL", "EUR_USD", 1000, profit=1.0)]
    guard.process_open_trades(armed_trade, now_utc=now)

    giveback_trade = [_trade("T-INTERVAL", "EUR_USD", 1000, profit=0.5)]
    closed = guard.process_open_trades(giveback_trade, now_utc=now + timedelta(seconds=5))

    assert closed == ["T-INTERVAL"]
    assert broker.calls == 1
    assert broker.closed == [{"instrument": "EUR_USD", "payload": {"longUnits": "ALL"}}]
    out = capsys.readouterr().out
    assert "giveback_met ticket=T-INTERVAL instrument=EUR_USD" in out
    assert "attempting_close ticket=T-INTERVAL instrument=EUR_USD reason=pnl_profit_protection" in out


def test_trailing_not_blocked_by_cooldown_and_uses_broker_profit(capsys):
    class CooldownBroker(DummyBroker):
        def __init__(self):
            super().__init__(profits={"EUR_USD": [1.2, 0.46]})
            self.calls = 0
        def close_position_side(self, instrument: str, long_units: float, short_units: float):
            self.calls += 1
            return super().close_position_side(instrument, long_units, short_units)
        def list_open_trades(self):
            return [{"id": "T-COOL", "instrument": "EUR_USD", "currentUnits": 1000}]

    broker = CooldownBroker()
    guard = ProfitProtection(broker, arm_ccy=1.0, giveback_ccy=0.5, min_check_interval_sec=120)

    armed = [_trade("T-COOL", "EUR_USD", 1000, profit=1.2)]
    guard.process_open_trades(armed, now_utc=datetime.now(timezone.utc))

    giveback = [_trade("T-COOL", "EUR_USD", 1000, profit=0.46)]
    closed = guard.process_open_trades(giveback, now_utc=datetime.now(timezone.utc) + timedelta(seconds=1))

    assert closed == ["T-COOL"]
    assert broker.calls == 1
    out = capsys.readouterr().out
    assert "armed ticket=T-COOL" in out
    assert "giveback_met ticket=T-COOL instrument=EUR_USD profit=0.46 floor=0.70 high_water=1.20 giveback=0.50" in out


def test_manual_close_reconciles_and_not_retried(capsys):
    broker = DummyBroker(profits={"EUR_USD": [0.9]}, open_trades=[])
    guard = ProfitProtection(broker, arm_ccy=0.5, giveback_ccy=0.25)

    stale = _trade("T-MANUAL", "EUR_USD", 1000)
    closed = guard.process_open_trades([stale])

    assert closed == ["T-MANUAL"]
    assert guard.snapshot() == {}
    assert broker.closed == []
    out = capsys.readouterr().out
    assert "reconcile_missing_position ticket=T-MANUAL instrument=EUR_USD action=mark_closed" in out

    # Second call should be skipped without attempting to close again.
    closed_second = guard.process_open_trades([stale])
    assert closed_second == []
    assert broker.closed == []
    out_second = capsys.readouterr().out
    assert "attempting_close" not in out_second


def test_correct_side_closeout_long_and_short():
    class SideBroker(DummyBroker):
        def __init__(self):
            super().__init__(
                profits={"EUR_USD": [1.2, 0.6], "GBP_USD": [1.2, 0.6]},
                open_trades=[
                    {"id": "T-LONG", "instrument": "EUR_USD", "currentUnits": 1000},
                    {"id": "T-SHORT", "instrument": "GBP_USD", "currentUnits": -1000},
                ],
            )

        def position_snapshot(self, instrument: str):
            if instrument == "EUR_USD":
                return {"instrument": instrument, "longUnits": "1000", "shortUnits": "0"}
            if instrument == "GBP_USD":
                return {"instrument": instrument, "longUnits": "0", "shortUnits": "-1000"}
            return super().position_snapshot(instrument)

    broker = SideBroker()
    guard = ProfitProtection(broker, arm_ccy=1.0, giveback_ccy=0.5)

    armed = [_trade("T-LONG", "EUR_USD", 1000), _trade("T-SHORT", "GBP_USD", -1000)]
    guard.process_open_trades(armed)

    giveback = [_trade("T-LONG", "EUR_USD", 1000), _trade("T-SHORT", "GBP_USD", -1000)]
    closed = guard.process_open_trades(giveback)

    assert set(closed) == {"T-LONG", "T-SHORT"}
    payloads = {(c["instrument"], frozenset(c["payload"].items())) for c in broker.closed}
    assert ("EUR_USD", frozenset({"longUnits": "ALL"}.items())) in payloads
    assert ("GBP_USD", frozenset({"shortUnits": "ALL"}.items())) in payloads


def test_broker_open_clears_local_closed_marker_and_continues_trailing(capsys):
    broker = DummyBroker(
        profits={"EUR_USD": [1.2, 0.6]},
        open_trades=[{"id": "T-RESYNC", "instrument": "EUR_USD", "currentUnits": 1000}],
    )
    guard = ProfitProtection(broker, arm_ccy=1.0, giveback_ccy=0.5)
    guard._mark_locally_closed("T-RESYNC", "EUR_USD")

    armed_trade = _trade("T-RESYNC", "EUR_USD", 1000)
    guard.process_open_trades([armed_trade])

    drop_trade = _trade("T-RESYNC", "EUR_USD", 1000)
    closed = guard.process_open_trades([drop_trade])

    assert closed == ["T-RESYNC"]
    assert broker.closed == [{"instrument": "EUR_USD", "payload": {"longUnits": "ALL"}}]
    out = capsys.readouterr().out
    assert "clearing_local_closed_marker" in out
    assert "giveback_met ticket=T-RESYNC instrument=EUR_USD" in out


def test_soft_scalp_mode_allows_unarmed_giveback_exit(capsys):
    broker = DummyBroker(profits={"EUR_USD": [0.3, -0.25]})
    guard = ProfitProtection(broker, arm_ccy=1.0, giveback_ccy=0.5, soft_scalp_mode=True)

    first = _trade("T-SOFT", "EUR_USD", 1000, profit=0.3)
    guard.process_open_trades([first])

    second = _trade("T-SOFT", "EUR_USD", 1000, profit=-0.25)
    closed = guard.process_open_trades([second])

    assert closed == ["T-SOFT"]
    out = capsys.readouterr().out
    assert "giveback_met ticket=T-SOFT instrument=EUR_USD" in out
    assert out.count("[TRADE-SUMMARY]") == 1


def test_soft_scalp_mode_default_preserves_behavior(capsys):
    broker = DummyBroker(profits={"EUR_USD": [0.3, -0.25]})
    guard = ProfitProtection(broker, arm_ccy=1.0, giveback_ccy=0.5, soft_scalp_mode=False)

    first = _trade("T-HARD", "EUR_USD", 1000, profit=0.3)
    guard.process_open_trades([first])

    second = _trade("T-HARD", "EUR_USD", 1000, profit=-0.25)
    closed = guard.process_open_trades([second])

    assert closed == []
    out = capsys.readouterr().out
    assert out.count("[TRADE-SUMMARY]") == 0


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
    assert aggressive_broker.closed[0]["instrument"] == "EUR_USD"

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
    assert output.count("[TIME-EXIT]") == 3


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
    assert broker.closed == [{"instrument": "XAU_USD", "payload": {"longUnits": "ALL"}}]
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

    assert any(entry.get("instrument") == "EUR_USD" for entry in broker.closed)


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
