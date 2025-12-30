from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Dict, List

import pytest

from src import main
from src.decision_engine import Evaluation


class Monday(datetime):
    @classmethod
    def now(cls, tz=None):
        return datetime(2024, 6, 3, 13, 0, tzinfo=timezone.utc)


def _reset_clock(original):
    main.datetime = original  # type: ignore[assignment]


def _allow_session_decision():
    return main.session_filter.SessionDecision(True, True, None, "STRICT")


def _set_heartbeat(monkeypatch):
    original_datetime = main.datetime
    monkeypatch.setattr(main, "datetime", Monday)
    from app.health import watchdog

    before = Monday.now(timezone.utc) - timedelta(hours=1)
    original_ts = watchdog.last_decision_ts
    watchdog.last_decision_ts = before
    return original_datetime, original_ts, before, watchdog


def test_macd_veto_blocks_trade(monkeypatch, capsys):
    class DummyRisk:
        risk_per_trade_pct = 0.001

        def should_open(self, *args, **kwargs):
            return True, "ok"

        def sl_distance_from_atr(self, atr, instrument=None):
            return 0.5

        def tp_distance_from_atr(self, atr, instrument=None):
            return 1.0

        def register_entry(self, now_utc, instrument: str):
            raise AssertionError("entry should be vetoed")

    class DummyEngine:
        def __init__(self) -> None:
            self.marked: List[str] = []

        def evaluate_all(self) -> List[Evaluation]:
            return [
                Evaluation(
                    instrument="EUR_USD",
                    signal="BUY",
                    diagnostics={
                        "atr": 0.01,
                        "atr_baseline_50": 0.01,
                        "rsi": 55.0,
                        "close": 1.2345,
                        "ema_trend_fast": 1.25,
                        "ema_trend_slow": 1.2,
                        "macd_line": 0.0,
                        "macd_signal": 1.0,
                        "macd_histogram": -1.0,
                    },
                    reason="bullish",
                    market_active=True,
                )
            ]

        def mark_trade(self, instrument: str) -> None:
            self.marked.append(instrument)

    class DummyBroker:
        def __init__(self) -> None:
            self.calls: List[Dict[str, object]] = []

        def place_order(
            self,
            instrument: str,
            signal: str,
            units: int,
            *args,
            **kwargs,
        ) -> Dict[str, str]:
            self.calls.append({"instrument": instrument, "signal": signal, "units": units})
            return {"status": "SENT"}

        def account_equity(self) -> float:
            return 10_000.0

        def current_spread(self, instrument: str) -> float:
            return 0.5

        def close_all_positions(self) -> None:
            pass

    dummy_engine = DummyEngine()
    dummy_broker = DummyBroker()
    dummy_risk = DummyRisk()
    monkeypatch.setitem(main.config, "use_macd_confirmation", True)
    monkeypatch.setattr(main, "engine", dummy_engine)
    monkeypatch.setattr(main, "broker", dummy_broker)
    monkeypatch.setattr(main, "risk", dummy_risk)
    monkeypatch.setattr(main, "profit_guard", type("PG", (), {"process_open_trades": lambda self, trades: []})())
    monkeypatch.setattr(main, "_open_trades_state", lambda: [])
    monkeypatch.setattr(main.session_filter, "session_decision", lambda *args, **kwargs: _allow_session_decision())
    monkeypatch.setattr(
        main.position_sizer,
        "units_for_risk",
        lambda equity, entry_price, stop_distance, risk_pct: 100,
    )
    original_datetime, original_ts, before, watchdog = _set_heartbeat(monkeypatch)

    asyncio.run(main.decision_cycle())

    captured = capsys.readouterr().out
    assert "[FILTER] MACD veto EUR_USD" in captured
    assert dummy_engine.marked == []
    assert dummy_broker.calls == []
    watchdog.last_decision_ts = original_ts
    _reset_clock(original_datetime)


def test_macd_confirmation_allows_trade(monkeypatch):
    class DummyRisk:
        risk_per_trade_pct = 0.001

        def should_open(self, *args, **kwargs):
            return True, "ok"

        def sl_distance_from_atr(self, atr, instrument=None):
            return 0.5

        def tp_distance_from_atr(self, atr, instrument=None):
            return 1.0

        def register_entry(self, now_utc, instrument: str):
            pass

    class DummyEngine:
        def __init__(self) -> None:
            self.marked: List[str] = []

        def evaluate_all(self) -> List[Evaluation]:
            return [
                Evaluation(
                    instrument="EUR_USD",
                    signal="BUY",
                    diagnostics={
                        "atr": 0.01,
                        "atr_baseline_50": 0.01,
                        "rsi": 60.0,
                        "close": 1.2345,
                        "ema_trend_fast": 1.25,
                        "ema_trend_slow": 1.2,
                        "macd_line": 1.5,
                        "macd_signal": 1.0,
                        "macd_histogram": 0.5,
                    },
                    reason="bullish",
                    market_active=True,
                    candles=[
                        {"o": 1.0, "h": 1.05, "l": 0.99, "c": 1.01},
                        {"o": 1.01, "h": 1.07, "l": 1.0, "c": 1.04},
                        {"o": 1.04, "h": 1.08, "l": 1.02, "c": 1.06},
                    ],
                )
            ]

        def mark_trade(self, instrument: str) -> None:
            self.marked.append(instrument)

    class DummyBroker:
        def __init__(self) -> None:
            self.calls: List[Dict[str, object]] = []

        def place_order(
            self,
            instrument: str,
            signal: str,
            units: int,
            *args,
            **kwargs,
        ) -> Dict[str, str]:
            self.calls.append({"instrument": instrument, "signal": signal, "units": units})
            return {"status": "SENT"}

        def account_equity(self) -> float:
            return 10_000.0

        def current_spread(self, instrument: str) -> float:
            return 0.5

        def close_all_positions(self) -> None:
            pass

    dummy_engine = DummyEngine()
    dummy_broker = DummyBroker()
    dummy_risk = DummyRisk()
    monkeypatch.setitem(main.config, "use_macd_confirmation", True)
    monkeypatch.setattr(main, "engine", dummy_engine)
    monkeypatch.setattr(main, "broker", dummy_broker)
    monkeypatch.setattr(main, "risk", dummy_risk)
    monkeypatch.setattr(main, "profit_guard", type("PG", (), {"process_open_trades": lambda self, trades: []})())
    monkeypatch.setattr(main, "_open_trades_state", lambda: [])
    monkeypatch.setattr(main.session_filter, "session_decision", lambda *args, **kwargs: _allow_session_decision())
    monkeypatch.setattr(
        main.position_sizer,
        "units_for_risk",
        lambda equity, entry_price, stop_distance, risk_pct: 100,
    )
    original_datetime, original_ts, before, watchdog = _set_heartbeat(monkeypatch)

    asyncio.run(main.decision_cycle())

    assert dummy_engine.marked == ["EUR_USD"]
    assert dummy_broker.calls == [{"instrument": "EUR_USD", "signal": "BUY", "units": 100}]
    watchdog.last_decision_ts = original_ts
    _reset_clock(original_datetime)


def test_trailing_flow_unchanged_with_macd(monkeypatch):
    class DummyRisk:
        risk_per_trade_pct = 0.001

        def should_open(self, *args, **kwargs):
            return False, "no-new-trades"

        def sl_distance_from_atr(self, atr, instrument=None):
            return 0.5

        def tp_distance_from_atr(self, atr, instrument=None):
            return 1.0

        def register_entry(self, now_utc, instrument: str):
            raise AssertionError("no entries expected")

    class DummyEngine:
        def evaluate_all(self) -> List[Evaluation]:
            return []

        def mark_trade(self, instrument: str) -> None:
            raise AssertionError("no marks expected")

    class DummyGuard:
        def __init__(self) -> None:
            self.calls: List[List[Dict[str, object]]] = []

        def process_open_trades(self, trades):
            self.calls.append(list(trades))
            return ["EUR_USD"]

    dummy_engine = DummyEngine()
    dummy_risk = DummyRisk()
    dummy_guard = DummyGuard()
    monkeypatch.setitem(main.config, "use_macd_confirmation", True)
    monkeypatch.setattr(main, "engine", dummy_engine)
    monkeypatch.setattr(main, "broker", type("B", (), {"account_equity": lambda self: 10_000.0, "close_all_positions": lambda self: None})())
    monkeypatch.setattr(main, "profit_guard", dummy_guard)
    monkeypatch.setattr(main, "risk", dummy_risk)
    monkeypatch.setattr(main, "_open_trades_state", lambda: [{"instrument": "EUR_USD", "id": "T1"}])
    monkeypatch.setattr(main.session_filter, "session_decision", lambda *args, **kwargs: _allow_session_decision())

    original_datetime, original_ts, before, watchdog = _set_heartbeat(monkeypatch)

    asyncio.run(main.decision_cycle())

    assert dummy_guard.calls == [[{"instrument": "EUR_USD", "id": "T1"}]]
    watchdog.last_decision_ts = original_ts
    _reset_clock(original_datetime)


def test_macd_does_not_create_new_signals(monkeypatch):
    class DummyRisk:
        risk_per_trade_pct = 0.001

        def should_open(self, *args, **kwargs):
            return True, "ok"

        def sl_distance_from_atr(self, atr, instrument=None):
            return 0.5

        def tp_distance_from_atr(self, atr, instrument=None):
            return 1.0

        def register_entry(self, now_utc, instrument: str):
            raise AssertionError("no entries expected")

    class DummyEngine:
        def evaluate_all(self) -> List[Evaluation]:
            return [
                Evaluation(
                    instrument="EUR_USD",
                    signal="HOLD",
                    diagnostics={
                        "atr": 0.01,
                        "atr_baseline_50": 0.01,
                        "rsi": 55.0,
                        "close": 1.2345,
                        "ema_trend_fast": 1.25,
                        "ema_trend_slow": 1.2,
                        "macd_line": 2.0,
                        "macd_signal": 1.0,
                        "macd_histogram": 1.0,
                    },
                    reason="neutral",
                    market_active=True,
                )
            ]

        def mark_trade(self, instrument: str) -> None:
            raise AssertionError("no marks expected")

    class DummyBroker:
        def __init__(self) -> None:
            self.calls: List[Dict[str, object]] = []

        def place_order(
            self,
            instrument: str,
            signal: str,
            units: int,
            *args,
            **kwargs,
        ) -> Dict[str, str]:
            self.calls.append({"instrument": instrument, "signal": signal, "units": units})
            return {"status": "SENT"}

        def account_equity(self) -> float:
            return 10_000.0

        def current_spread(self, instrument: str) -> float:
            return 0.5

        def close_all_positions(self) -> None:
            pass

    dummy_engine = DummyEngine()
    dummy_broker = DummyBroker()
    dummy_risk = DummyRisk()
    monkeypatch.setitem(main.config, "use_macd_confirmation", True)
    monkeypatch.setattr(main, "engine", dummy_engine)
    monkeypatch.setattr(main, "broker", dummy_broker)
    monkeypatch.setattr(main, "risk", dummy_risk)
    monkeypatch.setattr(main, "profit_guard", type("PG", (), {"process_open_trades": lambda self, trades: []})())
    monkeypatch.setattr(main, "_open_trades_state", lambda: [])
    monkeypatch.setattr(main.session_filter, "session_decision", lambda *args, **kwargs: _allow_session_decision())
    monkeypatch.setattr(
        main.position_sizer,
        "units_for_risk",
        lambda equity, entry_price, stop_distance, risk_pct: 100,
    )
    original_datetime, original_ts, before, watchdog = _set_heartbeat(monkeypatch)

    asyncio.run(main.decision_cycle())

    assert dummy_broker.calls == []
    watchdog.last_decision_ts = original_ts
    _reset_clock(original_datetime)


def test_macd_confirms_fx_and_xau(monkeypatch):
    class DummyRisk:
        risk_per_trade_pct = 0.001

        def __init__(self) -> None:
            self.entries: List[str] = []

        def should_open(self, *args, **kwargs):
            return True, "ok"

        def sl_distance_from_atr(self, atr, instrument=None):
            return 0.5

        def tp_distance_from_atr(self, atr, instrument=None):
            return 1.0

        def register_entry(self, now_utc, instrument: str):
            self.entries.append(instrument)

    class DummyEngine:
        def __init__(self) -> None:
            self.marked: List[str] = []

        def evaluate_all(self) -> List[Evaluation]:
            return [
                Evaluation(
                    instrument="EUR_USD",
                    signal="BUY",
                    diagnostics={
                        "atr": 0.01,
                        "atr_baseline_50": 0.01,
                        "rsi": 60.0,
                        "close": 1.2345,
                        "ema_trend_fast": 1.25,
                        "ema_trend_slow": 1.2,
                        "macd_line": 1.0,
                        "macd_signal": 0.5,
                        "macd_histogram": 0.5,
                    },
                    reason="bullish",
                    market_active=True,
                    candles=[
                        {"o": 1.0, "h": 1.05, "l": 0.99, "c": 1.01},
                        {"o": 1.01, "h": 1.07, "l": 1.0, "c": 1.04},
                        {"o": 1.04, "h": 1.08, "l": 1.02, "c": 1.06},
                    ],
                ),
                Evaluation(
                    instrument="XAU_USD",
                    signal="SELL",
                    diagnostics={
                        "atr": 1.2,
                        "atr_baseline_50": 1.0,
                        "rsi": 45.0,
                        "close": 1898.0,
                        "ema_trend_fast": 1890.0,
                        "ema_trend_slow": 1900.0,
                        "macd_line": -0.5,
                        "macd_signal": 0.0,
                        "macd_histogram": -0.5,
                    },
                    reason="bearish",
                    market_active=True,
                    candles=[
                        {"o": 1901.0, "h": 1905.0, "l": 1899.0, "c": 1902.0},
                        {"o": 1902.0, "h": 1903.0, "l": 1900.0, "c": 1901.5},
                        {"o": 1901.5, "h": 1902.0, "l": 1900.5, "c": 1901.0},
                    ],
                ),
            ]

        def mark_trade(self, instrument: str) -> None:
            self.marked.append(instrument)

    class DummyBroker:
        def __init__(self) -> None:
            self.calls: List[Dict[str, object]] = []

        def place_order(
            self,
            instrument: str,
            signal: str,
            units: int,
            *args,
            **kwargs,
        ) -> Dict[str, str]:
            self.calls.append({"instrument": instrument, "signal": signal, "units": units})
            return {"status": "SENT"}

        def account_equity(self) -> float:
            return 10_000.0

        def current_spread(self, instrument: str) -> float:
            return 0.5

        def close_all_positions(self) -> None:
            pass

    dummy_engine = DummyEngine()
    dummy_broker = DummyBroker()
    dummy_risk = DummyRisk()
    monkeypatch.setitem(main.config, "use_macd_confirmation", True)
    monkeypatch.setattr(main, "engine", dummy_engine)
    monkeypatch.setattr(main, "broker", dummy_broker)
    monkeypatch.setattr(main, "risk", dummy_risk)
    monkeypatch.setattr(main, "profit_guard", type("PG", (), {"process_open_trades": lambda self, trades: []})())
    monkeypatch.setattr(main, "_open_trades_state", lambda: [])
    monkeypatch.setattr(main.session_filter, "session_decision", lambda *args, **kwargs: _allow_session_decision())
    monkeypatch.setattr(
        main.position_sizer,
        "units_for_risk",
        lambda equity, entry_price, stop_distance, risk_pct: 50,
    )
    original_datetime, original_ts, before, watchdog = _set_heartbeat(monkeypatch)

    asyncio.run(main.decision_cycle())

    assert {"instrument": "EUR_USD", "signal": "BUY", "units": 50} in dummy_broker.calls
    assert {"instrument": "XAU_USD", "signal": "SELL", "units": 50} in dummy_broker.calls
    assert dummy_risk.entries == ["EUR_USD", "XAU_USD"]
    watchdog.last_decision_ts = original_ts
    _reset_clock(original_datetime)


def test_macd_histogram_rising_allows_negative_hist(monkeypatch):
    monkeypatch.setitem(main.config, "use_macd_confirmation", True)
    macd_result = main._macd_confirms(
        "BUY",
        {
            "macd_line": 0.5,
            "macd_signal": 0.1,
            "macd_histogram": -0.01,
            "macd_histogram_prev": -0.05,
        },
    )
    assert macd_result[0] is True
