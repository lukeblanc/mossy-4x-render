import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.main as main  # noqa: E402
from src.decision_engine import DecisionEngine, Evaluation  # noqa: E402
from src.projector import project_market  # noqa: E402


@pytest.fixture()
def sample_candles() -> List[Dict[str, float]]:
    return [
        {"o": 1.10, "h": 1.105, "l": 1.095, "c": 1.10},
        {"o": 1.11, "h": 1.115, "l": 1.105, "c": 1.11},
        {"o": 1.12, "h": 1.125, "l": 1.115, "c": 1.12},
        {"o": 1.13, "h": 1.135, "l": 1.125, "c": 1.13},
        {"o": 1.14, "h": 1.145, "l": 1.135, "c": 1.14},
    ]


def test_project_market_generates_bullish_projection(monkeypatch, sample_candles):
    monkeypatch.setenv("PROJECTOR_HORIZON", "4")
    monkeypatch.setenv("PROJECTOR_ATR_LENGTH", "3")
    indicators = {
        "ema_fast": 1.20,
        "ema_slow": 1.00,
        "rsi": 70.0,
        "atr": 0.015,
        "close": 1.20,
    }
    projection = project_market(
        "EUR_USD",
        sample_candles,
        indicators,
        datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
    )

    assert projection["bias"] == "BULL"
    assert pytest.approx(projection["bias_score"], rel=1e-3) == 0.92
    assert projection["range"]["horizon"] == 4
    assert pytest.approx(projection["range"]["low"], rel=1e-3) == 1.17
    assert pytest.approx(projection["range"]["high"], rel=1e-3) == 1.23
    assert projection["volatility"] == "NORMAL"
    assert pytest.approx(projection["confidence"], rel=1e-3) == 78.4


def test_project_market_handles_neutral_high_vol(monkeypatch, sample_candles):
    monkeypatch.setenv("PROJECTOR_HORIZON", "2")
    monkeypatch.setenv("PROJECTOR_ATR_LENGTH", "2")
    indicators = {
        "ema_fast": 1.00,
        "ema_slow": 1.00,
        "rsi": 50.0,
        "atr": 0.05,
        "close": 1.00,
    }
    projection = project_market(
        "GBP_USD",
        sample_candles,
        indicators,
        datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
    )

    assert projection["bias"] == "NEUTRAL"
    assert projection["volatility"] == "HIGH"
    assert pytest.approx(projection["range"]["low"], rel=1e-3) == 1.0 - (0.05 * 2**0.5)
    assert pytest.approx(projection["range"]["high"], rel=1e-3) == 1.0 + (0.05 * 2**0.5)
    assert projection["confidence"] == 30.0


def test_projector_called_when_enabled(monkeypatch, capfd):
    monkeypatch.setenv("ENABLE_PROJECTOR", "true")

    class DummyRisk:
        risk_per_trade_pct = 0.001
        demo_mode = False

        def enforce_equity_floor(self, *args, **kwargs):
            pass

        def should_open(self, *args, **kwargs):
            return True, "ok"

        def sl_distance_from_atr(self, atr, instrument=None):
            return atr * 1.5 if atr else 0.0

        def tp_distance_from_atr(self, atr, instrument=None):
            return atr * 3.0 if atr else 0.0

        def register_entry(self, *args, **kwargs):
            pass

        def register_exit(self, *args, **kwargs):
            pass

    class DummyBroker:
        def __init__(self) -> None:
            self.calls: List[Dict[str, object]] = []

        def place_order(self, instrument, signal, units, *, sl_distance=None, tp_distance=None, entry_price=None):
            self.calls.append(
                {
                    "instrument": instrument,
                    "signal": signal,
                    "units": units,
                    "sl_distance": sl_distance,
                    "tp_distance": tp_distance,
                    "entry_price": entry_price,
                }
            )
            return {"status": "SENT"}

        def account_equity(self) -> float:
            return 10_000.0

        def current_spread(self, instrument: str) -> float:
            return 0.5

        def close_all_positions(self) -> None:
            pass

        def list_open_trades(self):
            return []

    class DummyEngine:
        def __init__(self) -> None:
            self.marked: List[str] = []

        def evaluate_all(self) -> List[Evaluation]:
            return [
                Evaluation(
                    instrument="EUR_USD",
                    signal="BUY",
                    diagnostics={
                        "ema_fast": 1.1,
                        "ema_slow": 1.0,
                        "rsi": 55.0,
                        "atr": 0.01,
                        "close": 1.1234,
                        "atr_baseline_50": 0.01,
                        "ema_trend_fast": 1.2,
                        "ema_trend_slow": 1.1,
                    },
                    reason="bullish",
                    market_active=True,
                    candles=[{"o": 1.0, "h": 1.1, "l": 0.9, "c": 1.0}],
                )
            ]

        def mark_trade(self, instrument: str) -> None:
            self.marked.append(instrument)

    calls: Dict[str, object] = {"count": 0}

    def fake_project(pair, candles, indicators, now_utc):
        calls["count"] += 1
        calls["pair"] = pair
        calls["candles"] = list(candles)
        calls["indicators"] = indicators
        calls["ts"] = now_utc
        return {
            "pair": pair,
            "timestamp": now_utc,
            "bias": "BULL",
            "bias_score": 0.42,
            "range": {"low": 1.1762, "high": 1.1798},
            "volatility": "NORMAL",
            "confidence": 68,
        }

    dummy_engine = DummyEngine()
    dummy_broker = DummyBroker()
    dummy_risk = DummyRisk()
    monkeypatch.setattr(main, "engine", dummy_engine)
    monkeypatch.setattr(main, "broker", dummy_broker)
    monkeypatch.setattr(main, "risk", dummy_risk)
    monkeypatch.setattr(main, "_open_trades_state", lambda: [])
    monkeypatch.setattr(main, "profit_guard", type("PG", (), {"process_open_trades": lambda self, trades: []})())
    monkeypatch.setattr(main.session_filter, "is_entry_session", lambda *args, **kwargs: True)
    monkeypatch.setattr(main.position_sizer, "units_for_risk", lambda *args, **kwargs: 100)
    monkeypatch.setattr(main, "project_market", fake_project)

    asyncio.run(main.decision_cycle())

    captured = capfd.readouterr().out
    assert "[PROJECTOR] EUR_USD ts=" in captured
    assert "bias=BULL score=0.42 conf=68 vol=NORMAL range=1.1762..1.1798" in captured
    assert calls["count"] == 1
    assert calls["pair"] == "EUR_USD"
    assert dummy_broker.calls  # trade still placed


def test_projector_not_called_when_disabled(monkeypatch, capfd):
    monkeypatch.setenv("ENABLE_PROJECTOR", "false")

    class DummyRisk:
        risk_per_trade_pct = 0.001
        demo_mode = False

        def enforce_equity_floor(self, *args, **kwargs):
            pass

        def should_open(self, *args, **kwargs):
            return True, "ok"

        def sl_distance_from_atr(self, atr, instrument=None):
            return atr * 1.5 if atr else 0.0

        def tp_distance_from_atr(self, atr, instrument=None):
            return atr * 3.0 if atr else 0.0

        def register_entry(self, *args, **kwargs):
            pass

        def register_exit(self, *args, **kwargs):
            pass

    class DummyBroker:
        def __init__(self) -> None:
            self.calls: List[Dict[str, object]] = []

        def place_order(self, instrument, signal, units, *, sl_distance=None, tp_distance=None, entry_price=None):
            self.calls.append(
                {
                    "instrument": instrument,
                    "signal": signal,
                    "units": units,
                    "sl_distance": sl_distance,
                    "tp_distance": tp_distance,
                    "entry_price": entry_price,
                }
            )
            return {"status": "SENT"}

        def account_equity(self) -> float:
            return 10_000.0

        def current_spread(self, instrument: str) -> float:
            return 0.5

        def close_all_positions(self) -> None:
            pass

        def list_open_trades(self):
            return []

    class DummyEngine:
        def __init__(self) -> None:
            self.marked: List[str] = []

        def evaluate_all(self) -> List[Evaluation]:
            return [
                Evaluation(
                    instrument="EUR_USD",
                    signal="BUY",
                    diagnostics={
                        "ema_fast": 1.1,
                        "ema_slow": 1.0,
                        "rsi": 55.0,
                        "atr": 0.01,
                        "close": 1.1234,
                        "atr_baseline_50": 0.01,
                        "ema_trend_fast": 1.2,
                        "ema_trend_slow": 1.1,
                    },
                    reason="bullish",
                    market_active=True,
                    candles=[{"o": 1.0, "h": 1.1, "l": 0.9, "c": 1.0}],
                )
            ]

        def mark_trade(self, instrument: str) -> None:
            self.marked.append(instrument)

    calls: Dict[str, object] = {"count": 0}

    def fake_project(*args, **kwargs):
        calls["count"] += 1
        return {
            "pair": "EUR_USD",
            "timestamp": datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
            "bias": "BULL",
            "bias_score": 0.42,
            "range": {"low": 1.1762, "high": 1.1798},
            "volatility": "NORMAL",
            "confidence": 68,
        }

    dummy_engine = DummyEngine()
    dummy_broker = DummyBroker()
    dummy_risk = DummyRisk()
    monkeypatch.setattr(main, "engine", dummy_engine)
    monkeypatch.setattr(main, "broker", dummy_broker)
    monkeypatch.setattr(main, "risk", dummy_risk)
    monkeypatch.setattr(main, "_open_trades_state", lambda: [])
    monkeypatch.setattr(main, "profit_guard", type("PG", (), {"process_open_trades": lambda self, trades: []})())
    monkeypatch.setattr(main.session_filter, "is_entry_session", lambda *args, **kwargs: True)
    monkeypatch.setattr(main.position_sizer, "units_for_risk", lambda *args, **kwargs: 100)
    monkeypatch.setattr(main, "project_market", fake_project)

    asyncio.run(main.decision_cycle())

    captured = capfd.readouterr().out
    assert "[PROJECTOR]" not in captured
    assert calls["count"] == 0
    assert dummy_broker.calls  # trade still placed
