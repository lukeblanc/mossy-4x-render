import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.decision_engine import DecisionEngine  # noqa: E402
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


def test_decision_engine_logs_projector(monkeypatch, capfd):
    monkeypatch.setenv("ENABLE_PROJECTOR", "true")

    prices: Dict[str, List[Dict[str, float]]] = {
        "EUR_USD": [
            {"o": 1.0, "h": 1.1, "l": 0.9, "c": 1.0},
            {"o": 1.0, "h": 1.1, "l": 0.9, "c": 1.1},
        ]
    }

    def fetcher(instrument: str, **kwargs):
        return prices[instrument]

    calls: Dict[str, object] = {}

    def fake_project(pair, candles, indicators, now_utc):
        calls["pair"] = pair
        calls["candles"] = list(candles)
        calls["indicators"] = indicators
        calls["ts"] = now_utc
        return {
            "pair": pair,
            "timestamp": now_utc,
            "bias": "BULL",
            "bias_score": 0.42,
            "range": {"low": 1.0000, "high": 1.1000},
            "volatility": "NORMAL",
            "confidence": 68,
        }

    monkeypatch.setattr("src.decision_engine.project_market", fake_project)

    engine = DecisionEngine(
        {
            "instruments": ["EUR_USD"],
            "candles_to_fetch": 2,
            "timeframe": "M1",
            "ema_fast": 2,
            "ema_slow": 3,
            "rsi_length": 2,
            "atr_length": 2,
            "min_atr": 0.0001,
        },
        candle_fetcher=fetcher,
        now_fn=lambda: datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
    )

    engine.evaluate_all()

    captured = capfd.readouterr().out
    assert "[PROJECTOR] EUR_USD ts=00:00 bias=BULL score=0.42 conf=68 vol=NORMAL range=1.0000..1.1000" in captured
    assert calls["pair"] == "EUR_USD"
    assert calls["candles"]
    assert calls["indicators"]["atr"] is not None
