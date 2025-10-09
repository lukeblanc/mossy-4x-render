import asyncio
from datetime import datetime, timedelta, timezone
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from app.health import watchdog

from src.decision_engine import DecisionEngine
from src.decision_engine import Evaluation
from src import main


@pytest.fixture()
def sample_config() -> Dict:
    return {
        "instruments": [
            "EUR_USD",
            "AUD_USD",
            "GBP_USD",
            "USD_JPY",
            "XAU_USD",
        ],
        "cooldown_minutes": 0,
        "risk_per_trade": 0.02,
        "account_balance": 10000,
        "candles_to_fetch": 5,
        "timeframe": "M1",
        "ema_fast": 2,
        "ema_slow": 3,
        "rsi_length": 2,
        "rsi_buy": 60,
        "rsi_sell": 40,
        "atr_length": 2,
        "min_atr": 0.0001,
    }


def test_scans_all_instruments(capfd, sample_config):
    prices: Dict[str, List[Dict[str, float]]] = {
        "EUR_USD": [
            {"o": 1.0, "h": 1.1, "l": 0.9, "c": 1.0},
            {"o": 1.0, "h": 1.2, "l": 1.0, "c": 1.2},
            {"o": 1.2, "h": 1.3, "l": 1.1, "c": 1.3},
            {"o": 1.3, "h": 1.4, "l": 1.2, "c": 1.4},
        ],
        "AUD_USD": [
            {"o": 0.75, "h": 0.76, "l": 0.74, "c": 0.75},
            {"o": 0.75, "h": 0.75, "l": 0.73, "c": 0.74},
            {"o": 0.74, "h": 0.74, "l": 0.72, "c": 0.73},
            {"o": 0.73, "h": 0.73, "l": 0.71, "c": 0.72},
        ],
        "GBP_USD": [
            {"o": 1.3, "h": 1.31, "l": 1.29, "c": 1.3},
            {"o": 1.3, "h": 1.32, "l": 1.3, "c": 1.31},
            {"o": 1.31, "h": 1.33, "l": 1.31, "c": 1.33},
            {"o": 1.33, "h": 1.34, "l": 1.32, "c": 1.34},
        ],
        "USD_JPY": [
            {"o": 110.0, "h": 110.5, "l": 109.5, "c": 110.2},
            {"o": 110.2, "h": 110.4, "l": 109.8, "c": 110.0},
            {"o": 110.0, "h": 110.1, "l": 109.7, "c": 109.9},
            {"o": 109.9, "h": 110.0, "l": 109.5, "c": 109.6},
        ],
        "XAU_USD": [
            {"o": 1950.0, "h": 1951.0, "l": 1949.0, "c": 1950.5},
            {"o": 1950.5, "h": 1951.5, "l": 1949.5, "c": 1950.5},
        ],
    }

    def fetcher(instrument: str, **kwargs):
        return prices[instrument]

    engine = DecisionEngine(sample_config, candle_fetcher=fetcher, now_fn=lambda: datetime.now(timezone.utc))
    evaluations = engine.evaluate_all()

    assert [ev.instrument for ev in evaluations] == sample_config["instruments"]

    captured = capfd.readouterr()
    output_lines = captured.out.splitlines()
    scan_lines = [line for line in output_lines if line.startswith("[SCAN]")]
    decision_lines = [line for line in output_lines if line.startswith("[DECISION]")]

    assert len(scan_lines) == len(sample_config["instruments"])
    assert len(decision_lines) == len(sample_config["instruments"])

    for instrument in sample_config["instruments"]:
        assert any(f"[SCAN] Evaluating {instrument}" in line for line in scan_lines)
        assert any(f"[DECISION] {instrument} signal=" in line for line in decision_lines)


def test_skips_inactive_markets(capfd, sample_config):
    def fetcher(instrument: str, **kwargs):
        return [{"o": 1.0, "h": 1.0, "l": 1.0, "c": None}]

    engine = DecisionEngine(sample_config, candle_fetcher=fetcher, now_fn=lambda: datetime.now(timezone.utc))
    evaluations = engine.evaluate_all()

    assert all(not ev.market_active for ev in evaluations)
    assert all(ev.signal == "HOLD" for ev in evaluations)

    captured = capfd.readouterr()
    output_lines = captured.out.splitlines()
    scan_lines = [line for line in output_lines if line.startswith("[SCAN]")]
    decision_lines = [line for line in output_lines if line.startswith("[DECISION]")]

    assert len(scan_lines) == len(sample_config["instruments"])
    assert len(decision_lines) == len(sample_config["instruments"])

    assert all("rsi=n/a" in line for line in scan_lines)
    assert all("atr=n/a" in line for line in scan_lines)
    assert all("signal=HOLD" in line for line in decision_lines)
    assert all("reason=inactive-market" in line for line in decision_lines)


def test_decision_cycle_updates_watchdog_on_success(monkeypatch):
    class DummyEngine:
        def __init__(self) -> None:
            self.marked: List[str] = []

        def evaluate_all(self) -> List[Evaluation]:
            return [
                Evaluation(
                    instrument="EUR_USD",
                    signal="BUY",
                    diagnostics={},
                    reason="trend",
                    market_active=True,
                )
            ]

        def position_size(self, instrument: str, diagnostics: Dict) -> int:
            return 1

        def mark_trade(self, instrument: str) -> None:
            self.marked.append(instrument)

    class DummyBroker:
        def __init__(self) -> None:
            self.calls: List[Dict[str, str]] = []

        def place_order(self, instrument: str, signal: str, units: int) -> Dict[str, str]:
            self.calls.append({"instrument": instrument, "signal": signal, "units": units})
            return {"status": "SENT"}

    dummy_engine = DummyEngine()
    dummy_broker = DummyBroker()
    monkeypatch.setattr(main, "engine", dummy_engine)
    monkeypatch.setattr(main, "broker", dummy_broker)
    monkeypatch.setattr(main, "_open_trades_state", lambda: [])

    before = datetime.now(timezone.utc) - timedelta(hours=1)
    original_ts = watchdog.last_decision_ts
    watchdog.last_decision_ts = before

    asyncio.run(main.decision_cycle())

    try:
        assert dummy_engine.marked == ["EUR_USD"]
        assert dummy_broker.calls == [{"instrument": "EUR_USD", "signal": "BUY", "units": 1}]
        assert watchdog.last_decision_ts > before
    finally:
        watchdog.last_decision_ts = original_ts


def test_decision_cycle_updates_watchdog_on_error(monkeypatch):
    class FailingEngine:
        def evaluate_all(self) -> List[Evaluation]:
            raise RuntimeError("boom")

    events: Dict[str, bool] = {"error": False}

    def record_error() -> None:
        events["error"] = True

    failing_engine = FailingEngine()
    monkeypatch.setattr(main, "engine", failing_engine)
    monkeypatch.setattr(main.watchdog, "record_error", record_error)

    before = datetime.now(timezone.utc) - timedelta(hours=1)
    original_ts = watchdog.last_decision_ts
    watchdog.last_decision_ts = before

    asyncio.run(main.decision_cycle())

    try:
        assert events["error"] is True
        assert watchdog.last_decision_ts > before
    finally:
        watchdog.last_decision_ts = original_ts
