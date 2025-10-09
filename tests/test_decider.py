from datetime import datetime, timezone
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from src.decision_engine import DecisionEngine


@pytest.fixture()
def sample_config() -> Dict:
    return {
        "instruments": ["EUR_USD", "AUD_USD", "XAU_USD"],
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
    signals = {ev.instrument: ev.signal for ev in evaluations}
    assert signals["EUR_USD"] == "BUY"
    assert signals["AUD_USD"] == "SELL"
    assert signals["XAU_USD"] == "HOLD"

    captured = capfd.readouterr()
    output_lines = [line for line in captured.out.splitlines() if line.startswith("[SCAN]")]
    assert any("[SCAN] EUR_USD signal=BUY" in line for line in output_lines)
    assert any("[SCAN] AUD_USD signal=SELL" in line for line in output_lines)
    assert any("[SCAN] XAU_USD signal=HOLD" in line for line in output_lines)


def test_skips_inactive_markets(capfd, sample_config):
    def fetcher(instrument: str, **kwargs):
        return [{"o": 1.0, "h": 1.0, "l": 1.0, "c": None}]

    engine = DecisionEngine(sample_config, candle_fetcher=fetcher, now_fn=lambda: datetime.now(timezone.utc))
    evaluations = engine.evaluate_all()

    assert all(not ev.market_active for ev in evaluations)
    assert all(ev.signal == "HOLD" for ev in evaluations)

    captured = capfd.readouterr()
    output_lines = [line for line in captured.out.splitlines() if line.startswith("[SCAN]")]
    assert len(output_lines) == len(sample_config["instruments"])
    assert all("signal=HOLD" in line for line in output_lines)
    assert all("rsi=n/a" in line for line in output_lines)
    assert all("atr=n/a" in line for line in output_lines)
