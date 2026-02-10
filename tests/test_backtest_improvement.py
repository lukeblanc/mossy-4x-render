from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtest_engine import run_backtest


def _synthetic_candles(n: int = 1400, seed: int = 7) -> list[dict]:
    rng = random.Random(seed)
    price = 1.1000
    candles = []
    for i in range(n):
        regime = i % 200
        if regime < 40:
            drift = 0.00012  # short trend window
            vol = 0.00012
        else:
            drift = 0.0  # dominant choppy regime
            vol = 0.00045
        change = drift + rng.uniform(-vol, vol)
        close = max(0.5, price + change)
        high = max(price, close) + abs(rng.uniform(0.0, vol))
        low = min(price, close) - abs(rng.uniform(0.0, vol))
        candles.append({"o": price, "h": high, "l": low, "c": close})
        price = close
    return candles


def test_enhanced_strategy_improves_metrics_in_chop_heavy_regime():
    candles = _synthetic_candles()
    baseline = run_backtest(candles, enhanced=False)
    enhanced = run_backtest(candles, enhanced=True)

    assert enhanced.net_pnl > baseline.net_pnl
    assert enhanced.profit_factor > baseline.profit_factor
    assert enhanced.max_drawdown < baseline.max_drawdown
