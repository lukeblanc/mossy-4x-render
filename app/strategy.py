from __future__ import annotations

import httpx
from datetime import datetime, timezone
from typing import List, Dict, Tuple

from app.config import settings

# Signal type alias
Signal = str


def _fetch_candles(count: int = 200) -> List[Dict]:
    """
    Fetch recent candles for the configured instrument from OANDA.
    Returns a list of dicts with 'o','h','l','c' as floats.
    Uses the practice API when in demo mode. On failure returns an empty list.
    """
    base_url = "https://api-fxpractice.oanda.com/v3"
    instrument = settings.INSTRUMENT
    granularity = settings.STRAT_TIMEFRAME or "M1"
    params = {"count": str(count), "granularity": granularity, "price": "M"}
    headers: Dict[str, str] = {}
    if settings.OANDA_API_KEY:
        headers["Authorization"] = f"Bearer {settings.OANDA_API_KEY}"
    try:
        resp = httpx.get(
            f"{base_url}/instruments/{instrument}/candles",
            params=params,
            headers=headers,
            timeout=10.0,
        )
        resp.raise_for_status()
        data = resp.json()
        candles: List[Dict] = []
        for c in data.get("candles", []):
            mids = c.get("mid", {})
            candles.append(
                {
                    "o": float(mids.get("o")),
                    "h": float(mids.get("h")),
                    "l": float(mids.get("l")),
                    "c": float(mids.get("c")),
                }
            )
        return candles
    except Exception as e:
        # Log and return empty list if fetching fails
        print(f"[ERROR] failed to fetch candles: {e}")
        return []


def _ema(values: List[float], length: int) -> List[float]:
    """
    Compute exponential moving average for a series of values.
    Returns the full EMA series of the same length.
    """
    if not values:
        return []
    k = 2 / (length + 1)
    ema_vals = [values[0]]
    for price in values[1:]:
        ema_vals.append(price * k + ema_vals[-1] * (1 - k))
    return ema_vals


def _rsi(values: List[float], length: int) -> float:
    """
    Compute the Relative Strength Index (RSI) based on the last `length` periods.
    If insufficient data, return a neutral 50.
    """
    if length <= 0 or len(values) < length + 2:
        return 50.0
    gains: List[float] = []
    losses: List[float] = []
    for i in range(1, length + 1):
        delta = values[-i] - values[-(i + 1)]
        if delta >= 0:
            gains.append(delta)
            losses.append(0.0)
        else:
            gains.append(0.0)
            losses.append(-delta)
    avg_gain = sum(gains) / length
    avg_loss = sum(losses) / length
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _atr(highs: List[float], lows: List[float], closes: List[float], length: int) -> float:
    """
    Compute the Average True Range (ATR) based on the last `length` periods.
    If insufficient data, return 0.
    """
    if length <= 0 or len(highs) < length + 2:
        return 0.0
    trs: List[float] = []
    for i in range(1, length + 1):
        high = highs[-i]
        low = lows[-i]
        prev_close = closes[-(i + 1)]
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        trs.append(tr)
    return sum(trs) / length

# Cooldown state: number of bars remaining before new trades are allowed
_last_signal_bars_remaining = 0


def decide() -> Tuple[Signal, str, Dict]:
    """
    Determine whether to BUY, SELL, or HOLD based on EMA crossover, RSI, and ATR.
    Returns a tuple: (signal, reason, diagnostics). The diagnostics dict includes
    indicator values to assist with debugging and logging.
    """
    global _last_signal_bars_remaining
    # Enforce cooldown period
    if _last_signal_bars_remaining > 0:
        _last_signal_bars_remaining -= 1
        return "HOLD", "cooldown", {}

    candles = _fetch_candles()
    fast = settings.STRAT_EMA_FAST
    slow = settings.STRAT_EMA_SLOW
    rsi_len = settings.STRAT_RSI_LEN
    atr_len = settings.ATR_LEN
    # Need enough candles for slow EMA and indicators
    min_bars = max(fast, slow, rsi_len, atr_len) + 2
    if not candles or len(candles) < min_bars:
        return "HOLD", "not-enough-data", {}

    closes = [c["c"] for c in candles]
    highs = [c["h"] for c in candles]
    lows = [c["l"] for c in candles]

    ema_fast = _ema(closes, fast)
    ema_slow = _ema(closes, slow)
    ema_fast_prev, ema_fast_curr = ema_fast[-2], ema_fast[-1]
    ema_slow_prev, ema_slow_curr = ema_slow[-2], ema_slow[-1]
    rsi_val = _rsi(closes, rsi_len)
    atr_val = _atr(highs, lows, closes, atr_len)

    diagnostics = {
        "ema_fast": ema_fast_curr,
        "ema_slow": ema_slow_curr,
        "rsi": rsi_val,
        "atr": atr_val,
    }

    # Crossover detection
    cross_up = ema_fast_prev <= ema_slow_prev and ema_fast_curr > ema_slow_curr
    cross_down = ema_fast_prev >= ema_slow_prev and ema_fast_curr < ema_slow_curr

    # Evaluate buy/sell conditions
    if (
        cross_up
        and rsi_val > settings.STRAT_RSI_BUY
        and atr_val >= settings.MIN_ATR
    ):
        _last_signal_bars_remaining = settings.STRAT_COOLDOWN_BARS
        return "BUY", "ema_up & rsi_high", diagnostics
    elif (
        cross_down
        and rsi_val < settings.STRAT_RSI_SELL
        and atr_val >= settings.MIN_ATR
    ):
        _last_signal_bars_remaining = settings.STRAT_COOLDOWN_BARS
        return "SELL", "ema_down & rsi_low", diagnostics

    return "HOLD", "no-signal", diagnostics
