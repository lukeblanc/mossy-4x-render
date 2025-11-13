from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Tuple

import httpx

from app.config import settings

# Signal type alias
Signal = str


ASIA_ADX_THRESHOLD = 25.0
ASIA_SIZE_MULTIPLIER = 0.5

# Trend-following thresholds
MIN_TREND_ADX = 15.0
RSI_LONG_THRESHOLD = 55.0
RSI_SHORT_THRESHOLD = 45.0


def _current_session(now_utc: datetime) -> str:
    """Return the trading session label based on the UTC hour."""

    hour = now_utc.hour
    if 12 <= hour < 21:
        return "new_york"
    if 7 <= hour < 12:
        return "london"
    return "asia"


def _session_gate(now_utc: datetime, adx_value: float) -> Tuple[bool, float, str]:
    """Determine if entries are allowed for the active session and size multiplier."""

    session = _current_session(now_utc)
    if session == "asia":
        if adx_value >= ASIA_ADX_THRESHOLD:
            return True, ASIA_SIZE_MULTIPLIER, session
        return False, 0.0, session
    return True, 1.0, session


def decide_signal(
    ema_fast: float,
    ema_slow: float,
    rsi: float,
    atr: float,
    adx: float,
    session: str,
) -> Tuple[Signal, str]:
    """Simple trend-following decision rule using EMA, RSI, and ADX filters."""

    del atr  # ATR unused in the simplified logic but kept for signature symmetry
    del session  # session not part of the simplified rules

    signal: Signal = "HOLD"
    reason = "no-signal"

    if adx is not None and adx >= MIN_TREND_ADX:
        if ema_fast > ema_slow and rsi is not None and rsi > RSI_LONG_THRESHOLD:
            signal = "BUY"
            reason = "trend-long"
        elif ema_fast < ema_slow and rsi is not None and rsi < RSI_SHORT_THRESHOLD:
            signal = "SELL"
            reason = "trend-short"
        else:
            reason = "trend-but-no-rsi-confirmation"
    else:
        reason = "no-trend-adx-low"

    return signal, reason


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
        for candle in data.get("candles", []):
            mids = candle.get("mid", {})
            try:
                candles.append(
                    {
                        "o": float(mids["o"]),
                        "h": float(mids["h"]),
                        "l": float(mids["l"]),
                        "c": float(mids["c"]),
                    }
                )
            except (KeyError, TypeError, ValueError):
                # Skip malformed candles rather than failing the entire fetch.
                continue
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


def _adx(highs: List[float], lows: List[float], closes: List[float], length: int) -> float:
    """Compute the Average Directional Index using Wilder's smoothing."""

    if length <= 0 or len(highs) < length + 1:
        return 0.0

    true_ranges: List[float] = []
    plus_dm: List[float] = []
    minus_dm: List[float] = []

    for i in range(1, len(highs)):
        up_move = highs[i] - highs[i - 1]
        down_move = lows[i - 1] - lows[i]
        plus = up_move if up_move > down_move and up_move > 0 else 0.0
        minus = down_move if down_move > up_move and down_move > 0 else 0.0
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        true_ranges.append(tr)
        plus_dm.append(plus)
        minus_dm.append(minus)

    if len(true_ranges) < length:
        return 0.0

    tr_smooth = sum(true_ranges[:length])
    plus_dm_smooth = sum(plus_dm[:length])
    minus_dm_smooth = sum(minus_dm[:length])

    def _calc_dx(tr_val: float, plus_val: float, minus_val: float) -> float:
        if tr_val <= 0:
            return 0.0
        plus_di = 100.0 * plus_val / tr_val
        minus_di = 100.0 * minus_val / tr_val
        denom = plus_di + minus_di
        if denom <= 0:
            return 0.0
        return 100.0 * abs(plus_di - minus_di) / denom

    dx_values: List[float] = []
    dx_values.append(_calc_dx(tr_smooth, plus_dm_smooth, minus_dm_smooth))

    for i in range(length, len(true_ranges)):
        tr_smooth = tr_smooth - (tr_smooth / length) + true_ranges[i]
        plus_dm_smooth = plus_dm_smooth - (plus_dm_smooth / length) + plus_dm[i]
        minus_dm_smooth = minus_dm_smooth - (minus_dm_smooth / length) + minus_dm[i]
        dx_values.append(_calc_dx(tr_smooth, plus_dm_smooth, minus_dm_smooth))

    if not dx_values:
        return 0.0

    initial_period = min(length, len(dx_values))
    adx = sum(dx_values[:initial_period]) / initial_period
    for dx in dx_values[initial_period:]:
        adx = ((adx * (length - 1)) + dx) / length
    return adx

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
    adx_len = atr_len
    # Need enough candles for slow EMA and indicators
    min_bars = max(fast, slow, rsi_len, atr_len, adx_len) + 2
    if not candles or len(candles) < min_bars:
        return "HOLD", "not-enough-data", {}

    closes = [c["c"] for c in candles]
    highs = [c["h"] for c in candles]
    lows = [c["l"] for c in candles]

    ema_fast = _ema(closes, fast)
    ema_slow = _ema(closes, slow)
    ema_fast_curr = ema_fast[-1]
    ema_slow_curr = ema_slow[-1]
    rsi_val = _rsi(closes, rsi_len)
    atr_val = _atr(highs, lows, closes, atr_len)
    adx_val = _adx(highs, lows, closes, adx_len)

    now_utc = datetime.now(timezone.utc)
    allowed, size_mult, session = _session_gate(now_utc, adx_val)
    diagnostics = {
        "ema_fast": ema_fast_curr,
        "ema_slow": ema_slow_curr,
        "rsi": rsi_val,
        "atr": atr_val,
        "adx": adx_val,
        "session": session,
        "size_multiplier": size_mult,
    }

    if not allowed:
        return "HOLD", "asia-low-adx", diagnostics

    signal, reason = decide_signal(
        ema_fast_curr,
        ema_slow_curr,
        rsi_val,
        atr_val,
        adx_val,
        session,
    )

    if signal in ("BUY", "SELL"):
        _last_signal_bars_remaining = settings.STRAT_COOLDOWN_BARS

    return signal, reason, diagnostics
