from __future__ import annotations

import math
from typing import List, Tuple


def _ema(values: List[float], length: int) -> List[float]:
    if length <= 0 or not values:
        return []
    ema_vals = [values[0]]
    k = 2 / (length + 1)
    for price in values[1:]:
        ema_vals.append(price * k + ema_vals[-1] * (1 - k))
    return ema_vals


def calculate_macd(
    values: List[float],
    *,
    fast_length: int = 12,
    slow_length: int = 26,
    signal_length: int = 9,
) -> Tuple[float, float, float]:
    if (
        not values
        or fast_length <= 0
        or slow_length <= 0
        or signal_length <= 0
        or len(values) < slow_length
    ):
        return math.nan, math.nan, math.nan

    ema_fast = _ema(values, fast_length)
    ema_slow = _ema(values, slow_length)
    if not ema_fast or not ema_slow:
        return math.nan, math.nan, math.nan

    macd_series = [fast - slow for fast, slow in zip(ema_fast, ema_slow)]
    if not macd_series:
        return math.nan, math.nan, math.nan

    signal_series = _ema(macd_series, signal_length)
    if not signal_series:
        return math.nan, math.nan, math.nan

    macd_line = macd_series[-1]
    signal_line = signal_series[-1]
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


__all__ = ["calculate_macd"]
