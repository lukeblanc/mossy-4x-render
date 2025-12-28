from __future__ import annotations

import math
import os
from datetime import datetime
from typing import Dict, Iterable, Mapping, Tuple


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _true_ranges(candles: Iterable[Mapping[str, float]]) -> Tuple[float, ...]:
    ranges = []
    last_close = None
    for candle in candles:
        try:
            high = float(candle["h"])
            low = float(candle["l"])
            close = float(candle["c"])
        except (KeyError, TypeError, ValueError):
            continue
        if last_close is None:
            last_close = close
            continue
        tr = max(high - low, abs(high - last_close), abs(low - last_close))
        ranges.append(tr)
        last_close = close
    return tuple(ranges)


def _atr_series(true_ranges: Tuple[float, ...], length: int) -> Tuple[float, ...]:
    if length <= 0 or len(true_ranges) < length:
        return tuple()
    atr_values = []
    for idx in range(length, len(true_ranges) + 1):
        window = true_ranges[idx - length : idx]
        atr_values.append(sum(window) / length)
    return tuple(atr_values)


def _volatility_label(current_atr: float, true_ranges: Tuple[float, ...], atr_length: int) -> str:
    if current_atr <= 0 or math.isnan(current_atr):
        return "NORMAL"
    atr_values = _atr_series(true_ranges, atr_length)
    if not atr_values:
        return "NORMAL"
    rolling_mean = sum(atr_values) / len(atr_values)
    if rolling_mean <= 0 or math.isnan(rolling_mean):
        return "NORMAL"
    ratio = current_atr / rolling_mean
    if ratio < 0.9:
        return "LOW"
    if ratio > 1.1:
        return "HIGH"
    return "NORMAL"


def project_market(
    pair: str,
    candles: Iterable[Mapping[str, float]],
    indicators: Mapping[str, float],
    now_utc: datetime,
) -> Dict[str, object]:
    ema_fast = indicators.get("ema_fast", math.nan)
    ema_slow = indicators.get("ema_slow", math.nan)
    atr = indicators.get("atr", math.nan)
    rsi = indicators.get("rsi", math.nan)
    last_close = indicators.get("close", math.nan)

    candle_buffer = tuple(candles)

    atr_length = int(os.getenv("PROJECTOR_ATR_LENGTH", 14))

    slope_score = 0.0
    if all(not math.isnan(val) for val in (ema_fast, ema_slow, atr)):
        slope_score = _clamp((ema_fast - ema_slow) / max(atr, 1e-6), -1.0, 1.0)

    momentum_score = 0.0
    if not math.isnan(rsi):
        momentum_score = _clamp((rsi - 50.0) / 25.0, -1.0, 1.0)

    bias_score = 0.6 * slope_score + 0.4 * momentum_score
    if bias_score > 0.2:
        bias = "BULL"
    elif bias_score < -0.2:
        bias = "BEAR"
    else:
        bias = "NEUTRAL"

    try:
        horizon_env = int(os.getenv("PROJECTOR_HORIZON", 4))
    except ValueError:
        horizon_env = 4
    horizon = max(1, horizon_env)
    if math.isnan(last_close) and candle_buffer:
        try:
            last_close = float(candle_buffer[-1]["c"])
        except Exception:
            last_close = math.nan

    if math.isnan(atr):
        range_half = 0.0
    else:
        range_half = atr * math.sqrt(horizon)
    center = last_close if not math.isnan(last_close) else 0.0
    projected_low = center - range_half
    projected_high = center + range_half

    tr_list = _true_ranges(candle_buffer)
    volatility = _volatility_label(atr, tr_list, atr_length)

    confidence = 50.0
    confidence += 20.0 * abs(bias_score)
    if volatility == "NORMAL":
        confidence += 10.0
    if volatility == "HIGH":
        confidence -= 10.0
    if bias == "NEUTRAL":
        confidence -= 10.0
    confidence = _clamp(confidence, 0.0, 100.0)

    return {
        "pair": pair,
        "timestamp": now_utc,
        "bias": bias,
        "bias_score": bias_score,
        "range": {
            "center": center,
            "low": projected_low,
            "high": projected_high,
            "horizon": horizon,
        },
        "volatility": volatility,
        "confidence": confidence,
    }


__all__ = ["project_market"]
