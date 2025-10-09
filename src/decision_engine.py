from __future__ import annotations

import math
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, Iterable, List, Optional

import httpx

PRACTICE_BASE_URL = "https://api-fxpractice.oanda.com/v3"

DEFAULT_INSTRUMENTS: List[str] = [
    "EUR_USD",
    "AUD_USD",
    "GBP_USD",
    "USD_JPY",
    "XAU_USD",
]


@dataclass
class Evaluation:
    instrument: str
    signal: str
    diagnostics: Optional[Dict[str, float]]
    reason: str
    market_active: bool


def _default_now() -> datetime:
    return datetime.now(timezone.utc)


def _default_fetcher(
    instrument: str,
    *,
    count: int,
    granularity: str,
    api_key: Optional[str],
) -> List[Dict]:
    headers: Dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        with httpx.Client(base_url=PRACTICE_BASE_URL, headers=headers, timeout=15.0) as client:
            response = client.get(
                f"/instruments/{instrument}/candles",
                params={"count": str(count), "granularity": granularity, "price": "M"},
            )
            response.raise_for_status()
            return response.json().get("candles", [])
    except Exception as exc:  # pragma: no cover - network failure logging path
        print(f"[OANDA] Failed to fetch candles for {instrument}: {exc}", flush=True)
        return []


class DecisionEngine:
    def __init__(
        self,
        config: Dict,
        *,
        candle_fetcher: Optional[Callable[..., List[Dict]]] = None,
        now_fn: Callable[[], datetime] = _default_now,
    ) -> None:
        self.config = config
        self._now = now_fn
        self._fetcher = candle_fetcher or _default_fetcher
        self._cooldowns: Dict[str, datetime] = {}
        self._api_key = os.getenv("OANDA_API_KEY")
        self._instruments = self._resolve_instruments(config.get("instruments"))
        self.config["instruments"] = list(self._instruments)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def evaluate_all(self) -> List[Evaluation]:
        results: List[Evaluation] = []
        for instrument in self._instruments:
            evaluation = self._evaluate_instrument(instrument)
            results.append(evaluation)
        return results

    @property
    def instruments(self) -> List[str]:
        return list(self._instruments)

    def mark_trade(self, instrument: str) -> None:
        cooldown_minutes = int(self.config.get("cooldown_minutes", 0))
        if cooldown_minutes <= 0:
            return
        self._cooldowns[instrument] = self._now() + timedelta(minutes=cooldown_minutes)

    def position_size(self, instrument: str, diagnostics: Optional[Dict[str, float]] = None) -> int:
        balance = float(self.config.get("account_balance", 10_000))
        risk_pct = float(self.config.get("risk_per_trade", 0.01))
        atr = (diagnostics or {}).get("atr")
        if atr is None or math.isnan(atr) or atr <= 0:
            atr = float(self.config.get("min_atr", 0.0001) or 0.0001)
        risk_capital = max(balance * risk_pct, 1.0)
        if atr <= 0:
            return max(1, int(risk_capital))
        units = int(risk_capital / atr)
        return max(1, units)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _evaluate_instrument(self, instrument: str) -> Evaluation:
        granularity = self.config.get("timeframe", "M5")
        candle_count = int(self.config.get("candles_to_fetch", 200))
        raw_candles = self._fetcher(
            instrument,
            count=candle_count,
            granularity=granularity,
            api_key=self._api_key,
        )
        normalized = self._normalize_candles(raw_candles)
        if not normalized:
            self._log_scan(instrument, diagnostics=None)
            self._log_decision(
                instrument,
                signal="HOLD",
                reason="inactive-market",
                diagnostics=None,
                market_active=False,
            )
            return Evaluation(
                instrument=instrument,
                signal="HOLD",
                diagnostics=None,
                reason="inactive-market",
                market_active=False,
            )

        diagnostics = self._build_indicators(normalized)
        self._log_scan(instrument, diagnostics=diagnostics)

        signal, reason = self._generate_signal(diagnostics)

        cooldown_until = self._cooldowns.get(instrument)
        if cooldown_until and cooldown_until > self._now():
            signal = "HOLD"
            reason = "cooldown"

        self._log_decision(
            instrument,
            signal=signal,
            reason=reason,
            diagnostics=diagnostics,
            market_active=True,
        )
        return Evaluation(
            instrument=instrument,
            signal=signal,
            diagnostics=diagnostics,
            reason=reason,
            market_active=True,
        )

    def _resolve_instruments(self, configured: Optional[Iterable[str]]) -> List[str]:
        resolved: List[str] = []
        seen = set()

        if configured:
            for entry in configured:
                if not isinstance(entry, str):
                    continue
                symbol = entry.strip().upper()
                if not symbol or symbol in seen:
                    continue
                resolved.append(symbol)
                seen.add(symbol)

        for symbol in DEFAULT_INSTRUMENTS:
            if symbol not in seen:
                resolved.append(symbol)
                seen.add(symbol)

        return resolved

    def _log_scan(self, instrument: str, diagnostics: Optional[Dict[str, float]]) -> None:
        diag = diagnostics or {}
        ema_fast = self._format_value(diag.get("ema_fast"), decimals=5)
        ema_slow = self._format_value(diag.get("ema_slow"), decimals=5)
        rsi = self._format_value(diag.get("rsi"), decimals=2)
        atr = self._format_value(diag.get("atr"), decimals=5)
        print(
            (
                f"[SCAN] Evaluating {instrument} "
                f"ema_fast={ema_fast} ema_slow={ema_slow} rsi={rsi} atr={atr}"
            ),
            flush=True,
        )

    def _log_decision(
        self,
        instrument: str,
        *,
        signal: str,
        reason: str,
        diagnostics: Optional[Dict[str, float]],
        market_active: bool,
    ) -> None:
        diag = diagnostics or {}
        ema_fast = self._format_value(diag.get("ema_fast"), decimals=5)
        ema_slow = self._format_value(diag.get("ema_slow"), decimals=5)
        rsi = self._format_value(diag.get("rsi"), decimals=2)
        atr = self._format_value(diag.get("atr"), decimals=5)
        print(
            (
                f"[DECISION] {instrument} signal={signal} reason={reason} "
                f"market_active={market_active} "
                f"ema_fast={ema_fast} ema_slow={ema_slow} rsi={rsi} atr={atr}"
            ),
            flush=True,
        )

    def _format_value(self, value: Optional[float], *, decimals: int) -> str:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return "n/a"
        format_spec = f"{{:.{decimals}f}}"
        try:
            return format_spec.format(value)
        except (TypeError, ValueError):
            return "n/a"

    def _normalize_candles(self, candles: List[Dict]) -> List[Dict[str, float]]:
        normalized: List[Dict[str, float]] = []
        for candle in candles:
            if "mid" in candle:
                mids = candle.get("mid", {})
                raw_open, raw_high, raw_low, raw_close = (
                    mids.get("o"),
                    mids.get("h"),
                    mids.get("l"),
                    mids.get("c"),
                )
            else:
                raw_open = candle.get("o")
                raw_high = candle.get("h")
                raw_low = candle.get("l")
                raw_close = candle.get("c")
            try:
                o = float(raw_open)
                h = float(raw_high)
                l = float(raw_low)
                c = float(raw_close)
            except (TypeError, ValueError):
                continue
            normalized.append({"o": o, "h": h, "l": l, "c": c})
        return normalized

    def _build_indicators(self, candles: List[Dict[str, float]]) -> Dict[str, float]:
        closes = [c["c"] for c in candles]
        highs = [c["h"] for c in candles]
        lows = [c["l"] for c in candles]

        ema_fast_len = int(self.config.get("ema_fast", 10))
        ema_slow_len = int(self.config.get("ema_slow", 20))
        rsi_len = int(self.config.get("rsi_length", 14))
        atr_len = int(self.config.get("atr_length", 14))

        ema_fast_series = self._ema(closes, ema_fast_len)
        ema_slow_series = self._ema(closes, ema_slow_len)
        ema_fast = ema_fast_series[-1] if ema_fast_series else math.nan
        ema_slow = ema_slow_series[-1] if ema_slow_series else math.nan
        rsi_val = self._rsi(closes, rsi_len)
        atr_val = self._atr(highs, lows, closes, atr_len)

        return {
            "ema_fast": ema_fast,
            "ema_slow": ema_slow,
            "rsi": rsi_val,
            "atr": atr_val,
        }

    def _generate_signal(self, diagnostics: Dict[str, float]) -> (str, str):
        ema_fast = diagnostics.get("ema_fast", math.nan)
        ema_slow = diagnostics.get("ema_slow", math.nan)
        rsi_val = diagnostics.get("rsi", math.nan)
        atr_val = diagnostics.get("atr", 0.0)

        min_atr = float(self.config.get("min_atr", 0.0))
        if math.isnan(ema_fast) or math.isnan(ema_slow) or math.isnan(rsi_val):
            return "HOLD", "insufficient-data"
        if atr_val < min_atr:
            return "HOLD", "low-atr"

        rsi_buy = float(self.config.get("rsi_buy", 55))
        rsi_sell = float(self.config.get("rsi_sell", 45))

        if ema_fast > ema_slow and rsi_val >= rsi_buy:
            return "BUY", "bullish"
        if ema_fast < ema_slow and rsi_val <= rsi_sell:
            return "SELL", "bearish"
        return "HOLD", "neutral"

    # ------------------------------------------------------------------
    # Indicator helpers
    # ------------------------------------------------------------------
    def _ema(self, values: List[float], length: int) -> List[float]:
        if length <= 0 or not values:
            return []
        ema_vals = [values[0]]
        k = 2 / (length + 1)
        for price in values[1:]:
            ema_vals.append(price * k + ema_vals[-1] * (1 - k))
        return ema_vals

    def _rsi(self, values: List[float], length: int) -> float:
        if length <= 0 or len(values) < length + 1:
            return math.nan
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

    def _atr(
        self, highs: List[float], lows: List[float], closes: List[float], length: int
    ) -> float:
        if length <= 0 or len(highs) < length + 1:
            return math.nan
        true_ranges: List[float] = []
        for i in range(1, length + 1):
            high = highs[-i]
            low = lows[-i]
            prev_close = closes[-(i + 1)]
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            true_ranges.append(tr)
        return sum(true_ranges) / length


__all__ = ["DecisionEngine", "Evaluation"]
