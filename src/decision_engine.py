from __future__ import annotations

import math
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, Iterable, List, Optional, Sequence

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

        merge_default = self._as_bool(self.config.get("merge_default_instruments", False))
        resolved_instruments = self._resolve_instruments(
            self.config.get("instruments"), merge_default
        )
        self.instruments: List[str] = resolved_instruments
        self.config["instruments"] = resolved_instruments
        print(
            "[CONFIG] instruments resolved="
            f"{resolved_instruments} merge_default_instruments={str(merge_default).lower()}",
            flush=True,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def evaluate_all(self) -> List[Evaluation]:
        candle_count = int(self.config.get("candles_to_fetch", 200))
        granularity = self.config.get("timeframe", "M5")
        batched_candles = self.fetch_candles(
            count=candle_count,
            granularity=granularity,
        )

        instruments: Iterable[str] = self.instruments
        results: List[Evaluation] = []
        for instrument in instruments:
            raw_candles = batched_candles.get(instrument, [])
            evaluation = self._evaluate_instrument(instrument, raw_candles)
            results.append(evaluation)
        return results

    def fetch_candles(
        self,
        instruments: Optional[Iterable[str]] = None,
        *,
        count: Optional[int] = None,
        granularity: Optional[str] = None,
    ) -> Dict[str, List[Dict]]:
        targets = list(instruments or self.instruments)
        candle_count = int(count if count is not None else self.config.get("candles_to_fetch", 200))
        timeframe = granularity or self.config.get("timeframe", "M5")

        aggregated: Dict[str, List[Dict]] = {}
        for instrument in targets:
            try:
                candles = self._fetcher(
                    instrument,
                    count=candle_count,
                    granularity=timeframe,
                    api_key=self._api_key,
                )
            except Exception as exc:  # pragma: no cover - defensive logging
                print(
                    f"[OANDA] Candle fetch failed instrument={instrument} error={exc}",
                    flush=True,
                )
                candles = []

            if not isinstance(candles, list):
                candles = []

            aggregated[instrument] = candles

        return aggregated

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
    def _evaluate_instrument(
        self, instrument: str, raw_candles: Optional[Sequence[Dict]] = None
    ) -> Evaluation:
        if raw_candles is None:
            single_instrument = self.fetch_candles([instrument])
            raw_candles = single_instrument.get(instrument, [])

        normalized = self._normalize_candles(list(raw_candles or []))
        if not normalized:
            self._log_scan(instrument, "HOLD", rsi=None, atr=None)
            return Evaluation(
                instrument=instrument,
                signal="HOLD",
                diagnostics=None,
                reason="inactive-market",
                market_active=False,
            )

        diagnostics = self._build_indicators(normalized)
        signal, reason = self._generate_signal(diagnostics)

        cooldown_until = self._cooldowns.get(instrument)
        if cooldown_until and cooldown_until > self._now():
            signal = "HOLD"
            reason = "cooldown"

        self._log_scan(instrument, signal, diagnostics.get("rsi"), diagnostics.get("atr"))
        return Evaluation(
            instrument=instrument,
            signal=signal,
            diagnostics=diagnostics,
            reason=reason,
            market_active=True,
        )

    def _log_scan(self, instrument: str, signal: str, rsi: Optional[float], atr: Optional[float]) -> None:
        if rsi is None or math.isnan(rsi):
            rsi_str = "n/a"
        else:
            rsi_str = f"{rsi:.2f}"
        if atr is None or math.isnan(atr):
            atr_str = "n/a"
        else:
            atr_str = f"{atr:.5f}"
        print(f"[SCAN] {instrument} signal={signal} rsi={rsi_str} atr={atr_str}", flush=True)

    def _resolve_instruments(
        self, instruments: Optional[Iterable[str]], merge_default_instruments: bool
    ) -> List[str]:
        provided: Sequence = []
        if instruments is None:
            provided = []
        elif isinstance(instruments, str):
            provided = [instruments]
        elif isinstance(instruments, (set, frozenset)):
            provided = sorted(instruments, key=self._instrument_sort_key)
        else:
            try:
                provided = list(instruments)  # type: ignore[arg-type]
            except TypeError:
                provided = [instruments]  # type: ignore[list-item]

        normalized: List[str] = []
        seen = set()

        for entry in provided:
            if not isinstance(entry, str):
                continue
            candidate = entry.strip().upper()
            if not candidate:
                continue
            if candidate in seen:
                continue
            normalized.append(candidate)
            seen.add(candidate)

        if merge_default_instruments:
            for default in DEFAULT_INSTRUMENTS:
                if default not in seen:
                    normalized.append(default)
                    seen.add(default)

        return normalized

    def _instrument_sort_key(self, instrument: Optional[str]) -> tuple[int, str]:
        candidate = ""
        if isinstance(instrument, str):
            candidate = instrument.strip().upper()
        index = self._default_instrument_index(candidate)
        return index, candidate

    @staticmethod
    def _default_instrument_index(instrument: str) -> int:
        try:
            return DEFAULT_INSTRUMENTS.index(instrument)
        except ValueError:
            return len(DEFAULT_INSTRUMENTS)

    @staticmethod
    def _as_bool(value: object) -> bool:
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on", "y"}
        return bool(value)

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


__all__ = ["DecisionEngine", "Evaluation", "DEFAULT_INSTRUMENTS"]
