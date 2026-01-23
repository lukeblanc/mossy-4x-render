from __future__ import annotations

import math
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import httpx
from src.indicators import calculate_macd

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
    candles: Optional[List[Dict[str, float]]] = None


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

    url_path = f"/instruments/{instrument}/candles"
    params = {"count": str(count), "granularity": granularity, "price": "M"}

    with httpx.Client(base_url=PRACTICE_BASE_URL, headers=headers, timeout=15.0) as client:
        response = client.get(url_path, params=params)
        response.raise_for_status()
        payload = response.json()
        candles = payload.get("candles", [])
        if not isinstance(candles, list):
            return []
        return candles


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
        self._last_scan_success: Dict[str, bool] = {}
        self._fetch_retry_attempts = max(1, int(self.config.get("fetch_retry_attempts", 3)))
        self._fetch_retry_backoff = max(0.0, float(self.config.get("fetch_retry_backoff", 1.0)))

        merge_default = self._as_bool(self.config.get("merge_default_instruments", False))
        resolved_instruments = self._resolve_instruments(
            self.config.get("instruments"), merge_default
        )
        self.instruments: List[str] = resolved_instruments
        self.config["instruments"] = resolved_instruments
        print(
            "[CONFIG] instruments resolved="
            f"{resolved_instruments} merge_default_instruments={'true' if merge_default else 'false'}",
            flush=True,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def evaluate_all(self) -> List[Evaluation]:
        instruments: List[str] = list(self.instruments)
        results: List[Evaluation] = []
        self._last_scan_success.clear()
        for instrument in instruments:
            evaluation = self._evaluate_instrument(instrument)
            results.append(evaluation)
        if instruments and all(self._last_scan_success.get(inst) for inst in instruments):
            print("✅ Multi-Pair Candle Fetch Verified", flush=True)
        return results

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
        raw_candles = self._fetch_candles(
            instrument,
            candle_count=candle_count,
            granularity=granularity,
        )
        normalized = self._normalize_candles(raw_candles)
        if not normalized:
            self._log_signal(instrument, "HOLD", rsi=None, atr=None)
            return Evaluation(
                instrument=instrument,
                signal="HOLD",
                diagnostics=None,
                reason="inactive-market",
                market_active=False,
                candles=normalized,
            )

        diagnostics = self._build_indicators(normalized)
        signal, reason = self._generate_signal(diagnostics)

        cooldown_until = self._cooldowns.get(instrument)
        if cooldown_until and cooldown_until > self._now():
            signal = "HOLD"
            reason = "cooldown"

        self._log_signal(instrument, signal, diagnostics.get("rsi"), diagnostics.get("atr"))
        return Evaluation(
            instrument=instrument,
            signal=signal,
            diagnostics=diagnostics,
            reason=reason,
            market_active=True,
            candles=normalized,
        )

    def _fetch_candles(
        self,
        instrument: str,
        *,
        candle_count: int,
        granularity: str,
    ) -> List[Dict]:
        attempts = self._fetch_retry_attempts
        backoff = self._fetch_retry_backoff
        last_error: Optional[str] = None

        for attempt in range(1, attempts + 1):
            try:
                candles = self._fetcher(
                    instrument,
                    count=candle_count,
                    granularity=granularity,
                    api_key=self._api_key,
                )
            except httpx.HTTPStatusError as exc:
                status_code = exc.response.status_code if exc.response else "unknown"
                transient = isinstance(status_code, int) and (
                    status_code == 429 or 500 <= status_code < 600
                )
                last_error = f"{status_code}"
                if transient and attempt < attempts:
                    print(
                        f"[WARN] {instrument} fetch attempt {attempt} failed {status_code} – retrying",
                        flush=True,
                    )
                    if backoff > 0:
                        time.sleep(backoff)
                    continue
                print(
                    f"[WARN] {instrument} fetch failed {status_code} – skipping",
                    flush=True,
                )
                self._last_scan_success[instrument] = False
                return []
            except httpx.RequestError as exc:
                last_error = str(exc)
                if attempt < attempts:
                    print(
                        f"[WARN] {instrument} fetch attempt {attempt} failed network-error – retrying",
                        flush=True,
                    )
                    if backoff > 0:
                        time.sleep(backoff)
                    continue
                print(
                    f"[WARN] {instrument} fetch failed network-error – skipping",
                    flush=True,
                )
                self._last_scan_success[instrument] = False
                return []
            except Exception as exc:
                last_error = str(exc)
                print(f"[WARN] {instrument} fetch failed {exc} – skipping", flush=True)
                self._last_scan_success[instrument] = False
                return []

            candle_list = list(candles or [])
            print(f"[SCAN] {instrument} OK ({len(candle_list)} bars)", flush=True)
            self._last_scan_success[instrument] = True
            return candle_list

        if last_error:
            print(
                f"[WARN] {instrument} fetch failed after {attempts} attempts ({last_error})",
                flush=True,
            )
        self._last_scan_success[instrument] = False
        return []

    def _log_signal(self, instrument: str, signal: str, rsi: Optional[float], atr: Optional[float]) -> None:
        if rsi is None or math.isnan(rsi):
            rsi_str = "n/a"
        else:
            rsi_str = f"{rsi:.2f}"
        if atr is None or math.isnan(atr):
            atr_str = "n/a"
        else:
            atr_str = f"{atr:.5f}"
        print(f"[SIGNAL] {instrument} signal={signal} rsi={rsi_str} atr={atr_str}", flush=True)

    def _resolve_instruments(
        self, instruments: Optional[Iterable[str]], merge_default_instruments: bool
    ) -> List[str]:
        provided: Sequence = []
        if instruments is None:
            provided = []
        elif isinstance(instruments, str):
            tokens = [
                token.strip()
                for token in re.split(r"[,\s]+", instruments)
                if token.strip()
            ]
            provided = tokens
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

        ema_fast_len = int(self.config.get("ema_fast", 12))
        ema_slow_len = int(self.config.get("ema_slow", 26))
        ema_trend_fast_len = int(self.config.get("ema_trend_fast", 50))
        ema_trend_slow_len = int(self.config.get("ema_trend_slow", 200))
        rsi_len = int(self.config.get("rsi_length", 14))
        atr_len = int(self.config.get("atr_length", 14))

        ema_fast_series = self._ema(closes, ema_fast_len)
        ema_slow_series = self._ema(closes, ema_slow_len)
        ema_trend_fast_series = self._ema(closes, ema_trend_fast_len)
        ema_trend_slow_series = self._ema(closes, ema_trend_slow_len)
        ema_fast = ema_fast_series[-1] if ema_fast_series else math.nan
        ema_slow = ema_slow_series[-1] if ema_slow_series else math.nan
        ema_trend_fast = ema_trend_fast_series[-1] if ema_trend_fast_series else math.nan
        ema_trend_slow = ema_trend_slow_series[-1] if ema_trend_slow_series else math.nan
        rsi_val = self._rsi(closes, rsi_len)
        rsi_prev = self._rsi(closes[:-1], rsi_len)
        rsi_slope = math.nan
        if not math.isnan(rsi_val) and not math.isnan(rsi_prev):
            rsi_slope = rsi_val - rsi_prev
        atr_val, atr_baseline = self._atr_with_baseline(highs, lows, closes, atr_len, baseline_window=50)
        macd_line, macd_signal, macd_histogram = calculate_macd(
            closes,
            fast_length=12,
            slow_length=26,
            signal_length=9,
        )
        macd_line_prev, macd_signal_prev, macd_histogram_prev = calculate_macd(
            closes[:-1],
            fast_length=12,
            slow_length=26,
            signal_length=9,
        )

        last_close = closes[-1] if closes else math.nan
        return {
            "ema_fast": ema_fast,
            "ema_slow": ema_slow,
            "ema_trend_fast": ema_trend_fast,
            "ema_trend_slow": ema_trend_slow,
            "rsi": rsi_val,
            "rsi_prev": rsi_prev,
            "rsi_slope": rsi_slope,
            "atr": atr_val,
            "atr_baseline_50": atr_baseline,
            "close": last_close,
            "macd_line": macd_line,
            "macd_signal": macd_signal,
            "macd_histogram": macd_histogram,
            "macd_histogram_prev": macd_histogram_prev,
        }

    def _generate_signal(self, diagnostics: Dict[str, float]) -> (str, str):
        ema_fast = diagnostics.get("ema_fast", math.nan)
        ema_slow = diagnostics.get("ema_slow", math.nan)
        rsi_val = diagnostics.get("rsi", math.nan)
        rsi_slope = diagnostics.get("rsi_slope", math.nan)
        close_price = diagnostics.get("close", math.nan)
        ema_trend_fast = diagnostics.get("ema_trend_fast", math.nan)
        atr_val = diagnostics.get("atr", 0.0)

        min_atr = float(self.config.get("min_atr", 0.0))
        if math.isnan(ema_fast) or math.isnan(ema_slow) or math.isnan(rsi_val):
            return "HOLD", "insufficient-data"
        if atr_val < min_atr:
            return "HOLD", "low-atr"

        rsi_buy = float(self.config.get("rsi_buy", 52))
        rsi_sell = float(self.config.get("rsi_sell", 48))
        rsi_deadband = float(self.config.get("rsi_deadband", 1.0))
        if not math.isnan(rsi_val) and abs(rsi_val - 50.0) < rsi_deadband:
            return "HOLD", "rsi-jitter"

        rsi_slope_min = float(self.config.get("rsi_slope_min", 0.0))
        if rsi_slope_min < 0:
            rsi_slope_min = 0.0
        slope_rising = not math.isnan(rsi_slope) and rsi_slope >= rsi_slope_min
        slope_falling = not math.isnan(rsi_slope) and rsi_slope <= -rsi_slope_min
        close_above_ema = (
            not math.isnan(close_price)
            and not math.isnan(ema_trend_fast)
            and close_price >= ema_trend_fast
        )
        close_below_ema = (
            not math.isnan(close_price)
            and not math.isnan(ema_trend_fast)
            and close_price <= ema_trend_fast
        )

        if ema_fast > ema_slow and rsi_val >= rsi_buy:
            if slope_rising or close_above_ema:
                return "BUY", "bullish"
        if ema_fast < ema_slow and rsi_val <= rsi_sell:
            if slope_falling or close_below_ema:
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

    def _atr_with_baseline(
        self,
        highs: List[float],
        lows: List[float],
        closes: List[float],
        length: int,
        *,
        baseline_window: int = 50,
    ) -> Tuple[float, float]:
        if length <= 0 or len(highs) < length + 1:
            return math.nan, math.nan

        true_ranges: List[float] = []
        for i in range(1, len(highs)):
            high = highs[-i]
            low = lows[-i]
            prev_close = closes[-(i + 1)]
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            true_ranges.insert(0, tr)

        atr_values: List[float] = []
        if true_ranges:
            rolling_atr = true_ranges[0]
            atr_values.append(rolling_atr)
            k = 2 / (length + 1)
            for tr in true_ranges[1:]:
                rolling_atr = tr * k + rolling_atr * (1 - k)
                atr_values.append(rolling_atr)

        if len(atr_values) < 1:
            return math.nan, math.nan

        current_atr = atr_values[-1]
        baseline_slice = atr_values[-baseline_window:] if baseline_window > 0 else atr_values
        if baseline_slice:
            baseline_atr = sum(baseline_slice) / len(baseline_slice)
        else:
            baseline_atr = math.nan

        return current_atr, baseline_atr


__all__ = ["DecisionEngine", "Evaluation", "DEFAULT_INSTRUMENTS"]
