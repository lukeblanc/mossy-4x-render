from __future__ import annotations

import inspect
import math
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

from src.learning_cohort import (
    DEFAULT_LEARNING_COHORT_START_UTC,
    DEFAULT_LEARNING_RUN_TAG,
    load_clean_outcomes,
)
from src.shadow_learner import ShadowReport, run_shadow_analysis


AdaptiveReasonCode = Literal[
    "small_sample",
    "loss_streak",
    "negative_expectancy",
    "profit_factor_below_one",
    "low_win_rate",
    "normal",
]


@dataclass
class AdaptiveSnapshot:
    lifetime_closed_trades: int
    session_closed_trades: int
    wins: int
    losses: int
    loss_streak: int
    risk_multiplier: float
    source: str = "none"
    filter_run_tag: str = "all"
    filter_window_start_utc: str = "none"
    filter_window_end_utc: str = "none"
    reason_code: AdaptiveReasonCode = "normal"


class AdaptiveTuner:
    """Conservative adaptive sizing based on verified closed-trade outcomes.

    The tuner never increases risk above the configured base risk. It evaluates
    loss streak, expectancy, profit factor and drawdown over a bounded recent
    window. Setup-specific learning is handled separately by adaptive_policy.
    Phase-two shadow learning evaluates candidates without changing orders.
    """

    def __init__(
        self,
        db_path: Path,
        *,
        lookback: int = 40,
        min_sample: int = 10,
        run_tag: Optional[str] = DEFAULT_LEARNING_RUN_TAG,
        window_start_utc: Optional[str] = DEFAULT_LEARNING_COHORT_START_UTC,
    ) -> None:
        self.db_path = Path(db_path)
        self.lookback = max(10, int(lookback))
        self.min_sample = max(3, int(min_sample))
        self.run_tag = (run_tag or "").strip() or DEFAULT_LEARNING_RUN_TAG
        self.window_start_utc = (
            (window_start_utc or "").strip() or DEFAULT_LEARNING_COHORT_START_UTC
        )
        self._shadow_lock = threading.Lock()
        # None means the analysis has never run. Using 0.0 can suppress the
        # first run on a newly created container whose monotonic clock is young.
        self._last_shadow_run_monotonic: Optional[float] = None
        self.shadow_report: Optional[ShadowReport] = None

    @staticmethod
    def _as_bool(value: object) -> bool:
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on", "y"}
        return bool(value)

    def _maybe_run_shadow_learning(self) -> None:
        if not self._as_bool(os.getenv("SHADOW_LEARNING_ENABLED", "true")):
            return
        try:
            interval_seconds = max(300.0, float(os.getenv("SHADOW_INTERVAL_SECONDS", "3600")))
        except ValueError:
            interval_seconds = 3600.0

        now_monotonic = time.monotonic()
        if (
            self._last_shadow_run_monotonic is not None
            and now_monotonic - self._last_shadow_run_monotonic < interval_seconds
        ):
            return
        if not self._shadow_lock.acquire(blocking=False):
            return
        try:
            now_monotonic = time.monotonic()
            if (
                self._last_shadow_run_monotonic is not None
                and now_monotonic - self._last_shadow_run_monotonic < interval_seconds
            ):
                return
            # Set before running so a failed analysis cannot hammer the journal
            # on every heartbeat. The next normal interval will retry.
            self._last_shadow_run_monotonic = now_monotonic
            self.shadow_report = run_shadow_analysis(
                self.db_path,
                cohort_start_utc=self.window_start_utc,
                cohort_run_tag=self.run_tag,
            )
        except Exception as exc:
            print(f"[SHADOW][WARN] analysis failed error={exc}", flush=True)
        finally:
            self._shadow_lock.release()

    def _load_recent_pnl(
        self, *, as_of_utc: datetime | None = None
    ) -> tuple[list[float], str]:
        outcomes = load_clean_outcomes(
            self.db_path,
            run_tag=self.run_tag,
            entry_start_utc=self.window_start_utc,
            as_of_utc=as_of_utc,
            limit=self.lookback,
            descending=True,
        )
        return [outcome.realized_pnl_ccy for outcome in outcomes], "trades"

    @staticmethod
    def _loss_streak(recent_desc: list[float]) -> int:
        streak = 0
        for pnl in recent_desc:
            if pnl < 0:
                streak += 1
            else:
                break
        return streak

    @staticmethod
    def _statistics(recent_desc: list[float]) -> dict[str, float]:
        if not recent_desc:
            return {
                "win_rate": 0.0,
                "expectancy": 0.0,
                "profit_factor": 0.0,
                "drawdown": 0.0,
            }
        wins = [pnl for pnl in recent_desc if pnl > 0]
        losses = [pnl for pnl in recent_desc if pnl < 0]
        weights = [
            math.exp(-index / max(8.0, len(recent_desc) / 2.0))
            for index in range(len(recent_desc))
        ]
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        equity = 0.0
        peak = 0.0
        drawdown = 0.0
        for pnl in reversed(recent_desc):
            equity += pnl
            peak = max(peak, equity)
            drawdown = max(drawdown, peak - equity)
        return {
            "win_rate": len(wins) / len(recent_desc),
            "expectancy": sum(pnl * weight for pnl, weight in zip(recent_desc, weights))
            / sum(weights),
            "profit_factor": gross_profit / gross_loss
            if gross_loss > 0
            else (99.0 if gross_profit > 0 else 0.0),
            "drawdown": drawdown,
        }

    @staticmethod
    def _build_snapshot(**payload) -> AdaptiveSnapshot:
        try:
            return AdaptiveSnapshot(**payload)
        except TypeError:
            params = inspect.signature(AdaptiveSnapshot).parameters
            return AdaptiveSnapshot(**{key: value for key, value in payload.items() if key in params})

    def snapshot(self) -> AdaptiveSnapshot:
        self._maybe_run_shadow_learning()
        as_of_utc = datetime.now(timezone.utc)
        recent, source = self._load_recent_pnl(as_of_utc=as_of_utc)
        session_closed = len(recent)
        wins = sum(1 for pnl in recent if pnl > 0)
        losses = sum(1 for pnl in recent if pnl < 0)
        loss_streak = self._loss_streak(recent)
        stats = self._statistics(recent)
        filter_window_end_utc = as_of_utc.replace(microsecond=0).isoformat()

        lifetime_closed = len(
            load_clean_outcomes(self.db_path, as_of_utc=as_of_utc)
        )

        if session_closed < self.min_sample:
            multiplier = 0.85
            reason_code: AdaptiveReasonCode = "small_sample"
        elif loss_streak >= 4:
            multiplier = 0.5
            reason_code = "loss_streak"
        elif loss_streak >= 3:
            multiplier = 0.6
            reason_code = "loss_streak"
        elif loss_streak >= 2:
            multiplier = 0.75
            reason_code = "loss_streak"
        elif stats["expectancy"] < 0 and stats["profit_factor"] < 0.8:
            multiplier = 0.65
            reason_code = "negative_expectancy"
        elif stats["expectancy"] < 0:
            multiplier = 0.8
            reason_code = "negative_expectancy"
        elif stats["profit_factor"] < 1.0:
            multiplier = 0.8
            reason_code = "profit_factor_below_one"
        elif stats["win_rate"] < 0.4:
            multiplier = 0.8
            reason_code = "low_win_rate"
        elif stats["win_rate"] > 0.6 and stats["profit_factor"] >= 1.25:
            multiplier = 1.0
            reason_code = "normal"
        else:
            multiplier = 0.9
            reason_code = "normal"

        return self._build_snapshot(
            lifetime_closed_trades=lifetime_closed,
            session_closed_trades=session_closed,
            wins=wins,
            losses=losses,
            loss_streak=loss_streak,
            risk_multiplier=max(0.5, min(1.0, multiplier)),
            source=source,
            filter_run_tag=self.run_tag,
            filter_window_start_utc=self.window_start_utc,
            filter_window_end_utc=filter_window_end_utc,
            reason_code=reason_code,
        )
