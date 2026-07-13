from __future__ import annotations

import inspect
import math
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.shadow_learner import ShadowReport, run_shadow_analysis


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
        run_tag: Optional[str] = None,
        window_start_utc: Optional[str] = None,
    ) -> None:
        self.db_path = Path(db_path)
        self.lookback = max(10, int(lookback))
        self.min_sample = max(3, int(min_sample))
        self.run_tag = (run_tag or "").strip() or None
        self.window_start_utc = (window_start_utc or "").strip() or None
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
            self.shadow_report = run_shadow_analysis(self.db_path)
        except Exception as exc:
            print(f"[SHADOW][WARN] analysis failed error={exc}", flush=True)
        finally:
            self._shadow_lock.release()

    @staticmethod
    def _trade_columns(conn: sqlite3.Connection) -> set[str]:
        return {
            str(row[1])
            for row in conn.execute("PRAGMA table_info(trades)").fetchall()
            if len(row) > 1
        }

    def _load_recent_pnl_from_trades(self, conn: sqlite3.Connection) -> list[float]:
        columns = self._trade_columns(conn)
        params: list[object] = []
        predicates = ["exit_timestamp_utc IS NOT NULL", "realized_pnl_ccy IS NOT NULL"]
        if self.run_tag and "run_tag" in columns:
            predicates.append("run_tag = ?")
            params.append(self.run_tag)
        if self.window_start_utc:
            predicates.append("exit_timestamp_utc >= ?")
            params.append(self.window_start_utc)

        rows = conn.execute(
            f"""
            SELECT COALESCE(realized_pnl_ccy, 0.0) AS pnl
            FROM trades
            WHERE {' AND '.join(predicates)}
            ORDER BY exit_timestamp_utc DESC
            LIMIT ?
            """,
            (*params, self.lookback),
        ).fetchall()
        return [float(row[0] or 0.0) for row in rows]

    def _load_recent_pnl_from_events(self, conn: sqlite3.Connection) -> list[float]:
        rows = conn.execute(
            """
            SELECT COALESCE(profit, 0.0) AS pnl
            FROM trade_events
            WHERE reason != 'OPEN' AND profit IS NOT NULL
            ORDER BY id DESC
            LIMIT ?
            """,
            (self.lookback,),
        ).fetchall()
        return [float(row[0] or 0.0) for row in rows]

    def _load_recent_pnl(self) -> tuple[list[float], str]:
        if not self.db_path.exists():
            return [], "none"
        try:
            conn = sqlite3.connect(self.db_path, timeout=2.0)
            try:
                pnl_trades = self._load_recent_pnl_from_trades(conn)
                if len(pnl_trades) >= self.min_sample:
                    return pnl_trades, "trades"

                # A configured run tag or cohort start is a safety boundary.
                # Never bypass it by falling back to the legacy unfiltered event
                # table merely because the clean sample is still small.
                if self.run_tag or self.window_start_utc:
                    return pnl_trades, "trades"

                pnl_events = self._load_recent_pnl_from_events(conn)
                if pnl_events:
                    return pnl_events, "trade_events"
                return pnl_trades, "trades"
            finally:
                conn.close()
        except sqlite3.Error:
            return [], "none"

    def _count_trades(self, conn: sqlite3.Connection) -> tuple[int, int]:
        columns = self._trade_columns(conn)
        lifetime_row = conn.execute(
            "SELECT COUNT(*) FROM trades WHERE exit_timestamp_utc IS NOT NULL"
        ).fetchone()
        lifetime = int(lifetime_row[0] if lifetime_row else 0)

        params: list[object] = []
        predicates = ["exit_timestamp_utc IS NOT NULL"]
        if self.run_tag and "run_tag" in columns:
            predicates.append("run_tag = ?")
            params.append(self.run_tag)
        if self.window_start_utc:
            predicates.append("exit_timestamp_utc >= ?")
            params.append(self.window_start_utc)
        session_row = conn.execute(
            f"SELECT COUNT(*) FROM trades WHERE {' AND '.join(predicates)}",
            params,
        ).fetchone()
        return lifetime, int(session_row[0] if session_row else 0)

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
        recent, source = self._load_recent_pnl()
        session_closed = len(recent)
        wins = sum(1 for pnl in recent if pnl > 0)
        losses = sum(1 for pnl in recent if pnl < 0)
        loss_streak = self._loss_streak(recent)
        stats = self._statistics(recent)
        filter_window_end_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

        lifetime_closed = 0
        if self.db_path.exists():
            try:
                conn = sqlite3.connect(self.db_path, timeout=2.0)
                try:
                    lifetime_closed, _ = self._count_trades(conn)
                finally:
                    conn.close()
            except sqlite3.Error:
                lifetime_closed = 0

        if session_closed < self.min_sample:
            multiplier = 0.85
        elif loss_streak >= 4:
            multiplier = 0.5
        elif loss_streak >= 3:
            multiplier = 0.6
        elif loss_streak >= 2:
            multiplier = 0.75
        elif stats["expectancy"] < 0 and stats["profit_factor"] < 0.8:
            multiplier = 0.65
        elif stats["expectancy"] < 0 or stats["profit_factor"] < 1.0:
            multiplier = 0.8
        elif stats["win_rate"] < 0.4:
            multiplier = 0.8
        elif stats["win_rate"] > 0.6 and stats["profit_factor"] >= 1.25:
            multiplier = 1.0
        else:
            multiplier = 0.9

        return self._build_snapshot(
            lifetime_closed_trades=lifetime_closed,
            session_closed_trades=session_closed,
            wins=wins,
            losses=losses,
            loss_streak=loss_streak,
            risk_multiplier=max(0.5, min(1.0, multiplier)),
            source=source,
            filter_run_tag=self.run_tag or "all",
            filter_window_start_utc=self.window_start_utc or "none",
            filter_window_end_utc=filter_window_end_utc,
        )
