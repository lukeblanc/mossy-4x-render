from __future__ import annotations

import inspect
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


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
    """Read-only adaptive sizing helper based on recent closed-trade outcomes."""

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

    def _load_recent_pnl_from_trades(self, conn: sqlite3.Connection) -> list[float]:
        columns = self._trade_columns(conn)
        params: list[object] = []
        predicates = ["exit_timestamp_utc IS NOT NULL"]
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

        conn = sqlite3.connect(self.db_path)
        try:
            pnl_trades = self._load_recent_pnl_from_trades(conn)
            if len(pnl_trades) >= self.min_sample:
                return pnl_trades, "trades"

            pnl_events = self._load_recent_pnl_from_events(conn)
            if pnl_events:
                return pnl_events, "trade_events"

            return pnl_trades, "trades"
        finally:
            conn.close()

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
        session = int(session_row[0] if session_row else 0)
        return lifetime, session

    @staticmethod
    def _trade_columns(conn: sqlite3.Connection) -> set[str]:
        rows = conn.execute("PRAGMA table_info(trades)").fetchall()
        return {str(row[1]) for row in rows if len(row) > 1}

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
    def _build_snapshot(
        *,
        lifetime_closed_trades: int,
        session_closed_trades: int,
        wins: int,
        losses: int,
        loss_streak: int,
        risk_multiplier: float,
        source: str,
        filter_run_tag: str,
        filter_window_start_utc: str,
        filter_window_end_utc: str,
    ) -> AdaptiveSnapshot:
        """Build snapshot compatibly across mixed deployments.

        Some environments may still have an older AdaptiveSnapshot signature
        without the ``source`` field when stale bytecode is loaded.
        """
        payload = {
            "lifetime_closed_trades": lifetime_closed_trades,
            "session_closed_trades": session_closed_trades,
            "wins": wins,
            "losses": losses,
            "loss_streak": loss_streak,
            "risk_multiplier": risk_multiplier,
            "source": source,
            "filter_run_tag": filter_run_tag,
            "filter_window_start_utc": filter_window_start_utc,
            "filter_window_end_utc": filter_window_end_utc,
        }
        try:
            return AdaptiveSnapshot(**payload)
        except TypeError:
            params = inspect.signature(AdaptiveSnapshot).parameters
            compatible_payload = {key: value for key, value in payload.items() if key in params}
            return AdaptiveSnapshot(**compatible_payload)

    def snapshot(self) -> AdaptiveSnapshot:
        recent, source = self._load_recent_pnl()
        session_closed = len(recent)
        wins = sum(1 for pnl in recent if pnl > 0)
        losses = sum(1 for pnl in recent if pnl < 0)
        loss_streak = self._loss_streak(recent)
        filter_window_end_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

        lifetime_closed = 0
        if self.db_path.exists():
            conn = sqlite3.connect(self.db_path)
            try:
                lifetime_closed, _ = self._count_trades(conn)
            finally:
                conn.close()

        # Fast-start conservative mode until sufficient sample is available.
        if session_closed < self.min_sample:
            multiplier = 0.85
        else:
            win_rate = wins / session_closed if session_closed else 0.0
            if loss_streak >= 3:
                multiplier = 0.6
            elif loss_streak >= 2:
                multiplier = 0.75
            elif win_rate < 0.4:
                multiplier = 0.8
            elif win_rate > 0.6:
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
