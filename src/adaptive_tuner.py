from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path


@dataclass
class AdaptiveSnapshot:
    closed_trades: int
    wins: int
    losses: int
    loss_streak: int
    risk_multiplier: float
    source: str


class AdaptiveTuner:
    """Read-only adaptive sizing helper based on recent closed-trade outcomes."""

    def __init__(self, db_path: Path, *, lookback: int = 40, min_sample: int = 10) -> None:
        self.db_path = Path(db_path)
        self.lookback = max(10, int(lookback))
        self.min_sample = max(3, int(min_sample))

    def _load_recent_pnl_from_trades(self, conn: sqlite3.Connection) -> list[float]:
        rows = conn.execute(
            """
            SELECT COALESCE(realized_pnl_ccy, 0.0) AS pnl
            FROM trades
            WHERE exit_timestamp_utc IS NOT NULL
            ORDER BY exit_timestamp_utc DESC
            LIMIT ?
            """,
            (self.lookback,),
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

    @staticmethod
    def _loss_streak(recent_desc: list[float]) -> int:
        streak = 0
        for pnl in recent_desc:
            if pnl < 0:
                streak += 1
            else:
                break
        return streak

    def snapshot(self) -> AdaptiveSnapshot:
        recent, source = self._load_recent_pnl()
        closed = len(recent)
        wins = sum(1 for pnl in recent if pnl > 0)
        losses = sum(1 for pnl in recent if pnl < 0)
        loss_streak = self._loss_streak(recent)

        # Fast-start conservative mode until sufficient sample is available.
        if closed < self.min_sample:
            multiplier = 0.85
        else:
            win_rate = wins / closed if closed else 0.0
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

        return AdaptiveSnapshot(
            closed_trades=closed,
            wins=wins,
            losses=losses,
            loss_streak=loss_streak,
            risk_multiplier=max(0.5, min(1.0, multiplier)),
            source=source,
        )
