from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from src.risk_setup import resolve_state_dir
from src.trade_journal import default_journal_path

DEFAULT_EVENT_LOG = "audit_events.jsonl"


def _iso(ts: Optional[datetime]) -> Optional[str]:
    if ts is None:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc).isoformat()


@dataclass
class EventLogger:
    path: Path

    def log(self, event: str, payload: Dict[str, Any]) -> None:
        entry = {
            "ts": _iso(datetime.now(timezone.utc)),
            "event": event,
            **payload,
        }
        try:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, sort_keys=True, default=str) + "\n")
        except Exception:
            # Logging must never block trading.
            pass


_default_logger: Optional[EventLogger] = None


def get_event_logger(base_dir: Optional[Path] = None) -> EventLogger:
    global _default_logger
    if _default_logger is None or base_dir is not None:
        state_dir = resolve_state_dir(base_dir)
        path = state_dir / DEFAULT_EVENT_LOG
        path.parent.mkdir(parents=True, exist_ok=True)
        _default_logger = EventLogger(path)
    return _default_logger


def log_event(event: str, payload: Dict[str, Any]) -> None:
    logger = get_event_logger()
    logger.log(event, payload)


def health_report_payload(
    *,
    logger: EventLogger,
    uptime_seconds: float,
    errors_count: int,
    max_trade_cap_hit: bool,
    max_drawdown_pct: Optional[float],
) -> Dict[str, Any]:
    now_utc = datetime.now(timezone.utc)
    day_start = datetime(now_utc.year, now_utc.month, now_utc.day, tzinfo=timezone.utc)
    journal_path = default_journal_path()
    trades_today = 0
    worst_trade = None
    try:
        with sqlite3.connect(journal_path) as conn:
            trades_today = (
                conn.execute(
                    """
                    SELECT COUNT(*)
                    FROM audit_trades
                    WHERE timestamp_open >= ?
                    """,
                    (_iso(day_start),),
                ).fetchone()[0]
                or 0
            )
            worst_trade = conn.execute(
                """
                SELECT realized_pnl
                FROM audit_trades
                WHERE timestamp_open >= ?
                ORDER BY realized_pnl ASC
                LIMIT 1
                """,
                (_iso(day_start),),
            ).fetchone()
    except Exception:
        trades_today = 0
        worst_trade = None

    worst_trade_val = None
    if worst_trade and worst_trade[0] is not None:
        try:
            worst_trade_val = float(worst_trade[0])
        except (TypeError, ValueError):
            worst_trade_val = None

    return {
        "trades_today": int(trades_today),
        "max_trade_cap_hit": bool(max_trade_cap_hit),
        "worst_trade": worst_trade_val,
        "max_drawdown": max_drawdown_pct,
        "errors_count": int(errors_count),
        "uptime": uptime_seconds,
        "logger_path": str(logger.path),
        "journal_path": str(journal_path),
        "report_ts": _iso(now_utc),
    }
