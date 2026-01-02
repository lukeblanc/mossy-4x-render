from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional

DEFAULT_DB_NAME = "trade_journal.db"


def default_journal_path(root: Optional[Path] = None) -> Path:
    """Return the default path for the trade journal, honoring MOSSY_STATE_PATH."""

    base_dir = root or Path(os.getenv("MOSSY_STATE_PATH", "data"))
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir / DEFAULT_DB_NAME


def _iso(ts: Optional[datetime]) -> Optional[str]:
    if ts is None:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc).isoformat()


def _json_dumps(payload: Mapping[str, Any] | None) -> str:
    try:
        return json.dumps(payload or {}, sort_keys=True, default=str)
    except TypeError:
        # Fallback to string representation for non-serializable payloads
        safe_payload: Dict[str, Any] = {}
        for key, value in (payload or {}).items():
            try:
                json.dumps(value)
                safe_payload[key] = value
            except TypeError:
                safe_payload[key] = str(value)
        return json.dumps(safe_payload, sort_keys=True)


@dataclass
class TradeJournal:
    """Lightweight SQLite-backed trade journal for entry/exit metadata."""

    db_path: Path | str | None = None

    def __post_init__(self) -> None:
        path = Path(self.db_path) if self.db_path else default_journal_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            self.path,
            timeout=1.5,
            isolation_level=None,  # autocommit
            check_same_thread=False,
        )
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS trades (
                    trade_id TEXT PRIMARY KEY,
                    timestamp_utc TEXT,
                    instrument TEXT,
                    side TEXT,
                    units REAL,
                    entry_price REAL,
                    stop_loss_price REAL,
                    take_profit_price REAL,
                    spread_at_entry REAL,
                    session_id TEXT,
                    session_mode TEXT,
                    gating_flags TEXT,
                    indicators_snapshot TEXT,
                    exit_timestamp_utc TEXT,
                    exit_price REAL,
                    spread_at_exit REAL,
                    max_profit_ccy REAL,
                    realized_pnl_ccy REAL,
                    exit_reason TEXT,
                    duration_seconds INTEGER,
                    broker_confirmed INTEGER
                );
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_trades_instrument_ts ON trades (instrument, timestamp_utc);"
            )

    def record_entry(
        self,
        *,
        trade_id: str,
        timestamp_utc: datetime,
        instrument: str,
        side: str,
        units: float,
        entry_price: Optional[float],
        stop_loss_price: Optional[float],
        take_profit_price: Optional[float],
        spread_at_entry: Optional[float],
        session_id: str,
        session_mode: str,
        gating_flags: Mapping[str, Any],
        indicators_snapshot: Mapping[str, Any],
    ) -> None:
        if not trade_id:
            return

        payload: MutableMapping[str, Any] = {
            "trade_id": str(trade_id),
            "timestamp_utc": _iso(timestamp_utc) or _iso(datetime.now(timezone.utc)),
            "instrument": instrument,
            "side": side,
            "units": units,
            "entry_price": entry_price,
            "stop_loss_price": stop_loss_price,
            "take_profit_price": take_profit_price,
            "spread_at_entry": spread_at_entry,
            "session_id": session_id,
            "session_mode": session_mode,
            "gating_flags": _json_dumps(gating_flags),
            "indicators_snapshot": _json_dumps(indicators_snapshot),
        }
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO trades (
                    trade_id,
                    timestamp_utc,
                    instrument,
                    side,
                    units,
                    entry_price,
                    stop_loss_price,
                    take_profit_price,
                    spread_at_entry,
                    session_id,
                    session_mode,
                    gating_flags,
                    indicators_snapshot
                ) VALUES (
                    :trade_id,
                    :timestamp_utc,
                    :instrument,
                    :side,
                    :units,
                    :entry_price,
                    :stop_loss_price,
                    :take_profit_price,
                    :spread_at_entry,
                    :session_id,
                    :session_mode,
                    :gating_flags,
                    :indicators_snapshot
                )
                ON CONFLICT(trade_id) DO UPDATE SET
                    timestamp_utc=excluded.timestamp_utc,
                    instrument=excluded.instrument,
                    side=excluded.side,
                    units=excluded.units,
                    entry_price=excluded.entry_price,
                    stop_loss_price=excluded.stop_loss_price,
                    take_profit_price=excluded.take_profit_price,
                    spread_at_entry=excluded.spread_at_entry,
                    session_id=excluded.session_id,
                    session_mode=excluded.session_mode,
                    gating_flags=excluded.gating_flags,
                    indicators_snapshot=excluded.indicators_snapshot;
                """,
                payload,
            )

    def record_exit(
        self,
        *,
        trade_id: str,
        exit_timestamp_utc: datetime,
        exit_price: Optional[float],
        spread_at_exit: Optional[float],
        max_profit_ccy: Optional[float],
        realized_pnl_ccy: Optional[float],
        exit_reason: Optional[str],
        duration_seconds: Optional[int],
        broker_confirmed: Optional[bool],
    ) -> None:
        if not trade_id:
            return

        payload: MutableMapping[str, Any] = {
            "trade_id": str(trade_id),
            "exit_timestamp_utc": _iso(exit_timestamp_utc) or _iso(datetime.now(timezone.utc)),
            "exit_price": exit_price,
            "spread_at_exit": spread_at_exit,
            "max_profit_ccy": max_profit_ccy,
            "realized_pnl_ccy": realized_pnl_ccy,
            "exit_reason": exit_reason,
            "duration_seconds": duration_seconds,
            "broker_confirmed": 1 if broker_confirmed else 0 if broker_confirmed is not None else None,
        }

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO trades (
                    trade_id,
                    exit_timestamp_utc,
                    exit_price,
                    spread_at_exit,
                    max_profit_ccy,
                    realized_pnl_ccy,
                    exit_reason,
                    duration_seconds,
                    broker_confirmed
                ) VALUES (
                    :trade_id,
                    :exit_timestamp_utc,
                    :exit_price,
                    :spread_at_exit,
                    :max_profit_ccy,
                    :realized_pnl_ccy,
                    :exit_reason,
                    :duration_seconds,
                    :broker_confirmed
                )
                ON CONFLICT(trade_id) DO UPDATE SET
                    exit_timestamp_utc=excluded.exit_timestamp_utc,
                    exit_price=COALESCE(excluded.exit_price, trades.exit_price),
                    spread_at_exit=COALESCE(excluded.spread_at_exit, trades.spread_at_exit),
                    max_profit_ccy=COALESCE(excluded.max_profit_ccy, trades.max_profit_ccy),
                    realized_pnl_ccy=COALESCE(excluded.realized_pnl_ccy, trades.realized_pnl_ccy),
                    exit_reason=COALESCE(excluded.exit_reason, trades.exit_reason),
                    duration_seconds=COALESCE(excluded.duration_seconds, trades.duration_seconds),
                    broker_confirmed=CASE
                        WHEN excluded.broker_confirmed IS NOT NULL THEN excluded.broker_confirmed
                        ELSE trades.broker_confirmed
                    END;
                """,
                payload,
            )


__all__ = ["TradeJournal", "default_journal_path"]
