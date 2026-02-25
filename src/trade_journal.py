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
        conn.execute("PRAGMA busy_timeout=5000;")
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
                    run_tag TEXT,
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
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS trade_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    instrument TEXT,
                    direction TEXT,
                    entry_price REAL,
                    exit_price REAL,
                    profit REAL,
                    reason TEXT,
                    equity_after REAL
                );
                """
            )
            # Migration-safe: add run_tag if missing.
            columns = {row[1] for row in conn.execute("PRAGMA table_info(trades);").fetchall()}
            if "run_tag" not in columns:
                conn.execute("ALTER TABLE trades ADD COLUMN run_tag TEXT;")

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
        run_tag: Optional[str] = None,
        gating_flags: Mapping[str, Any],
        indicators_snapshot: Mapping[str, Any],
        equity_after: Optional[float] = None,
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
            "run_tag": run_tag,
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
                    run_tag,
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
                    :run_tag,
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
                    run_tag=COALESCE(excluded.run_tag, trades.run_tag),
                    gating_flags=excluded.gating_flags,
                    indicators_snapshot=excluded.indicators_snapshot;
                """,
                payload,
            )
            conn.execute(
                """
                INSERT INTO trade_events (
                    timestamp,
                    instrument,
                    direction,
                    entry_price,
                    exit_price,
                    profit,
                    reason,
                    equity_after
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    payload["timestamp_utc"],
                    instrument,
                    side,
                    entry_price,
                    None,
                    None,
                    "OPEN",
                    equity_after,
                ),
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
        run_tag: Optional[str] = None,
        instrument: Optional[str] = None,
        direction: Optional[str] = None,
        entry_price: Optional[float] = None,
        equity_after: Optional[float] = None,
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
            "run_tag": run_tag,
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
                    broker_confirmed,
                    run_tag
                ) VALUES (
                    :trade_id,
                    :exit_timestamp_utc,
                    :exit_price,
                    :spread_at_exit,
                    :max_profit_ccy,
                    :realized_pnl_ccy,
                    :exit_reason,
                    :duration_seconds,
                    :broker_confirmed,
                    :run_tag
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
                    END,
                    run_tag=COALESCE(excluded.run_tag, trades.run_tag);
                """,
                payload,
            )
            existing_trade = conn.execute(
                "SELECT instrument, side, entry_price FROM trades WHERE trade_id=?",
                (str(trade_id),),
            ).fetchone()
            resolved_instrument = instrument if instrument is not None else (existing_trade[0] if existing_trade else None)
            resolved_direction = direction if direction is not None else (existing_trade[1] if existing_trade else None)
            resolved_entry = entry_price if entry_price is not None else (existing_trade[2] if existing_trade else None)
            conn.execute(
                """
                INSERT INTO trade_events (
                    timestamp,
                    instrument,
                    direction,
                    entry_price,
                    exit_price,
                    profit,
                    reason,
                    equity_after
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    payload["exit_timestamp_utc"],
                    resolved_instrument,
                    resolved_direction,
                    resolved_entry,
                    exit_price,
                    realized_pnl_ccy,
                    exit_reason,
                    equity_after,
                ),
            )

    def count_trade_events(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) FROM trade_events").fetchone()
            return int(row[0] if row else 0)


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _compute_segment_metrics(trades: list[dict[str, Any]]) -> dict[str, float | int]:
    total_trades = len(trades)
    wins = [trade["pnl"] for trade in trades if trade["pnl"] > 0]
    losses = [trade["pnl"] for trade in trades if trade["pnl"] < 0]

    win_count = len(wins)
    loss_count = len(losses)
    win_rate_ratio = _safe_div(win_count, total_trades)
    win_rate_pct = win_rate_ratio * 100.0

    avg_win = _safe_div(sum(wins), win_count)
    avg_loss = _safe_div(sum(losses), loss_count)

    gross_profit = sum(wins)
    gross_loss = sum(losses)
    profit_factor = _safe_div(gross_profit, abs(gross_loss))
    expectancy = (win_rate_ratio * avg_win) - ((1.0 - win_rate_ratio) * abs(avg_loss))

    avg_trade_duration = _safe_div(
        sum(float(trade.get("duration_seconds") or 0.0) for trade in trades),
        total_trades,
    )

    return {
        "total_trades": total_trades,
        "wins": win_count,
        "losses": loss_count,
        "win_rate_pct": win_rate_pct,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "avg_trade_duration": avg_trade_duration,
    }


def _max_drawdown_and_losing_streak(trades: list[dict[str, Any]]) -> tuple[float, int]:
    max_drawdown = 0.0
    longest_losing_streak = 0
    current_losing_streak = 0

    peak_equity = 0.0
    equity = 0.0

    for trade in trades:
        pnl = float(trade["pnl"])
        equity += pnl
        peak_equity = max(peak_equity, equity)
        max_drawdown = max(max_drawdown, peak_equity - equity)

        if pnl < 0:
            current_losing_streak += 1
            longest_losing_streak = max(longest_losing_streak, current_losing_streak)
        else:
            current_losing_streak = 0

    return max_drawdown, longest_losing_streak


def run_performance_analysis(db_path: Path | str | None = None) -> None:
    """Compute and print performance analytics from all closed trades in trade_journal.db."""

    db_file = Path(db_path) if db_path is not None else default_journal_path()
    if not db_file.exists():
        print(f"[PERFORMANCE_SUMMARY]\nerror=database_not_found\npath={db_file}", flush=True)
        return

    conn = sqlite3.connect(db_file)
    try:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT
                trade_id,
                instrument,
                COALESCE(realized_pnl_ccy, 0.0) AS pnl,
                COALESCE(duration_seconds, 0) AS duration_seconds,
                exit_timestamp_utc
            FROM trades
            WHERE exit_timestamp_utc IS NOT NULL
            ORDER BY exit_timestamp_utc ASC, trade_id ASC
            """
        ).fetchall()
    finally:
        conn.close()

    trades = [
        {
            "trade_id": row["trade_id"],
            "instrument": (row["instrument"] or "UNKNOWN").upper(),
            "pnl": float(row["pnl"] or 0.0),
            "duration_seconds": int(row["duration_seconds"] or 0),
        }
        for row in rows
    ]

    metrics = _compute_segment_metrics(trades)
    max_drawdown, longest_losing_streak = _max_drawdown_and_losing_streak(trades)

    print("[PERFORMANCE_SUMMARY]", flush=True)
    print(f"total_trades={metrics['total_trades']}", flush=True)
    print(f"win_rate={metrics['win_rate_pct']:.2f}", flush=True)
    print(f"wins={metrics['wins']}", flush=True)
    print(f"losses={metrics['losses']}", flush=True)
    print(f"avg_win={metrics['avg_win']:.2f}", flush=True)
    print(f"avg_loss={metrics['avg_loss']:.2f}", flush=True)
    print(f"gross_profit={metrics['gross_profit']:.2f}", flush=True)
    print(f"gross_loss={metrics['gross_loss']:.2f}", flush=True)
    print(f"profit_factor={metrics['profit_factor']:.4f}", flush=True)
    print(f"expectancy={metrics['expectancy']:.4f}", flush=True)
    print(f"max_drawdown={max_drawdown:.2f}", flush=True)
    print(f"longest_losing_streak={longest_losing_streak}", flush=True)
    print(f"avg_trade_duration={metrics['avg_trade_duration']:.2f}", flush=True)

    by_instrument: dict[str, list[dict[str, Any]]] = {}
    for trade in trades:
        by_instrument.setdefault(trade["instrument"], []).append(trade)

    print("\n[PERFORMANCE_BY_INSTRUMENT]", flush=True)
    for instrument in sorted(by_instrument.keys()):
        segment_metrics = _compute_segment_metrics(by_instrument[instrument])
        print(f"instrument={instrument}", flush=True)
        print(f"trades={segment_metrics['total_trades']}", flush=True)
        print(f"win_rate={segment_metrics['win_rate_pct']:.2f}", flush=True)
        print(f"avg_win={segment_metrics['avg_win']:.2f}", flush=True)
        print(f"avg_loss={segment_metrics['avg_loss']:.2f}", flush=True)
        print(f"expectancy={segment_metrics['expectancy']:.4f}", flush=True)
        print(f"profit_factor={segment_metrics['profit_factor']:.4f}", flush=True)
        print("", flush=True)


__all__ = ["TradeJournal", "default_journal_path", "run_performance_analysis"]
