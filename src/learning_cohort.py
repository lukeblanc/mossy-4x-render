from __future__ import annotations

import math
import sqlite3
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Optional


DEFAULT_LEARNING_RUN_TAG = "MINI_RUN"
DEFAULT_LEARNING_COHORT_START_UTC = "2026-07-13T12:47:00+00:00"
FUTURE_TIMESTAMP_TOLERANCE = timedelta(seconds=60)


@dataclass(frozen=True)
class CleanOutcome:
    """A closed outcome that is safe for the learning-only consumers.

    The broker confirmation and invalidation checks live in this module so the
    tuner, setup policy, and shadow learner cannot silently define different
    evidence cohorts.
    """

    trade_id: str
    entry_timestamp_utc: Optional[str]
    exit_timestamp_utc: str
    instrument: str
    side: str
    session_id: str
    indicators_snapshot: str
    realized_pnl_ccy: float
    run_tag: Optional[str]


def _parse_utc(value: object) -> Optional[datetime]:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except (TypeError, ValueError, OverflowError):
        return None


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (name,),
    ).fetchone()
    return row is not None


def load_clean_outcomes(
    db_path: Path | str,
    *,
    instruments: Iterable[str] | None = None,
    run_tag: str | None = DEFAULT_LEARNING_RUN_TAG,
    entry_start_utc: str | None = DEFAULT_LEARNING_COHORT_START_UTC,
    exit_start_utc: str | None = None,
    as_of_utc: str | datetime | None = None,
    limit: int | None = None,
    descending: bool = False,
) -> list[CleanOutcome]:
    """Load the single verified cohort used by every adaptive learner.

    A qualifying outcome must use the configured nonblank run tag, begin after
    the cohort boundary, be broker-confirmed and closed, have finite realized
    P&L, have ordered entry/exit timestamps that are knowable as of the supplied
    time, and have no entry-resolution record. If a malformed database contains
    the same trade ID more than once, every row for that ambiguous ID is
    excluded. Database/schema failures return an empty cohort so learning cannot
    interrupt the trading loop.
    """

    path = Path(db_path)
    if not path.exists():
        return []

    normalized_instruments = {
        str(instrument).strip().upper()
        for instrument in (instruments or ())
        if str(instrument).strip()
    }
    normalized_run_tag = str(run_tag or "").strip() or DEFAULT_LEARNING_RUN_TAG
    normalized_entry_start = (
        str(entry_start_utc or "").strip() or DEFAULT_LEARNING_COHORT_START_UTC
    )
    entry_start = _parse_utc(normalized_entry_start)
    exit_start = _parse_utc(exit_start_utc) if exit_start_utc else None
    if isinstance(as_of_utc, datetime):
        try:
            as_of = as_of_utc
            if as_of.tzinfo is None:
                as_of = as_of.replace(tzinfo=timezone.utc)
            as_of = as_of.astimezone(timezone.utc)
        except (ValueError, OverflowError):
            as_of = None
    elif as_of_utc is not None:
        as_of = _parse_utc(as_of_utc)
    else:
        as_of = datetime.now(timezone.utc)
    if entry_start is None or (exit_start_utc and exit_start is None) or as_of is None:
        return []
    try:
        latest_allowed_exit = as_of + FUTURE_TIMESTAMP_TOLERANCE
    except OverflowError:
        return []

    try:
        conn = sqlite3.connect(path, timeout=2.0)
        conn.row_factory = sqlite3.Row
        try:
            if not _table_exists(conn, "trades"):
                return []
            columns = {
                str(row[1])
                for row in conn.execute("PRAGMA table_info(trades)").fetchall()
                if len(row) > 1
            }
            required = {
                "trade_id",
                "timestamp_utc",
                "exit_timestamp_utc",
                "realized_pnl_ccy",
                "broker_confirmed",
            }
            if not required.issubset(columns):
                return []
            if "run_tag" not in columns:
                return []

            optional = {
                "timestamp_utc": "NULL",
                "instrument": "''",
                "side": "''",
                "session_id": "''",
                "indicators_snapshot": "'{}'",
                "run_tag": "NULL",
            }
            projections = [
                name if name in columns else f"{fallback} AS {name}"
                for name, fallback in optional.items()
            ]
            rows = conn.execute(
                f"""
                SELECT trade_id, exit_timestamp_utc, realized_pnl_ccy,
                       {', '.join(projections)}
                FROM trades
                WHERE broker_confirmed = 1
                  AND exit_timestamp_utc IS NOT NULL
                  AND TRIM(exit_timestamp_utc) <> ''
                  AND realized_pnl_ccy IS NOT NULL
                """
            ).fetchall()

            invalidated: set[str] = set()
            if _table_exists(conn, "journal_entry_resolutions"):
                invalidated = {
                    str(row[0]).strip()
                    for row in conn.execute(
                        "SELECT trade_id FROM journal_entry_resolutions"
                    ).fetchall()
                }
        finally:
            conn.close()
    except (OSError, sqlite3.Error):
        return []

    trade_ids = [str(row["trade_id"] or "").strip() for row in rows]
    duplicate_ids = {
        trade_id
        for trade_id, count in Counter(trade_ids).items()
        if trade_id and count > 1
    }
    loaded: list[tuple[datetime, CleanOutcome]] = []
    for row in rows:
        trade_id = str(row["trade_id"] or "").strip()
        if not trade_id or trade_id in invalidated or trade_id in duplicate_ids:
            continue
        exit_time = _parse_utc(row["exit_timestamp_utc"])
        if (
            exit_time is None
            or exit_time > latest_allowed_exit
            or (exit_start is not None and exit_time < exit_start)
        ):
            continue
        entry_timestamp = row["timestamp_utc"]
        parsed_entry = _parse_utc(entry_timestamp)
        if parsed_entry is None or parsed_entry > exit_time:
            continue
        if entry_start is not None:
            if parsed_entry < entry_start:
                continue
        try:
            pnl = float(row["realized_pnl_ccy"])
        except (TypeError, ValueError, OverflowError):
            continue
        if not math.isfinite(pnl):
            continue

        instrument = str(row["instrument"] or "").strip().upper()
        if normalized_instruments and instrument not in normalized_instruments:
            continue
        row_run_tag = (
            str(row["run_tag"]).strip() if row["run_tag"] is not None else None
        )
        if row_run_tag != normalized_run_tag:
            continue

        loaded.append(
            (
                exit_time,
                CleanOutcome(
                    trade_id=trade_id,
                    entry_timestamp_utc=parsed_entry.isoformat(),
                    exit_timestamp_utc=exit_time.isoformat(),
                    instrument=instrument,
                    side=str(row["side"] or "").strip().upper(),
                    session_id=str(row["session_id"] or ""),
                    indicators_snapshot=str(row["indicators_snapshot"] or "{}"),
                    realized_pnl_ccy=pnl,
                    run_tag=row_run_tag,
                ),
            )
        )

    loaded.sort(key=lambda item: (item[0], item[1].trade_id), reverse=descending)
    outcomes = [item[1] for item in loaded]
    if limit is not None:
        outcomes = outcomes[: max(0, int(limit))]
    return outcomes


__all__ = [
    "CleanOutcome",
    "DEFAULT_LEARNING_COHORT_START_UTC",
    "DEFAULT_LEARNING_RUN_TAG",
    "FUTURE_TIMESTAMP_TOLERANCE",
    "load_clean_outcomes",
]
