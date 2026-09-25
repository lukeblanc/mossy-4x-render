from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_DATABASE_PATH = Path(__file__).resolve().parent / "runtime_status.db"


def database_path() -> Path:
    configured = os.getenv("MOSSY_MCP_DATABASE_PATH", "").strip()
    return Path(configured) if configured else DEFAULT_DATABASE_PATH


def _connect(path: Path | None = None) -> sqlite3.Connection:
    target = path or database_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(target)
    connection.row_factory = sqlite3.Row
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS runtime_heartbeat (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            observed_at TEXT NOT NULL,
            received_at TEXT NOT NULL,
            payload_json TEXT NOT NULL
        )
        """
    )
    return connection


def save_runtime_heartbeat(
    payload: dict, *, received_at: datetime | None = None, path: Path | None = None
) -> bool:
    """Save only a newer observation and return whether it replaced the snapshot."""

    received = received_at or datetime.now(timezone.utc)
    observed = str(payload["observed_at"])
    with _connect(path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO runtime_heartbeat (id, observed_at, received_at, payload_json)
            VALUES (1, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                observed_at = excluded.observed_at,
                received_at = excluded.received_at,
                payload_json = excluded.payload_json
            WHERE excluded.observed_at > runtime_heartbeat.observed_at
            """,
            (observed, received.isoformat(), json.dumps(payload, sort_keys=True)),
        )
        connection.commit()
        return cursor.rowcount > 0


def load_runtime_heartbeat(*, path: Path | None = None) -> dict | None:
    with _connect(path) as connection:
        row = connection.execute(
            "SELECT observed_at, received_at, payload_json FROM runtime_heartbeat WHERE id = 1"
        ).fetchone()
    if row is None:
        return None
    return {
        "observed_at": row["observed_at"],
        "received_at": row["received_at"],
        "payload": json.loads(row["payload_json"]),
    }
