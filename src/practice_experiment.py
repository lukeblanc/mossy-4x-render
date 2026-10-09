"""Opt-in AUD/USD notional experiment; durable reservations never auto-recover."""
from __future__ import annotations

from contextlib import closing, contextmanager
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import re
import sqlite3
from uuid import uuid4


class ExperimentBlocked(RuntimeError):
    """Bounded reason safe to include in diagnostics."""


class PracticeExperiment:
    """Ten new A$10 entries, then persistently stop entries for review.

    A nonempty MOSSY_PRACTICE_EXPERIMENT_ID opts in. Runtime never creates,
    activates, resets or repairs this ledger. Prepare/activate require a
    separate operator action; removing the setting is not an automatic exit.
    """

    def __init__(self, root: Path, experiment_id: str, account: str):
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", experiment_id) or not account:
            raise ExperimentBlocked("experiment-identity-invalid")
        self.path = root.resolve() / "practice_experiment.sqlite"
        self.id = experiment_id
        self.account_hash = hashlib.sha256(account.encode()).hexdigest()

    @classmethod
    def configured(cls, root: Path, account: str):
        experiment_id = os.getenv("MOSSY_PRACTICE_EXPERIMENT_ID", "")
        return cls(root, experiment_id, account) if experiment_id else None

    @contextmanager
    def _transaction(self):
        try:
            # mode=rw prevents a missing ledger silently becoming a new quota.
            with closing(sqlite3.connect(self.path.as_uri() + "?mode=rw", uri=True,
                                         timeout=5, isolation_level=None)) as connection:
                connection.row_factory = sqlite3.Row
                connection.execute("PRAGMA synchronous=FULL")
                connection.execute("BEGIN IMMEDIATE")
                try:
                    yield connection
                    connection.execute("COMMIT")
                except BaseException:
                    connection.execute("ROLLBACK")
                    raise
        except (sqlite3.Error, OSError, ValueError) as exc:
            raise ExperimentBlocked("experiment-ledger-unavailable") from exc

    def prepare(self):
        """Explicit local operator action only: create an inactive, fixed plan."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path)) as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS experiments (
                    id TEXT PRIMARY KEY, account_hash TEXT NOT NULL,
                    state TEXT NOT NULL CHECK(state IN ('prepared','active','paused','complete')),
                    filled_count INTEGER NOT NULL CHECK(filled_count BETWEEN 0 AND 10)
                );
                CREATE TABLE IF NOT EXISTS experiment_intents (
                    token TEXT PRIMARY KEY, experiment_id TEXT NOT NULL,
                    units INTEGER NOT NULL CHECK(units=10),
                    state TEXT NOT NULL CHECK(state IN ('reserved','skipped','filled')),
                    trade_id TEXT UNIQUE, created_at TEXT NOT NULL
                );
            """)
            connection.execute("INSERT INTO experiments VALUES (?,?,'prepared',0)",
                               (self.id, self.account_hash))
            connection.commit()

    def _snapshot(self, connection):
        row = connection.execute("SELECT * FROM experiments WHERE id=?", (self.id,)).fetchone()
        if row is None or row['account_hash'] != self.account_hash:
            raise ExperimentBlocked("experiment-account-or-plan-mismatch")
        filled = row['filled_count']
        recorded = connection.execute(
            "SELECT count(*) FROM experiment_intents WHERE experiment_id=? AND state='filled'",
            (self.id,)).fetchone()[0]
        if type(filled) is not int or not 0 <= filled <= 10 or filled != recorded:
            raise ExperimentBlocked("experiment-ledger-inconsistent")
        pending = connection.execute(
            "SELECT count(*) FROM experiment_intents WHERE experiment_id=? AND state='reserved'",
            (self.id,)).fetchone()[0]
        return {'state': row['state'], 'filled_count': filled, 'pending': pending, 'units': 10}

    def status(self):
        with self._transaction() as connection:
            return self._snapshot(connection)

    def last_opened_trade(self):
        with self._transaction() as connection:
            self._snapshot(connection)
            row = connection.execute(
                "SELECT trade_id FROM experiment_intents "
                "WHERE experiment_id=? AND state='filled' ORDER BY rowid DESC LIMIT 1",
                (self.id,)).fetchone()
            return row[0] if row is not None else None

    def activate(self):
        """Explicit approval step only; never restarts or clears a broker halt."""
        with self._transaction() as connection:
            snapshot = self._snapshot(connection)
            if snapshot['state'] != 'prepared' or snapshot['filled_count'] or snapshot['pending']:
                raise ExperimentBlocked("experiment-cannot-reactivate")
            connection.execute("UPDATE experiments SET state='active' WHERE id=?", (self.id,))

    def pause(self):
        """Persistently stop new entries; existing positions still need protection."""
        with self._transaction() as connection:
            self._snapshot(connection)
            connection.execute("UPDATE experiments SET state='paused' WHERE id=?", (self.id,))

    def ready_units(self):
        snapshot = self.status()
        self._require_ready(snapshot)
        return snapshot['units']

    @staticmethod
    def _require_ready(snapshot):
        if snapshot['state'] != 'active' or snapshot['filled_count'] >= 10:
            raise ExperimentBlocked("experiment-not-active")
        if snapshot['pending']:
            raise ExperimentBlocked("experiment-submission-unresolved")

    def reserve(self):
        with self._transaction() as connection:
            snapshot = self._snapshot(connection)
            self._require_ready(snapshot)
            token = uuid4().hex
            connection.execute("INSERT INTO experiment_intents VALUES (?,?,?,'reserved',NULL,?)",
                               (token, self.id, snapshot['units'], datetime.now(timezone.utc).isoformat()))
            return token, snapshot['units']

    def skip_before_submission(self, token):
        """Only the current caller may release a slot it knows was never sent."""
        with self._transaction() as connection:
            changed = connection.execute(
                "UPDATE experiment_intents SET state='skipped' "
                "WHERE token=? AND experiment_id=? AND state='reserved'", (token, self.id)).rowcount
            if changed != 1:
                raise ExperimentBlocked("experiment-reservation-mismatch")

    def confirm_opening(self, token, trade_id):
        ticket = str(trade_id)
        if not ticket.isascii() or not ticket.isdigit() or int(ticket) <= 0:
            raise ExperimentBlocked("experiment-fill-unverified")
        with self._transaction() as connection:
            self._snapshot(connection)
            intent = connection.execute(
                "SELECT * FROM experiment_intents WHERE token=? AND experiment_id=?",
                (token, self.id)).fetchone()
            if intent is not None and intent['state'] == 'filled' and intent['trade_id'] == ticket:
                return  # Exact repeated confirmation is idempotent.
            if intent is None or intent['state'] != 'reserved':
                raise ExperimentBlocked("experiment-reservation-mismatch")
            connection.execute("UPDATE experiment_intents SET state='filled', trade_id=? WHERE token=?",
                               (ticket, token))
            connection.execute(
                "UPDATE experiments SET filled_count=filled_count+1, "
                "state=CASE WHEN filled_count=9 THEN 'complete' ELSE state END WHERE id=?",
                (self.id,))
