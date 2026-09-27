from __future__ import annotations

import hashlib
import json
import math
import queue
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Optional


SCHEMA_VERSION = 1
# Broker timestamps can arrive a fraction after the local cycle timestamp.
# One second accommodates clock/serialization jitter without admitting a
# genuinely future completed candle into the learning cohort.
COMPLETED_BAR_FUTURE_TOLERANCE = timedelta(seconds=1)
GATE_STAGES = frozenset(
    {
        "cycle",
        "signal",
        "session",
        "exposure",
        "spread",
        "risk",
        "weekend",
        "trend",
        "xau",
        "macd",
        "orb",
        "sizing",
        "order",
        "executed",
    }
)
_FEATURE_KEYS = (
    "ema_fast",
    "ema_slow",
    "ema_trend_fast",
    "ema_trend_slow",
    "rsi",
    "rsi_prev",
    "rsi_slope",
    "atr",
    "atr_baseline_50",
    "close",
    "macd_line",
    "macd_signal",
    "macd_histogram",
    "macd_histogram_prev",
)


def _utc(value: datetime) -> datetime:
    timestamp = value
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc)


def _utc_iso(value: datetime) -> str:
    return _utc(value).isoformat()


def _minute_utc_iso(value: datetime) -> str:
    return _utc(value).replace(second=0, microsecond=0).isoformat()


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
        allow_nan=True,
    )


def configuration_fingerprint(config: Mapping[str, Any]) -> str:
    """Hash resolved configuration without persisting any configuration value."""

    return hashlib.sha256(_canonical_json(config).encode("utf-8")).hexdigest()


def _finite_number(value: object) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _safe_features(diagnostics: Mapping[str, Any] | None) -> dict[str, Optional[float]]:
    values = diagnostics or {}
    return {key: _finite_number(values.get(key)) for key in _FEATURE_KEYS}


def _gate_stage(value: object) -> str:
    stage = str(value or "unknown").strip().lower().replace("_", "-") or "unknown"
    return stage if stage in GATE_STAGES else "unknown"


def _side_hint(diagnostics: Mapping[str, Any] | None) -> str:
    values = diagnostics or {}
    fast = _finite_number(values.get("ema_fast"))
    slow = _finite_number(values.get("ema_slow"))
    if fast is None or slow is None:
        return "UNKNOWN"
    if fast > slow:
        return "BUY"
    if fast < slow:
        return "SELL"
    return "UNKNOWN"


def _opportunity_id(
    *,
    instrument: str,
    timeframe: str,
    completed_bar_open_utc: datetime,
    runtime_revision: str,
    config_fingerprint: str,
) -> str:
    identity = {
        "schema_version": SCHEMA_VERSION,
        "instrument": instrument,
        "timeframe": timeframe,
        "completed_bar_open_utc": _utc_iso(completed_bar_open_utc),
        "runtime_revision": runtime_revision,
        "config_fingerprint": config_fingerprint,
    }
    return hashlib.sha256(_canonical_json(identity).encode("utf-8")).hexdigest()


def _decision_event_id(*, opportunity_id: str, observed_at_utc: str) -> str:
    identity = {
        "schema_version": SCHEMA_VERSION,
        "opportunity_id": opportunity_id,
        "observed_at_utc": observed_at_utc,
    }
    return hashlib.sha256(_canonical_json(identity).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class DecisionObservation:
    """One stable completed-bar opportunity plus one runtime decision event."""

    opportunity_id: str
    decision_event_id: str
    observed_at_utc: str
    evaluation_tick_utc: str
    completed_bar_open_utc: str
    completed_bar_close_utc: str
    instrument: str
    timeframe: str
    observer_signal: str
    observer_side_hint: str
    observer_signal_reason: str
    runtime_signal: str
    runtime_signal_reason: str
    market_active: bool
    final_outcome: str
    gate_stage: str
    gate_reason: str
    session_mode: str
    session_name: str
    spread_pips: Optional[float]
    reference_price: Optional[float]
    stop_distance: Optional[float]
    target_distance: Optional[float]
    features_json: str
    runtime_revision: str
    config_fingerprint: str

    @property
    def observation_id(self) -> str:
        """Compatibility alias for the exact-timestamp decision event ID."""

        return self.decision_event_id

    def opportunity_row(self) -> tuple[object, ...]:
        return (
            self.opportunity_id,
            SCHEMA_VERSION,
            self.completed_bar_open_utc,
            self.completed_bar_close_utc,
            self.instrument,
            self.timeframe,
            self.observer_signal,
            self.observer_side_hint,
            self.observer_signal_reason,
            self.reference_price,
            self.stop_distance,
            self.target_distance,
            self.features_json,
            self.runtime_revision,
            self.config_fingerprint,
        )

    def event_row(self) -> tuple[object, ...]:
        return (
            self.decision_event_id,
            self.opportunity_id,
            self.observed_at_utc,
            self.evaluation_tick_utc,
            self.runtime_signal,
            self.runtime_signal_reason,
            1 if self.market_active else 0,
            self.final_outcome,
            self.gate_stage,
            self.gate_reason,
            self.session_mode,
            self.session_name,
            self.spread_pips,
        )


@dataclass
class DecisionObservationDraft:
    observed_at_utc: datetime
    instrument: str
    timeframe: str
    runtime_signal: str
    runtime_signal_reason: str
    market_active: bool
    completed_bar_open_utc: Optional[datetime]
    completed_bar_close_utc: Optional[datetime]
    observer_signal: Optional[str]
    observer_signal_reason: Optional[str]
    completed_diagnostics: Mapping[str, Any] | None
    runtime_revision: str
    config_fingerprint: str
    reference_price: Optional[float] = None
    stop_distance: Optional[float] = None
    target_distance: Optional[float] = None
    session_mode: str = "UNKNOWN"
    session_name: str = "UNKNOWN"
    spread_pips: Optional[float] = None
    _finalized: bool = field(default=False, init=False, repr=False)

    @property
    def finalized(self) -> bool:
        return self._finalized

    def set_session(self, *, mode: object, name: object) -> None:
        self.session_mode = str(mode or "UNKNOWN").strip().upper() or "UNKNOWN"
        self.session_name = str(name or "UNKNOWN").strip().upper() or "UNKNOWN"

    def set_spread(self, spread_pips: object) -> None:
        self.spread_pips = _finite_number(spread_pips)

    def finalize(
        self,
        *,
        outcome: object,
        gate_stage: object,
        gate_reason: object,
    ) -> Optional[DecisionObservation]:
        """Freeze one terminal event; repeat calls are harmless."""

        if self._finalized:
            return None
        self._finalized = True
        if self.completed_bar_open_utc is None or self.completed_bar_close_utc is None:
            return None

        try:
            observed_at = _utc(self.observed_at_utc)
            bar_open = _utc(self.completed_bar_open_utc)
            bar_close = _utc(self.completed_bar_close_utc)
        except (AttributeError, TypeError, ValueError, OverflowError):
            return None
        if bar_open >= bar_close:
            return None
        if bar_close > observed_at + COMPLETED_BAR_FUTURE_TOLERANCE:
            return None

        instrument = str(self.instrument or "").strip().upper()
        timeframe = str(self.timeframe or "").strip().upper()
        revision = str(self.runtime_revision or "unknown").strip() or "unknown"
        fingerprint = str(self.config_fingerprint or "").strip().lower()
        if not instrument or not timeframe or len(fingerprint) != 64:
            return None

        observer_signal = str(self.observer_signal or "HOLD").strip().upper() or "HOLD"
        observer_reason = str(self.observer_signal_reason or "unknown").strip().lower() or "unknown"
        opportunity_id = _opportunity_id(
            instrument=instrument,
            timeframe=timeframe,
            completed_bar_open_utc=self.completed_bar_open_utc,
            runtime_revision=revision,
            config_fingerprint=fingerprint,
        )
        observed_at_utc = _utc_iso(observed_at)
        evaluation_tick_utc = _minute_utc_iso(observed_at)
        features = _safe_features(self.completed_diagnostics)
        return DecisionObservation(
            opportunity_id=opportunity_id,
            decision_event_id=_decision_event_id(
                opportunity_id=opportunity_id,
                observed_at_utc=observed_at_utc,
            ),
            observed_at_utc=observed_at_utc,
            evaluation_tick_utc=evaluation_tick_utc,
            completed_bar_open_utc=_utc_iso(bar_open),
            completed_bar_close_utc=_utc_iso(bar_close),
            instrument=instrument,
            timeframe=timeframe,
            observer_signal=observer_signal,
            observer_side_hint=_side_hint(self.completed_diagnostics),
            observer_signal_reason=observer_reason,
            runtime_signal=str(self.runtime_signal or "HOLD").strip().upper() or "HOLD",
            runtime_signal_reason=str(self.runtime_signal_reason or "unknown").strip().lower() or "unknown",
            market_active=bool(self.market_active),
            final_outcome=str(outcome or "unknown").strip().lower() or "unknown",
            gate_stage=_gate_stage(gate_stage),
            gate_reason=str(gate_reason or "unknown").strip().lower().replace("_", "-") or "unknown",
            session_mode=self.session_mode,
            session_name=self.session_name,
            spread_pips=self.spread_pips,
            reference_price=_finite_number(self.reference_price),
            stop_distance=_finite_number(self.stop_distance),
            target_distance=_finite_number(self.target_distance),
            features_json=json.dumps(features, sort_keys=True, separators=(",", ":"), allow_nan=False),
            runtime_revision=revision,
            config_fingerprint=fingerprint,
        )


class DecisionObservationLedger:
    """Two-table immutable ledger stored away from the order journal."""

    def __init__(self, db_path: Path | str) -> None:
        self.path = Path(db_path)
        self._schema_ready = False
        self._schema_lock = threading.Lock()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.path,
            timeout=1.5,
            isolation_level=None,
            check_same_thread=False,
        )
        connection.execute("PRAGMA busy_timeout=5000;")
        connection.execute("PRAGMA journal_mode=WAL;")
        connection.execute("PRAGMA synchronous=NORMAL;")
        connection.execute("PRAGMA foreign_keys=ON;")
        return connection

    def _ensure_schema(self) -> None:
        if self._schema_ready:
            return
        with self._schema_lock:
            if self._schema_ready:
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self._connect() as connection:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS learning_opportunities (
                        opportunity_id TEXT PRIMARY KEY,
                        schema_version INTEGER NOT NULL,
                        completed_bar_open_utc TEXT NOT NULL,
                        completed_bar_close_utc TEXT NOT NULL,
                        instrument TEXT NOT NULL,
                        timeframe TEXT NOT NULL,
                        observer_signal TEXT NOT NULL,
                        observer_side_hint TEXT NOT NULL,
                        observer_signal_reason TEXT NOT NULL,
                        reference_price REAL,
                        stop_distance REAL,
                        target_distance REAL,
                        features_json TEXT NOT NULL,
                        runtime_revision TEXT NOT NULL,
                        config_fingerprint TEXT NOT NULL
                    );
                    """
                )
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS runtime_decision_events (
                        decision_event_id TEXT PRIMARY KEY,
                        opportunity_id TEXT NOT NULL,
                        observed_at_utc TEXT NOT NULL,
                        evaluation_tick_utc TEXT NOT NULL,
                        runtime_signal TEXT NOT NULL,
                        runtime_signal_reason TEXT NOT NULL,
                        market_active INTEGER NOT NULL,
                        final_outcome TEXT NOT NULL,
                        gate_stage TEXT NOT NULL,
                        gate_reason TEXT NOT NULL,
                        session_mode TEXT NOT NULL,
                        session_name TEXT NOT NULL,
                        spread_pips REAL,
                        FOREIGN KEY(opportunity_id) REFERENCES learning_opportunities(opportunity_id)
                    );
                    """
                )
                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_learning_opportunities_bar
                    ON learning_opportunities (completed_bar_open_utc, instrument);
                    """
                )
                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_runtime_decision_events_opportunity
                    ON runtime_decision_events (opportunity_id, evaluation_tick_utc);
                    """
                )
                for table in ("learning_opportunities", "runtime_decision_events"):
                    connection.execute(
                        f"""
                        CREATE TRIGGER IF NOT EXISTS {table}_no_update
                        BEFORE UPDATE ON {table}
                        BEGIN
                            SELECT RAISE(ABORT, '{table} is append-only');
                        END;
                        """
                    )
                    connection.execute(
                        f"""
                        CREATE TRIGGER IF NOT EXISTS {table}_no_delete
                        BEFORE DELETE ON {table}
                        BEGIN
                            SELECT RAISE(ABORT, '{table} is append-only');
                        END;
                        """
                    )
            self._schema_ready = True

    def append(self, observation: DecisionObservation) -> bool:
        """Atomically insert one opportunity and its runtime decision event."""

        self._ensure_schema()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                opportunity_cursor = connection.execute(
                    """
                    INSERT INTO learning_opportunities (
                        opportunity_id, schema_version, completed_bar_open_utc,
                        completed_bar_close_utc, instrument, timeframe,
                        observer_signal, observer_side_hint, observer_signal_reason,
                        reference_price, stop_distance, target_distance,
                        features_json, runtime_revision, config_fingerprint
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(opportunity_id) DO NOTHING;
                    """,
                    observation.opportunity_row(),
                )
                if opportunity_cursor.rowcount == 0:
                    existing = connection.execute(
                        """
                        SELECT opportunity_id, schema_version,
                               completed_bar_open_utc, completed_bar_close_utc,
                               instrument, timeframe, observer_signal,
                               observer_side_hint, observer_signal_reason,
                               reference_price, stop_distance, target_distance,
                               features_json, runtime_revision, config_fingerprint
                        FROM learning_opportunities
                        WHERE opportunity_id = ?
                        """,
                        (observation.opportunity_id,),
                    ).fetchone()
                    if existing is None or tuple(existing) != observation.opportunity_row():
                        raise sqlite3.IntegrityError(
                            "immutable learning opportunity payload conflict"
                        )
                event_cursor = connection.execute(
                    """
                    INSERT INTO runtime_decision_events (
                        decision_event_id, opportunity_id, observed_at_utc,
                        evaluation_tick_utc, runtime_signal, runtime_signal_reason,
                        market_active, final_outcome, gate_stage, gate_reason,
                        session_mode, session_name, spread_pips
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(decision_event_id) DO NOTHING;
                    """,
                    observation.event_row(),
                )
            except Exception:
                connection.execute("ROLLBACK")
                raise
            else:
                connection.execute("COMMIT")
                return event_cursor.rowcount == 1


class DecisionObservationSink:
    """Best-effort non-blocking writer that cannot hold the trading path.

    Accepted records live in a bounded in-memory queue. ``flush`` supports tests
    and clean shutdown, but an abrupt process/container loss can still lose
    queued records. Queue drops and write errors are counted for health reporting.
    """

    def __init__(
        self,
        ledger: DecisionObservationLedger,
        *,
        max_queue_size: int = 2048,
        start_worker: bool = True,
    ) -> None:
        self._ledger = ledger
        self._queue: queue.Queue[DecisionObservation] = queue.Queue(
            maxsize=max(1, int(max_queue_size))
        )
        self._start_worker = start_worker
        self._started = False
        self._start_lock = threading.Lock()
        self._warned: set[str] = set()
        self._warn_lock = threading.Lock()
        self._counter_lock = threading.Lock()
        self._counters = {
            "submitted": 0,
            "persisted_events": 0,
            "duplicate_events": 0,
            "queue_drops": 0,
            "write_errors": 0,
        }

    def _increment(self, key: str) -> None:
        with self._counter_lock:
            self._counters[key] += 1

    def counters(self) -> dict[str, int]:
        with self._counter_lock:
            snapshot = dict(self._counters)
        with self._queue.all_tasks_done:
            snapshot["pending_events"] = max(
                0,
                int(self._queue.unfinished_tasks),
            )
        return snapshot

    def _warn_once(self, kind: str, exc: BaseException | None = None) -> None:
        with self._warn_lock:
            if kind in self._warned:
                return
            self._warned.add(kind)
        suffix = f" error={type(exc).__name__}" if exc is not None else ""
        print(f"[LEARNING-OBS][WARN] {kind}{suffix}", flush=True)

    def _ensure_worker(self) -> bool:
        if not self._start_worker:
            return True
        if self._started:
            return True
        with self._start_lock:
            if self._started:
                return True
            try:
                worker = threading.Thread(
                    target=self._run,
                    name="decision-observation-writer",
                    daemon=True,
                )
                worker.start()
                self._started = True
            except Exception as exc:  # pragma: no cover - platform failure
                self._warn_once("writer-start-failed", exc)
                self._increment("queue_drops")
                return False
        return True

    def submit(self, observation: DecisionObservation) -> bool:
        try:
            if not self._ensure_worker():
                return False
            self._queue.put_nowait(observation)
            self._increment("submitted")
            return True
        except queue.Full:
            self._increment("queue_drops")
            self._warn_once("queue-full-observation-dropped")
            return False
        except Exception as exc:  # pragma: no cover - defensive fail-open
            self._increment("queue_drops")
            self._warn_once("queue-submit-failed", exc)
            return False

    def _run(self) -> None:
        while True:
            observation = self._queue.get()
            try:
                inserted = self._ledger.append(observation)
                self._increment("persisted_events" if inserted else "duplicate_events")
            except Exception as exc:
                self._increment("write_errors")
                self._warn_once("database-write-failed", exc)
            finally:
                self._queue.task_done()

    def flush(self, timeout_seconds: float = 2.0) -> bool:
        """Wait until accepted writes finish during tests or clean shutdown."""

        deadline = time.monotonic() + max(0.0, timeout_seconds)
        while time.monotonic() <= deadline:
            if self._queue.unfinished_tasks == 0:
                return True
            time.sleep(0.005)
        return self._queue.unfinished_tasks == 0

    def wait_until_empty(self, timeout_seconds: float = 2.0) -> bool:
        """Backward-compatible alias for ``flush``."""

        return self.flush(timeout_seconds)


__all__ = [
    "DecisionObservation",
    "DecisionObservationDraft",
    "DecisionObservationLedger",
    "DecisionObservationSink",
    "GATE_STAGES",
    "configuration_fingerprint",
]
