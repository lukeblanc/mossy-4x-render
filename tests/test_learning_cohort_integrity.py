from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone

from src import adaptive_policy
from src.adaptive_tuner import AdaptiveTuner
from src.learning_cohort import (
    DEFAULT_LEARNING_COHORT_START_UTC,
    DEFAULT_LEARNING_RUN_TAG,
    FUTURE_TIMESTAMP_TOLERANCE,
    load_clean_outcomes,
)
from src.shadow_learner import run_shadow_analysis


INDICATORS = json.dumps(
    {
        "rsi": 57.0,
        "ema_fast": 1.2,
        "ema_slow": 1.1,
        "ema50": 1.15,
        "ema200": 1.0,
    }
)


def _make_integrity_db(path) -> None:
    with sqlite3.connect(path) as conn:
        # No primary key is intentional: the loader must reject an ambiguous
        # duplicate ID even if a damaged/legacy schema allows one.
        conn.execute(
            """
            CREATE TABLE trades (
                trade_id TEXT,
                timestamp_utc TEXT,
                instrument TEXT,
                side TEXT,
                session_id TEXT,
                indicators_snapshot TEXT,
                exit_timestamp_utc TEXT,
                realized_pnl_ccy REAL,
                broker_confirmed INTEGER,
                run_tag TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE journal_entry_resolutions (
                trade_id TEXT PRIMARY KEY,
                kind TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE trade_events (
                id INTEGER PRIMARY KEY,
                profit REAL,
                reason TEXT
            )
            """
        )

        def add(
            trade_id: str,
            pnl: float | None,
            confirmed: int,
            minute: int,
            *,
            exit_time: bool = True,
        ) -> None:
            conn.execute(
                """
                INSERT INTO trades VALUES (?, '2026-07-13T12:50:00+00:00',
                    'AUD_USD', 'BUY', 'LONDON', ?, ?, ?, ?, 'MINI_RUN')
                """,
                (
                    trade_id,
                    INDICATORS,
                    f"2026-07-13T13:{minute:02d}:00+00:00" if exit_time else None,
                    pnl,
                    confirmed,
                ),
            )

        add("good-win", 1.0, 1, 1)
        add("good-loss", -0.5, 1, 2)
        add("unconfirmed", -100.0, 0, 3)
        add("cancelled", -100.0, 1, 4)
        add("duplicate-alias", 100.0, 1, 5)
        add("ambiguous", 100.0, 1, 6)
        add("ambiguous", 100.0, 1, 7)
        add("still-open", 100.0, 1, 8, exit_time=False)
        add("missing-pnl", None, 1, 9)
        conn.executemany(
            "INSERT INTO trades VALUES (?, ?, 'AUD_USD', 'BUY', 'LONDON', ?, ?, ?, 1, ?)",
            (
                (
                    "missing-entry",
                    None,
                    INDICATORS,
                    "2026-07-13T13:10:00+00:00",
                    100.0,
                    DEFAULT_LEARNING_RUN_TAG,
                ),
                (
                    "malformed-entry",
                    "not-a-timestamp",
                    INDICATORS,
                    "2026-07-13T13:11:00+00:00",
                    100.0,
                    DEFAULT_LEARNING_RUN_TAG,
                ),
                (
                    "exit-before-entry",
                    "2026-07-13T13:30:00+00:00",
                    INDICATORS,
                    "2026-07-13T13:20:00+00:00",
                    100.0,
                    DEFAULT_LEARNING_RUN_TAG,
                ),
                (
                    "nonfinite-pnl",
                    "2026-07-13T12:55:00+00:00",
                    INDICATORS,
                    "2026-07-13T13:12:00+00:00",
                    float("inf"),
                    DEFAULT_LEARNING_RUN_TAG,
                ),
                (
                    "wrong-tag",
                    "2026-07-13T12:56:00+00:00",
                    INDICATORS,
                    "2026-07-13T13:13:00+00:00",
                    100.0,
                    "LEGACY",
                ),
                (
                    "before-cohort",
                    "2026-07-13T12:40:00+00:00",
                    INDICATORS,
                    "2026-07-13T13:14:00+00:00",
                    100.0,
                    DEFAULT_LEARNING_RUN_TAG,
                ),
                (
                    "future-exit",
                    "2026-07-13T12:57:00+00:00",
                    INDICATORS,
                    "2999-01-01T00:00:00+00:00",
                    100.0,
                    DEFAULT_LEARNING_RUN_TAG,
                ),
            ),
        )
        conn.executemany(
            "INSERT INTO journal_entry_resolutions VALUES (?, ?)",
            (
                ("cancelled", "CANCELLED_ORDER"),
                ("duplicate-alias", "DUPLICATE_ALIAS"),
            ),
        )
        conn.execute("INSERT INTO trade_events VALUES (1, 9999.0, 'TP')")


def test_all_learners_share_only_verified_noninvalidated_outcomes(tmp_path, monkeypatch):
    db_path = tmp_path / "journal.db"
    _make_integrity_db(db_path)
    monkeypatch.setenv("SHADOW_LEARNING_ENABLED", "false")
    monkeypatch.setenv("ADAPTIVE_POLICY_MIN_EXACT", "3")
    monkeypatch.setenv("ADAPTIVE_POLICY_CACHE_SECONDS", "1")

    outcomes = load_clean_outcomes(
        db_path,
        as_of_utc=datetime(2026, 7, 13, 14, 0, tzinfo=timezone.utc),
    )
    assert [outcome.trade_id for outcome in outcomes] == ["good-win", "good-loss"]
    assert {outcome.run_tag for outcome in outcomes} == {DEFAULT_LEARNING_RUN_TAG}
    assert all(
        outcome.entry_timestamp_utc >= DEFAULT_LEARNING_COHORT_START_UTC
        for outcome in outcomes
    )

    snapshot = AdaptiveTuner(db_path, lookback=20, min_sample=3).snapshot()
    assert snapshot.lifetime_closed_trades == 2
    assert snapshot.session_closed_trades == 2
    assert snapshot.wins == 1
    assert snapshot.losses == 1
    assert snapshot.risk_multiplier <= 1.0
    assert snapshot.source == "trades"

    adaptive_policy.clear_policy_caches()
    now = datetime(2026, 7, 13, 14, 0, tzinfo=timezone.utc)
    adaptive_policy.publish_market_context(
        "AUD_USD",
        json.loads(INDICATORS),
        now,
        side="BUY",
        session="LONDON",
    )
    decision = adaptive_policy.evaluate_instrument_policy(
        "AUD_USD", db_path=db_path, now_utc=now
    )
    assert decision.exact_samples == 2
    assert decision.risk_scale <= 1.0
    assert decision.blocked is False

    report = run_shadow_analysis(
        db_path,
        output_path=tmp_path / "shadow.json",
        cohort_start_utc="2026-07-13T12:47:00+00:00",
    )
    assert report.total_clean_trades == 2
    assert report.auto_apply is False


def test_database_failures_are_neutral_and_never_interrupt(tmp_path, monkeypatch):
    db_path = tmp_path / "corrupt.db"
    db_path.write_bytes(b"this is not a sqlite database")
    monkeypatch.setenv("SHADOW_LEARNING_ENABLED", "false")

    assert load_clean_outcomes(db_path) == []

    snapshot = AdaptiveTuner(db_path, lookback=20, min_sample=3).snapshot()
    assert snapshot.session_closed_trades == 0
    assert 0.0 <= snapshot.risk_multiplier <= 1.0

    adaptive_policy.clear_policy_caches()
    now = datetime(2026, 7, 13, 14, 0, tzinfo=timezone.utc)
    adaptive_policy.publish_market_context(
        "AUD_USD",
        json.loads(INDICATORS),
        now,
        side="BUY",
        session="LONDON",
    )
    decision = adaptive_policy.evaluate_instrument_policy(
        "AUD_USD", db_path=db_path, now_utc=now
    )
    assert decision.risk_scale == 1.0
    assert decision.blocked is False

    report = run_shadow_analysis(
        db_path,
        output_path=tmp_path / "shadow.json",
    )
    assert report.total_clean_trades == 0
    assert report.recommendation is None
    assert report.auto_apply is False


def test_loader_enforces_as_of_with_only_small_clock_tolerance(tmp_path):
    db_path = tmp_path / "as_of.db"
    as_of = datetime(2026, 7, 14, 12, 0, tzinfo=timezone.utc)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE trades (
                trade_id TEXT,
                timestamp_utc TEXT,
                instrument TEXT,
                side TEXT,
                session_id TEXT,
                indicators_snapshot TEXT,
                exit_timestamp_utc TEXT,
                realized_pnl_ccy REAL,
                broker_confirmed INTEGER,
                run_tag TEXT
            )
            """
        )
        rows = []
        for trade_id, exit_at in (
            ("known", as_of),
            ("within-tolerance", as_of + FUTURE_TIMESTAMP_TOLERANCE),
            (
                "beyond-tolerance",
                as_of + FUTURE_TIMESTAMP_TOLERANCE + timedelta(microseconds=1),
            ),
        ):
            rows.append(
                (
                    trade_id,
                    "2026-07-14T11:00:00+00:00",
                    "AUD_USD",
                    "BUY",
                    "LONDON",
                    INDICATORS,
                    exit_at.isoformat(),
                    1.0,
                    1,
                    DEFAULT_LEARNING_RUN_TAG,
                )
            )
        conn.executemany("INSERT INTO trades VALUES (?,?,?,?,?,?,?,?,?,?)", rows)

    outcomes = load_clean_outcomes(db_path, as_of_utc=as_of)

    assert [outcome.trade_id for outcome in outcomes] == [
        "known",
        "within-tolerance",
    ]
    assert load_clean_outcomes(db_path, as_of_utc="not-a-timestamp") == []


def test_loader_requires_entry_timestamp_and_run_tag_columns(tmp_path):
    db_path = tmp_path / "legacy.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE trades (
                trade_id TEXT,
                exit_timestamp_utc TEXT,
                realized_pnl_ccy REAL,
                broker_confirmed INTEGER,
                run_tag TEXT
            )
            """
        )
        conn.execute(
            "INSERT INTO trades VALUES ('legacy', '2026-07-14T12:00:00+00:00', 1.0, 1, 'MINI_RUN')"
        )

    assert load_clean_outcomes(db_path) == []


def test_custom_window_excludes_cross_window_trade_for_every_learner(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "journal.db"
    _make_integrity_db(db_path)
    window_start = "2026-07-13T13:00:00+00:00"
    now = datetime(2026, 7, 13, 14, 0, tzinfo=timezone.utc)
    monkeypatch.setenv("SHADOW_LEARNING_ENABLED", "false")

    # Both otherwise-clean rows enter at 12:50 and close after 13:00. A shared
    # prospective cohort must exclude them based on entry time everywhere.
    assert load_clean_outcomes(
        db_path,
        entry_start_utc=window_start,
        as_of_utc=now,
    ) == []

    tuner = AdaptiveTuner(
        db_path,
        lookback=20,
        min_sample=3,
        run_tag=DEFAULT_LEARNING_RUN_TAG,
        window_start_utc=window_start,
    )
    assert tuner.snapshot().session_closed_trades == 0

    adaptive_policy.clear_policy_caches()
    monkeypatch.setenv("ADAPTIVE_RUN_TAG", DEFAULT_LEARNING_RUN_TAG)
    monkeypatch.setenv("ADAPTIVE_WINDOW_START_UTC", window_start)
    adaptive_policy.publish_market_context(
        "AUD_USD",
        json.loads(INDICATORS),
        now,
        side="BUY",
        session="LONDON",
    )
    policy = adaptive_policy.evaluate_instrument_policy(
        "AUD_USD", db_path=db_path, now_utc=now
    )
    assert policy.exact_samples == 0

    shadow = run_shadow_analysis(
        db_path,
        output_path=tmp_path / "cross-window-shadow.json",
        cohort_start_utc=window_start,
        cohort_run_tag=DEFAULT_LEARNING_RUN_TAG,
    )
    assert shadow.total_clean_trades == 0


def test_tuner_can_never_scale_above_base_risk(tmp_path, monkeypatch):
    db_path = tmp_path / "journal.db"
    _make_integrity_db(db_path)
    monkeypatch.setenv("SHADOW_LEARNING_ENABLED", "false")
    with sqlite3.connect(db_path) as conn:
        for index in range(10):
            conn.execute(
                """
                INSERT INTO trades VALUES (?, '2026-07-13T13:00:00+00:00',
                    'AUD_USD', 'BUY', 'LONDON', ?, ?, 5.0, 1, 'MINI_RUN')
                """,
                (
                    f"positive-{index}",
                    INDICATORS,
                    f"2026-07-13T14:{index:02d}:00+00:00",
                ),
            )

    snapshot = AdaptiveTuner(db_path, lookback=40, min_sample=3).snapshot()
    assert snapshot.risk_multiplier == 1.0
