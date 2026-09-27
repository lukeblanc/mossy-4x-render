from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

import src.learning_review as learning_review
from src.learning_review import (
    build_learning_review,
    publish_learning_review,
    validate_learning_review,
)
from src.trade_journal import TradeJournal


@pytest.fixture
def anyio_backend():
    return "asyncio"


def _closed_trade(
    journal: TradeJournal,
    trade_id: str,
    *,
    instrument: str,
    pnl: float,
    exit_at: datetime,
    broker_confirmed: bool = True,
    run_tag: str = "MINI_RUN",
) -> None:
    journal.record_entry(
        trade_id=trade_id,
        timestamp_utc=exit_at - timedelta(minutes=15),
        instrument=instrument,
        side="BUY",
        units=100,
        entry_price=1.0,
        stop_loss_price=0.99,
        take_profit_price=1.01,
        spread_at_entry=0.0001,
        session_id="LONDON",
        session_mode="STRICT",
        run_tag=run_tag,
        gating_flags={},
        indicators_snapshot={},
    )
    journal.record_exit(
        trade_id=trade_id,
        exit_timestamp_utc=exit_at,
        exit_price=1.01,
        spread_at_exit=0.0001,
        max_profit_ccy=max(0.0, pnl),
        realized_pnl_ccy=pnl,
        exit_reason="BROKER_CLOSE",
        duration_seconds=900,
        broker_confirmed=broker_confirmed,
    )


def _metrics(
    *,
    trades: int,
    expectancy: float,
    profit_factor: float,
    drawdown: float,
    net_profit: float | None = None,
    wins: int | None = None,
    losses: int | None = None,
) -> dict:
    selected_wins = wins if wins is not None else trades // 2
    selected_losses = losses if losses is not None else trades - selected_wins
    return {
        "trades": trades,
        "wins": selected_wins,
        "losses": selected_losses,
        "win_rate": selected_wins / trades if trades else 0.0,
        "net_profit": (
            net_profit if net_profit is not None else expectancy * trades
        ),
        "expectancy": expectancy,
        "profit_factor": profit_factor,
        "max_drawdown": drawdown,
    }


def _shadow_report(
    now: datetime,
    candidates: list[dict] | None = None,
    *,
    evidence_revision: str = "abc1234",
) -> dict:
    baseline_validation = _metrics(
        trades=40,
        expectancy=0.10,
        profit_factor=1.20,
        drawdown=10.0,
    )
    return {
        "generated_utc": now.isoformat(),
        "evidence_revision": evidence_revision,
        "cohort_start_utc": "2026-07-13T12:47:00+00:00",
        "cohort_run_tag": "MINI_RUN",
        "instruments": ["AUD_USD", "GBP_USD"],
        "total_clean_trades": 140,
        "train_trades": 100,
        "validation_trades": 40,
        "baseline": {
            "name": "baseline",
            "validation": baseline_validation,
        },
        "candidates": candidates
        if candidates is not None
        else [
            {
                "name": "trend_full_only",
                "train": _metrics(
                    trades=60,
                    expectancy=0.15,
                    profit_factor=1.30,
                    drawdown=8.0,
                ),
                "validation": _metrics(
                    trades=32,
                    expectancy=0.20,
                    profit_factor=1.30,
                    drawdown=8.0,
                ),
                "validation_coverage": 0.8,
            }
        ],
        "recommendation": "trend_full_only",
        "auto_apply": True,
    }


def _write_report(path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _observation_snapshot(**overrides) -> dict:
    snapshot = {
        "submitted": 12,
        "persisted_events": 10,
        "duplicate_events": 1,
        "queue_drops": 0,
        "write_errors": 0,
        "pending_events": 1,
    }
    snapshot.update(overrides)
    return snapshot


def test_pair_metrics_use_only_latest_40_clean_broker_confirmed_rows(tmp_path):
    db_path = tmp_path / "trade_journal.db"
    journal = TradeJournal(db_path)
    start = datetime(2026, 9, 1, tzinfo=timezone.utc)
    # Five old wins must fall outside the bounded window. The retained 40 are
    # 20 +2 wins and 20 -1 losses: net +20, PF 2, expectancy 0.5.
    for index in range(45):
        pnl = 50.0 if index < 5 else (2.0 if index % 2 else -1.0)
        _closed_trade(
            journal,
            f"AUD-{index}",
            instrument="AUD_USD",
            pnl=pnl,
            exit_at=start + timedelta(minutes=index),
        )
    _closed_trade(
        journal,
        "GBP-clean",
        instrument="GBP_USD",
        pnl=-3.0,
        exit_at=start + timedelta(hours=2),
    )
    _closed_trade(
        journal,
        "GBP-unconfirmed",
        instrument="GBP_USD",
        pnl=999.0,
        exit_at=start + timedelta(hours=3),
        broker_confirmed=False,
    )
    _closed_trade(
        journal,
        "GBP-invalidated",
        instrument="GBP_USD",
        pnl=999.0,
        exit_at=start + timedelta(hours=4),
    )
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO journal_entry_resolutions
                (trade_id, kind, broker_id, resolved_at, run_id,
                 original_row_json, evidence_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "GBP-invalidated",
                "DUPLICATE_ALIAS",
                "broker-redacted",
                start.isoformat(),
                "cleanup",
                "{}",
                "{}",
            ),
        )

    review = build_learning_review(
        db_path,
        shadow_report_path=tmp_path / "missing-report.json",
        adaptive_snapshot={"session_closed_trades": 40, "risk_multiplier": 0.8},
        source_revision="abc1234",
        observed_at=start + timedelta(days=1),
    )

    aud, gbp = review["pair_performance"]
    assert aud == {
        "instrument": "AUD_USD",
        "window": "last_40_broker_confirmed_closed",
        "sample_count": 40,
        "wins": 20,
        "losses": 20,
        "flat": 0,
        "win_rate": 0.5,
        "closed_net_pnl_aud": 20.0,
        "expectancy_aud": 0.5,
        "profit_factor": 2.0,
        "profit_factor_state": "finite",
        "max_drawdown_aud": 1.0,
        "evidence_status": "sufficient",
    }
    assert gbp["sample_count"] == 1
    assert gbp["closed_net_pnl_aud"] == -3.0
    assert gbp["evidence_status"] == "small_sample"
    assert "GBP-unconfirmed" not in json.dumps(review)
    assert "broker-redacted" not in json.dumps(review)


def test_pair_metrics_stop_at_review_observation_time(tmp_path):
    db_path = tmp_path / "trade_journal.db"
    journal = TradeJournal(db_path)
    observed = datetime(2026, 9, 10, 12, 0, tzinfo=timezone.utc)
    _closed_trade(
        journal,
        "known",
        instrument="AUD_USD",
        pnl=1.0,
        exit_at=observed - timedelta(minutes=1),
    )
    _closed_trade(
        journal,
        "not-yet-known",
        instrument="AUD_USD",
        pnl=999.0,
        exit_at=observed + timedelta(minutes=5),
    )

    review = build_learning_review(
        db_path,
        shadow_report_path=tmp_path / "missing.json",
        source_revision="abc1234",
        observed_at=observed,
    )

    aud = review["pair_performance"][0]
    assert aud["sample_count"] == 1
    assert aud["closed_net_pnl_aud"] == 1.0


def test_pair_metrics_share_adaptive_run_tag_and_entry_cohort(tmp_path):
    db_path = tmp_path / "trade_journal.db"
    journal = TradeJournal(db_path)
    cohort_start = datetime(2026, 9, 10, 12, 0, tzinfo=timezone.utc)
    observed = cohort_start + timedelta(hours=1)
    _closed_trade(
        journal,
        "prior-regime-entry",
        instrument="AUD_USD",
        pnl=999.0,
        exit_at=cohort_start + timedelta(minutes=2),
        run_tag="CUSTOM_RUN",
    )
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE trades SET timestamp_utc=? WHERE trade_id=?",
            ((cohort_start - timedelta(minutes=1)).isoformat(), "prior-regime-entry"),
        )
    _closed_trade(
        journal,
        "wrong-run",
        instrument="AUD_USD",
        pnl=999.0,
        exit_at=cohort_start + timedelta(minutes=20),
        run_tag="MINI_RUN",
    )
    _closed_trade(
        journal,
        "shared-cohort",
        instrument="AUD_USD",
        pnl=2.0,
        exit_at=cohort_start + timedelta(minutes=30),
        run_tag="CUSTOM_RUN",
    )

    review = build_learning_review(
        db_path,
        shadow_report_path=tmp_path / "missing.json",
        adaptive_snapshot={
            "filter_run_tag": "CUSTOM_RUN",
            "filter_window_start_utc": cohort_start.isoformat(),
            "session_closed_trades": 1,
            "risk_multiplier": 0.85,
        },
        source_revision="abc1234",
        observed_at=observed,
    )

    aud = review["pair_performance"][0]
    assert aud["sample_count"] == 1
    assert aud["closed_net_pnl_aud"] == 2.0


def test_journal_missing_learning_chronology_column_is_unavailable(tmp_path):
    db_path = tmp_path / "trade_journal.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE trades (
                trade_id TEXT PRIMARY KEY,
                instrument TEXT,
                exit_timestamp_utc TEXT,
                realized_pnl_ccy REAL,
                broker_confirmed INTEGER,
                run_tag TEXT
            )
            """
        )

    review = build_learning_review(
        db_path,
        shadow_report_path=tmp_path / "missing.json",
        source_revision="abc1234",
    )

    assert review["challenger_review"]["readiness"]["reason_codes"] == [
        "journal_unavailable"
    ]
    assert all(item["sample_count"] == 0 for item in review["pair_performance"])


def test_executed_trade_subset_evidence_cannot_request_luke_review(tmp_path):
    db_path = tmp_path / "trade_journal.db"
    TradeJournal(db_path)
    now = datetime(2026, 9, 27, 6, 0, tzinfo=timezone.utc)
    report_path = tmp_path / "shadow_learning_report.json"
    _write_report(
        report_path,
        _shadow_report(now, evidence_revision="b304a03d"),
    )

    review = build_learning_review(
        db_path,
        shadow_report_path=report_path,
        adaptive_snapshot={
            "session_closed_trades": 80,
            "risk_multiplier": 0.75,
            "loss_streak": 2,
        },
        observation_snapshot=_observation_snapshot(),
        source_revision="b304a03d",
        observed_at=now + timedelta(minutes=5),
    )

    candidate = review["challenger_review"]["candidates"][0]
    readiness = review["challenger_review"]["readiness"]
    assert candidate["status"] == "collecting"
    assert set(candidate["passed_gates"]) == {
        "sufficient_train_sample",
        "sufficient_validation_sample",
        "sufficient_validation_coverage",
        "training_expectancy_positive",
        "training_profit_factor_at_least_one",
        "validation_expectancy_gate_passed",
        "validation_profit_factor_gate_passed",
        "validation_drawdown_gate_passed",
    }
    assert candidate["failed_gates"] == [
        "prospective_validation_required",
        "forward_shadow_required",
        "promotion_protocol_not_implemented",
    ]
    assert readiness == {
        "state": "collecting_data",
        "leading_candidate_id": None,
        "reason_codes": [
            "prospective_validation_required",
            "forward_shadow_required",
            "promotion_protocol_not_implemented",
        ],
        "next_allowed_action": "collect_more_evidence",
        "luke_approval_required": True,
        "auto_promotion_permitted": False,
    }
    assert review["learning_status"]["shadow_learning"]["auto_apply"] is False
    assert review["learning_status"]["shadow_learning"]["state"] == "collecting"
    assert review["learning_status"]["overall"] == "collecting"
    assert review["governance"] == {
        "read_only": True,
        "demo_only": True,
        "can_trade": False,
        "can_write_configuration": False,
        "can_run_optimisation": False,
        "can_deploy": False,
        "can_promote": False,
    }
    assert validate_learning_review(review) is True


def test_prospective_evidence_needs_all_statistical_and_governance_gates(tmp_path):
    db_path = tmp_path / "trade_journal.db"
    TradeJournal(db_path)
    now = datetime(2026, 9, 27, 6, 0, tzinfo=timezone.utc)
    report = _shadow_report(now, evidence_revision="b304a03d")
    report.update(
        {
            "evidence_scope": "prospective_shadow_observations",
            "total_clean_trades": 300,
            "train_trades": 200,
            "validation_trades": 100,
            "validation_protocol": {
                "immutable_registration": True,
                "post_registration_evidence": True,
                "frozen_referee": True,
                "false_discovery_controlled": True,
                "false_discovery_alpha": 0.05,
                "validation_pair_count": 3,
                "forward_shadow_passed": True,
            },
        }
    )
    report["baseline"]["validation"] = _metrics(
        trades=100,
        expectancy=0.10,
        profit_factor=1.20,
        drawdown=10.0,
    )
    report["candidates"][0]["train"] = _metrics(
        trades=150,
        expectancy=0.15,
        profit_factor=1.30,
        drawdown=8.0,
    )
    report["candidates"][0]["validation"] = _metrics(
        trades=100,
        expectancy=0.20,
        profit_factor=1.30,
        drawdown=8.0,
    )
    report["candidates"][0]["validation_coverage"] = 1.0
    report_path = tmp_path / "prospective.json"
    _write_report(report_path, report)

    review = build_learning_review(
        db_path,
        shadow_report_path=report_path,
        adaptive_snapshot={"session_closed_trades": 80, "risk_multiplier": 0.8},
        observation_snapshot=_observation_snapshot(),
        source_revision="b304a03d",
        observed_at=now,
    )

    candidate = review["challenger_review"]["candidates"][0]
    readiness = review["challenger_review"]["readiness"]
    assert candidate["status"] == "collecting"
    assert set(candidate["passed_gates"]) == learning_review.PASSED_GATE_CODES
    assert candidate["failed_gates"] == ["promotion_protocol_not_implemented"]
    assert readiness["state"] == "collecting_data"
    assert readiness["leading_candidate_id"] is None
    assert readiness["reason_codes"] == ["promotion_protocol_not_implemented"]
    assert readiness["next_allowed_action"] == "collect_more_evidence"
    assert readiness["luke_approval_required"] is True
    assert readiness["auto_promotion_permitted"] is False
    assert review["learning_status"]["shadow_learning"]["state"] == "collecting"
    assert review["learning_status"]["overall"] == "collecting"
    assert validate_learning_review(review) is True


@pytest.mark.parametrize("evidence_revision", [None, "unknown", "deadbee"])
def test_missing_unknown_or_mismatched_evidence_revision_fails_closed(
    tmp_path, evidence_revision
):
    db_path = tmp_path / "trade_journal.db"
    TradeJournal(db_path)
    now = datetime(2026, 9, 27, 6, 0, tzinfo=timezone.utc)
    report = _shadow_report(now)
    if evidence_revision is None:
        report.pop("evidence_revision")
    else:
        report["evidence_revision"] = evidence_revision
    report_path = tmp_path / "shadow.json"
    _write_report(report_path, report)

    review = build_learning_review(
        db_path,
        shadow_report_path=report_path,
        adaptive_snapshot={"session_closed_trades": 80, "risk_multiplier": 0.8},
        source_revision="abc1234",
        observed_at=now,
    )

    assert review["challenger_review"]["evidence_revision"] == (
        "unknown" if evidence_revision in {None, "unknown"} else "deadbee"
    )
    assert review["learning_status"]["overall"] == "degraded"
    assert review["learning_status"]["shadow_learning"]["state"] == "error"
    readiness = review["challenger_review"]["readiness"]
    assert readiness["state"] == "unavailable"
    assert readiness["reason_codes"] == ["revision_mismatch"]
    assert readiness["next_allowed_action"] == "none"
    assert validate_learning_review(review) is True


@pytest.mark.parametrize("cohort_run_tag", [None, "LEGACY"])
def test_missing_or_wrong_shadow_run_tag_fails_closed(tmp_path, cohort_run_tag):
    db_path = tmp_path / "trade_journal.db"
    TradeJournal(db_path)
    now = datetime(2026, 9, 27, 6, 0, tzinfo=timezone.utc)
    report = _shadow_report(now)
    if cohort_run_tag is None:
        report.pop("cohort_run_tag")
    else:
        report["cohort_run_tag"] = cohort_run_tag
    report_path = tmp_path / "shadow.json"
    _write_report(report_path, report)

    review = build_learning_review(
        db_path,
        shadow_report_path=report_path,
        adaptive_snapshot={"session_closed_trades": 80, "risk_multiplier": 0.8},
        source_revision="abc1234",
        observed_at=now,
    )

    assert review["learning_status"]["shadow_learning"]["state"] == "error"
    readiness = review["challenger_review"]["readiness"]
    assert readiness["state"] == "unavailable"
    assert readiness["reason_codes"] == ["report_invalid"]
    assert readiness["next_allowed_action"] == "none"


def test_shadow_lineage_follows_adaptive_snapshot_not_separate_shadow_env(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "trade_journal.db"
    TradeJournal(db_path)
    now = datetime(2026, 9, 27, 6, 0, tzinfo=timezone.utc)
    cohort_start = datetime(2026, 9, 1, 0, 0, tzinfo=timezone.utc)
    report = _shadow_report(now)
    report["cohort_run_tag"] = "CUSTOM_RUN"
    report["cohort_start_utc"] = cohort_start.isoformat()
    report_path = tmp_path / "shadow.json"
    _write_report(report_path, report)
    monkeypatch.setenv("SHADOW_RUN_TAG", "WRONG_AND_IGNORED")

    review = build_learning_review(
        db_path,
        shadow_report_path=report_path,
        adaptive_snapshot={
            "filter_run_tag": "CUSTOM_RUN",
            "filter_window_start_utc": cohort_start.isoformat(),
            "session_closed_trades": 80,
            "risk_multiplier": 0.8,
        },
        observation_snapshot=_observation_snapshot(),
        source_revision="abc1234",
        observed_at=now,
    )

    readiness = review["challenger_review"]["readiness"]
    assert readiness["state"] == "collecting_data"
    assert "report_invalid" not in readiness["reason_codes"]


@pytest.mark.parametrize(
    "cohort_start",
    [None, "not-a-timestamp", "2026-07-14T12:47:00+00:00"],
)
def test_missing_invalid_or_mismatched_shadow_cohort_start_fails_closed(
    tmp_path, cohort_start
):
    db_path = tmp_path / "trade_journal.db"
    TradeJournal(db_path)
    now = datetime(2026, 9, 27, 6, 0, tzinfo=timezone.utc)
    report = _shadow_report(now)
    if cohort_start is None:
        report.pop("cohort_start_utc")
    else:
        report["cohort_start_utc"] = cohort_start
    report_path = tmp_path / "shadow.json"
    _write_report(report_path, report)

    review = build_learning_review(
        db_path,
        shadow_report_path=report_path,
        adaptive_snapshot={"session_closed_trades": 80, "risk_multiplier": 0.8},
        source_revision="abc1234",
        observed_at=now,
    )

    assert review["learning_status"]["shadow_learning"]["state"] == "error"
    assert review["challenger_review"]["readiness"]["reason_codes"] == [
        "report_invalid"
    ]


@pytest.mark.parametrize(
    "instruments",
    [None, ["GBP_USD", "AUD_USD"], ["AUD_USD"], ["AUD_USD", "XAU_USD"]],
)
def test_missing_or_wrong_shadow_instrument_lineage_fails_closed(
    tmp_path, instruments
):
    db_path = tmp_path / "trade_journal.db"
    TradeJournal(db_path)
    now = datetime(2026, 9, 27, 6, 0, tzinfo=timezone.utc)
    report = _shadow_report(now)
    if instruments is None:
        report.pop("instruments")
    else:
        report["instruments"] = instruments
    report_path = tmp_path / "shadow.json"
    _write_report(report_path, report)

    review = build_learning_review(
        db_path,
        shadow_report_path=report_path,
        adaptive_snapshot={"session_closed_trades": 80, "risk_multiplier": 0.8},
        source_revision="abc1234",
        observed_at=now,
    )

    assert review["learning_status"]["shadow_learning"]["state"] == "error"
    assert review["challenger_review"]["readiness"]["reason_codes"] == [
        "report_invalid"
    ]


def test_small_candidate_sample_reports_collecting_with_fixed_reasons(tmp_path):
    db_path = tmp_path / "trade_journal.db"
    TradeJournal(db_path)
    now = datetime(2026, 9, 27, 6, 0, tzinfo=timezone.utc)
    candidate = {
        "name": "aud_only",
        "train": _metrics(
            trades=20, expectancy=0.2, profit_factor=1.4, drawdown=2.0
        ),
        "validation": _metrics(
            trades=10, expectancy=0.3, profit_factor=1.5, drawdown=2.0
        ),
        "validation_coverage": 0.25,
    }
    report_path = tmp_path / "shadow.json"
    _write_report(report_path, _shadow_report(now, [candidate]))

    review = build_learning_review(
        db_path,
        shadow_report_path=report_path,
        adaptive_snapshot={"session_closed_trades": 5, "risk_multiplier": 0.85},
        source_revision="abc1234",
        observed_at=now,
    )

    result = review["challenger_review"]["candidates"][0]
    assert result["status"] == "collecting"
    assert {
        "insufficient_train_sample",
        "insufficient_validation_sample",
        "insufficient_validation_coverage",
    }.issubset(result["failed_gates"])
    assert review["challenger_review"]["readiness"]["state"] == "collecting_data"
    assert review["learning_status"]["adaptive_tuner"]["state"] == "collecting"
    assert review["learning_status"]["adaptive_tuner"]["reason_code"] == "small_sample"


def test_adaptive_negative_expectancy_reason_is_preserved(tmp_path):
    db_path = tmp_path / "trade_journal.db"
    TradeJournal(db_path)

    review = build_learning_review(
        db_path,
        shadow_report_path=tmp_path / "missing.json",
        adaptive_snapshot={
            "session_closed_trades": 40,
            "risk_multiplier": 0.65,
            "reason_code": "negative_expectancy",
        },
        observation_snapshot=_observation_snapshot(),
        source_revision="abc1234",
    )

    adaptive = review["learning_status"]["adaptive_tuner"]
    assert adaptive["state"] == "active"
    assert adaptive["sample_count"] == 40
    assert adaptive["risk_multiplier"] == 0.65
    assert adaptive["reason_code"] == "negative_expectancy"


def test_adaptive_low_win_rate_reason_is_preserved(tmp_path):
    db_path = tmp_path / "trade_journal.db"
    TradeJournal(db_path)

    review = build_learning_review(
        db_path,
        shadow_report_path=tmp_path / "missing.json",
        adaptive_snapshot={
            "session_closed_trades": 40,
            "risk_multiplier": 0.8,
            "reason_code": "low_win_rate",
        },
        observation_snapshot=_observation_snapshot(),
        source_revision="abc1234",
    )

    adaptive = review["learning_status"]["adaptive_tuner"]
    assert adaptive["state"] == "active"
    assert adaptive["risk_multiplier"] == 0.8
    assert adaptive["reason_code"] == "low_win_rate"


def test_observation_capture_is_strict_aggregate_and_active(tmp_path):
    db_path = tmp_path / "trade_journal.db"
    TradeJournal(db_path)

    review = build_learning_review(
        db_path,
        observation_snapshot=_observation_snapshot(),
        source_revision="abc1234",
    )

    assert review["learning_status"]["observation_capture"] == {
        "state": "active",
        "mode": "best_effort",
        "submitted": 12,
        "persisted_events": 10,
        "duplicate_events": 1,
        "queue_drops": 0,
        "write_errors": 0,
        "pending_events": 1,
    }
    assert validate_learning_review(review) is True


@pytest.mark.parametrize(
    "fault",
    [
        {"queue_drops": 2},
        {"write_errors": 1},
    ],
)
def test_observation_capture_faults_force_overall_degraded(tmp_path, fault):
    db_path = tmp_path / "trade_journal.db"
    TradeJournal(db_path)

    review = build_learning_review(
        db_path,
        observation_snapshot=_observation_snapshot(**fault),
        source_revision="abc1234",
    )

    capture = review["learning_status"]["observation_capture"]
    assert capture["state"] == "degraded"
    assert review["learning_status"]["overall"] == "degraded"
    assert validate_learning_review(review) is True


@pytest.mark.parametrize(
    "snapshot",
    [
        None,
        {"submitted": -1},
        {**_observation_snapshot(), "raw_rows": [{"trade_id": "forbidden"}]},
    ],
)
def test_missing_or_invalid_observation_capture_fails_closed(tmp_path, snapshot):
    db_path = tmp_path / "trade_journal.db"
    TradeJournal(db_path)

    review = build_learning_review(
        db_path,
        observation_snapshot=snapshot,
        source_revision="abc1234",
    )

    assert review["learning_status"]["observation_capture"] == {
        "state": "degraded",
        "mode": "best_effort",
        "submitted": 0,
        "persisted_events": 0,
        "duplicate_events": 0,
        "queue_drops": 0,
        "write_errors": 0,
        "pending_events": None,
    }
    assert "raw_rows" not in json.dumps(review)
    assert review["learning_status"]["overall"] == "degraded"
    assert validate_learning_review(review) is True


@pytest.mark.parametrize("report_contents", [None, "{broken", "[]"])
def test_missing_or_corrupt_shadow_evidence_is_unavailable(tmp_path, report_contents):
    db_path = tmp_path / "trade_journal.db"
    TradeJournal(db_path)
    report_path = tmp_path / "shadow.json"
    if report_contents is not None:
        report_path.write_text(report_contents, encoding="utf-8")

    review = build_learning_review(
        db_path,
        shadow_report_path=report_path,
        adaptive_snapshot=None,
        source_revision="abc1234",
    )

    assert review["learning_status"]["overall"] == "degraded"
    assert review["learning_status"]["shadow_learning"]["state"] == "error"
    assert review["challenger_review"]["readiness"]["state"] == "unavailable"
    assert review["challenger_review"]["readiness"]["reason_codes"] == [
        "report_invalid"
    ]
    assert validate_learning_review(review) is True


def test_missing_or_corrupt_journal_never_invents_pair_results(tmp_path):
    corrupt = tmp_path / "trade_journal.db"
    corrupt.write_text("not sqlite", encoding="utf-8")
    now = datetime(2026, 9, 27, 6, 0, tzinfo=timezone.utc)
    report_path = tmp_path / "shadow.json"
    _write_report(report_path, _shadow_report(now))

    review = build_learning_review(
        corrupt,
        shadow_report_path=report_path,
        adaptive_snapshot={"session_closed_trades": 80, "risk_multiplier": 0.8},
        source_revision="abc1234",
        observed_at=now,
    )

    assert all(pair["sample_count"] == 0 for pair in review["pair_performance"])
    assert all(
        pair["evidence_status"] == "no_sample"
        for pair in review["pair_performance"]
    )
    readiness = review["challenger_review"]["readiness"]
    assert readiness["state"] == "unavailable"
    assert readiness["reason_codes"] == ["journal_unavailable"]


def test_unregistered_candidate_is_not_relayed(tmp_path):
    db_path = tmp_path / "trade_journal.db"
    TradeJournal(db_path)
    now = datetime(2026, 9, 27, 6, 0, tzinfo=timezone.utc)
    unknown = {
        "name": "turn_every_safety_check_off",
        "train": _metrics(
            trades=60, expectancy=99.0, profit_factor=99.0, drawdown=0.0
        ),
        "validation": _metrics(
            trades=40, expectancy=99.0, profit_factor=99.0, drawdown=0.0
        ),
        "validation_coverage": 1.0,
    }
    report_path = tmp_path / "shadow.json"
    _write_report(report_path, _shadow_report(now, [unknown]))

    review = build_learning_review(
        db_path,
        shadow_report_path=report_path,
        adaptive_snapshot={"session_closed_trades": 80, "risk_multiplier": 0.8},
        source_revision="abc1234",
        observed_at=now,
    )

    serialized = json.dumps(review)
    assert "turn_every_safety_check_off" not in serialized
    assert review["challenger_review"]["candidates"] == []
    assert review["challenger_review"]["readiness"]["state"] == "unavailable"


def test_stale_and_future_shadow_evidence_never_becomes_ready(tmp_path):
    db_path = tmp_path / "trade_journal.db"
    TradeJournal(db_path)
    now = datetime(2026, 9, 27, 12, 0, tzinfo=timezone.utc)
    for generated in (now - timedelta(hours=3), now + timedelta(minutes=10)):
        report_path = tmp_path / f"shadow-{generated.minute}.json"
        _write_report(report_path, _shadow_report(generated))
        review = build_learning_review(
            db_path,
            shadow_report_path=report_path,
            adaptive_snapshot={"session_closed_trades": 80, "risk_multiplier": 0.8},
            source_revision="abc1234",
            observed_at=now,
            shadow_max_age_seconds=7_200,
        )

        assert review["learning_status"]["shadow_learning"]["state"] == "stale"
        assert review["challenger_review"]["readiness"]["state"] == "stale_evidence"
        assert review["challenger_review"]["candidates"][0]["status"] == "stale"


def test_validation_rejects_unknown_or_sensitive_fields(tmp_path):
    db_path = tmp_path / "trade_journal.db"
    TradeJournal(db_path)
    review = build_learning_review(db_path, source_revision="abc1234")
    review["equity"] = 10_000

    assert validate_learning_review(review) is False


@pytest.mark.anyio
async def test_publisher_derives_separate_route_and_sends_only_valid_snapshot(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "trade_journal.db"
    TradeJournal(db_path)
    review = build_learning_review(db_path, source_revision="abc1234")
    monkeypatch.setenv(
        "MOSSY_MCP_STATUS_URL", "https://example.test/internal/runtime-heartbeat"
    )
    monkeypatch.setenv("MOSSY_MCP_STATUS_KEY", "test-key")
    monkeypatch.setattr(learning_review, "_last_publish_monotonic", None)
    monkeypatch.setattr(learning_review, "_last_publish_fingerprint", None)
    seen = {}

    class Response:
        def raise_for_status(self):
            return None

    class Client:
        def __init__(self, **kwargs):
            seen["options"] = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, url, **kwargs):
            seen["url"] = url
            seen["payload"] = kwargs["json"]
            return Response()

    monkeypatch.setattr(learning_review.httpx, "AsyncClient", Client)

    sent, status = await publish_learning_review(review)

    assert (sent, status) == (True, "sent")
    assert seen["url"] == "https://example.test/internal/learning-snapshot"
    assert seen["payload"] is review


@pytest.mark.anyio
async def test_publisher_failure_is_fail_open(tmp_path, monkeypatch):
    db_path = tmp_path / "trade_journal.db"
    TradeJournal(db_path)
    review = build_learning_review(db_path, source_revision="abc1234")
    monkeypatch.setenv(
        "MOSSY_MCP_LEARNING_URL", "https://example.test/internal/learning-snapshot"
    )
    monkeypatch.setenv("MOSSY_MCP_STATUS_KEY", "test-key")
    monkeypatch.setattr(learning_review, "_last_publish_monotonic", None)
    monkeypatch.setattr(learning_review, "_last_publish_fingerprint", None)

    class Client:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, *args, **kwargs):
            raise httpx.ConnectError("offline")

    import httpx

    monkeypatch.setattr(learning_review.httpx, "AsyncClient", Client)

    sent, status = await publish_learning_review(review)

    assert sent is False
    assert status == "http-error:ConnectError"


@pytest.mark.anyio
async def test_publisher_throttle_ignores_timestamp_only_changes(tmp_path, monkeypatch):
    db_path = tmp_path / "trade_journal.db"
    TradeJournal(db_path)
    first = build_learning_review(
        db_path,
        source_revision="b304a03d",
        observed_at=datetime(2026, 9, 27, 6, 0, tzinfo=timezone.utc),
    )
    second = build_learning_review(
        db_path,
        source_revision="b304a03d",
        observed_at=datetime(2026, 9, 27, 6, 1, tzinfo=timezone.utc),
    )
    monkeypatch.setenv(
        "MOSSY_MCP_LEARNING_URL", "https://example.test/internal/learning-snapshot"
    )
    monkeypatch.setenv("MOSSY_MCP_STATUS_KEY", "test-key")
    monkeypatch.setattr(learning_review, "_last_publish_monotonic", None)
    monkeypatch.setattr(learning_review, "_last_publish_fingerprint", None)
    clock = iter((100.0, 101.0))
    monkeypatch.setattr(
        learning_review, "time", SimpleNamespace(monotonic=lambda: next(clock))
    )

    class Response:
        def raise_for_status(self):
            return None

    class Client:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, *args, **kwargs):
            return Response()

    monkeypatch.setattr(learning_review.httpx, "AsyncClient", Client)

    assert await publish_learning_review(first) == (True, "sent")
    assert await publish_learning_review(second) == (False, "throttled")


@pytest.mark.anyio
async def test_publisher_rejects_payload_before_network_egress(monkeypatch):
    monkeypatch.setenv(
        "MOSSY_MCP_LEARNING_URL", "https://example.test/internal/learning-snapshot"
    )
    monkeypatch.setenv("MOSSY_MCP_STATUS_KEY", "test-key")

    sent, status = await publish_learning_review(
        {"schema_version": "mossy.learning-review.v1", "signal": "BUY"}
    )

    assert (sent, status) == (False, "invalid-payload")
