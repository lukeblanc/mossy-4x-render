from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import httpx

from src.learning_cohort import DEFAULT_LEARNING_RUN_TAG, load_clean_outcomes
from src.shadow_learner import CLEAN_COHORT_START_UTC


SCHEMA_VERSION = "mossy.learning-review.v1"
PAIR_WINDOW = "last_40_broker_confirmed_closed"
PAIR_WINDOW_SIZE = 40
REGISTRY_VERSION = "mossy-shadow-registry.v1"
DEFINITION_VERSION = "shadow-filter.v1"
UNKNOWN_EVIDENCE_TIME = "1970-01-01T00:00:00+00:00"

INSTRUMENTS = ("AUD_USD", "GBP_USD")
REGISTERED_CANDIDATES = (
    "trend_full_only",
    "avoid_countertrend",
    "momentum_50",
    "momentum_55_45",
    "london_only",
    "newyork_only",
    "aud_only",
    "gbp_only",
    "aud_buy",
    "aud_sell",
    "gbp_buy",
    "gbp_sell",
)

PASSED_GATE_CODES = frozenset(
    {
        "sufficient_train_sample",
        "sufficient_validation_sample",
        "sufficient_validation_coverage",
        "training_expectancy_positive",
        "training_profit_factor_at_least_one",
        "validation_expectancy_gate_passed",
        "validation_profit_factor_gate_passed",
        "validation_drawdown_gate_passed",
        "immutable_candidate_registered",
        "post_registration_evidence",
        "frozen_referee",
        "false_discovery_control_passed",
        "minimum_trade_count_passed",
        "minimum_pair_coverage_passed",
        "forward_shadow_passed",
    }
)
FAILURE_CODES = frozenset(
    {
        "unregistered_candidate",
        "insufficient_train_sample",
        "insufficient_validation_sample",
        "insufficient_validation_coverage",
        "training_expectancy_not_positive",
        "training_profit_factor_below_one",
        "validation_expectancy_gate_failed",
        "validation_profit_factor_gate_failed",
        "validation_drawdown_gate_failed",
        "evidence_stale",
        "revision_mismatch",
        "journal_unavailable",
        "report_invalid",
        "prospective_validation_required",
        "forward_shadow_required",
        "immutable_registration_required",
        "post_registration_evidence_required",
        "frozen_referee_required",
        "false_discovery_control_required",
        "minimum_trade_count_required",
        "minimum_pair_coverage_required",
        "promotion_protocol_not_implemented",
    }
)
ADAPTIVE_REASON_CODES = frozenset(
    {
        "small_sample",
        "loss_streak",
        "negative_expectancy",
        "profit_factor_below_one",
        "low_win_rate",
        "normal",
        "unavailable",
    }
)

_SAFE_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,79}$")
_SAFE_REVISION = re.compile(r"^[0-9a-f]{7,64}$")
_FORBIDDEN_KEY_PARTS = (
    "account",
    "balance",
    "credential",
    "equity",
    "order",
    "password",
    "price",
    "secret",
    "signal",
    "token",
    "trade_id",
    "transaction",
)

PUBLISH_INTERVAL_SECONDS = 3_600.0
_last_publish_monotonic: float | None = None
_last_publish_fingerprint: str | None = None


def _as_bool(value: object, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "y"}
    return bool(value)


def _env_bool(name: str, default: bool) -> bool:
    return _as_bool(os.getenv(name), default)


def _safe_int(value: object, default: int = 0, *, minimum: int = 0) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return max(minimum, result)


def _finite(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(result):
        return None
    return 0.0 if result == 0.0 else result


def _safe_version(value: object, default: str = "unknown") -> str:
    candidate = str(value or "").strip()
    return candidate if _SAFE_VERSION.fullmatch(candidate) else default


def _safe_revision(value: object) -> str:
    candidate = str(value or "").strip()
    if candidate == "unknown" or _SAFE_REVISION.fullmatch(candidate):
        return candidate
    return "unknown"


def _utc(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    try:
        return parsed.astimezone(timezone.utc)
    except (OverflowError, ValueError):
        return None


def _iso(value: datetime) -> str:
    timestamp = value
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc).replace(microsecond=0).isoformat()


def _snapshot_value(snapshot: object, name: str, default: object = None) -> object:
    if isinstance(snapshot, Mapping):
        return snapshot.get(name, default)
    return getattr(snapshot, name, default)


def _journal_available(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=2.0)
        try:
            columns = {
                str(row[1])
                for row in conn.execute("PRAGMA table_info(trades)").fetchall()
                if len(row) > 1
            }
            required = {
                "trade_id",
                "timestamp_utc",
                "instrument",
                "exit_timestamp_utc",
                "realized_pnl_ccy",
                "broker_confirmed",
                "run_tag",
            }
            if not required.issubset(columns):
                return False
            conn.execute("SELECT 1 FROM trades LIMIT 1").fetchone()
            return True
        finally:
            conn.close()
    except (OSError, sqlite3.Error):
        return False


def _pair_metrics(
    db_path: Path,
    instrument: str,
    *,
    journal_ok: bool,
    as_of_utc: datetime,
    cohort_run_tag: str,
    cohort_start_utc: str,
) -> dict[str, Any]:
    outcomes = []
    if journal_ok:
        outcomes = load_clean_outcomes(
            db_path,
            instruments=(instrument,),
            run_tag=cohort_run_tag,
            entry_start_utc=cohort_start_utc,
            as_of_utc=as_of_utc,
            limit=PAIR_WINDOW_SIZE,
            descending=True,
        )
    # The loader returns newest first for a bounded query. Drawdown must be
    # calculated in the order the outcomes became knowable.
    pnl = [outcome.realized_pnl_ccy for outcome in reversed(outcomes)]
    wins = sum(value > 0 for value in pnl)
    losses = sum(value < 0 for value in pnl)
    flat = len(pnl) - wins - losses
    if not pnl:
        return {
            "instrument": instrument,
            "window": PAIR_WINDOW,
            "sample_count": 0,
            "wins": 0,
            "losses": 0,
            "flat": 0,
            "win_rate": None,
            "closed_net_pnl_aud": None,
            "expectancy_aud": None,
            "profit_factor": None,
            "profit_factor_state": "no_sample",
            "max_drawdown_aud": None,
            "evidence_status": "no_sample",
        }

    gross_profit = sum(value for value in pnl if value > 0)
    gross_loss = abs(sum(value for value in pnl if value < 0))
    if gross_loss == 0:
        profit_factor = None
        profit_factor_state = "no_losses"
    else:
        profit_factor = gross_profit / gross_loss
        profit_factor_state = "finite"
    running = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for value in pnl:
        running += value
        peak = max(peak, running)
        max_drawdown = max(max_drawdown, peak - running)
    net = sum(pnl)
    return {
        "instrument": instrument,
        "window": PAIR_WINDOW,
        "sample_count": len(pnl),
        "wins": wins,
        "losses": losses,
        "flat": flat,
        "win_rate": wins / len(pnl),
        "closed_net_pnl_aud": net,
        "expectancy_aud": net / len(pnl),
        "profit_factor": profit_factor,
        "profit_factor_state": profit_factor_state,
        "max_drawdown_aud": max_drawdown,
        "evidence_status": (
            "sufficient" if len(pnl) == PAIR_WINDOW_SIZE else "small_sample"
        ),
    }


def _empty_metrics() -> dict[str, Any]:
    return {
        "trades": 0,
        "win_rate": None,
        "closed_net_pnl_aud": None,
        "expectancy_aud": None,
        "profit_factor": None,
        "profit_factor_state": "no_sample",
        "max_drawdown_aud": None,
    }


def _shadow_metrics(value: object) -> tuple[dict[str, Any], bool, float]:
    """Return a sanitized metric block, validity, and comparable PF value."""

    if not isinstance(value, Mapping):
        return _empty_metrics(), False, 0.0
    trades = _safe_int(value.get("trades"), -1, minimum=-1)
    wins = _safe_int(value.get("wins"), -1, minimum=-1)
    losses = _safe_int(value.get("losses"), -1, minimum=-1)
    if trades < 0 or wins < 0 or losses < 0 or wins + losses > trades:
        return _empty_metrics(), False, 0.0
    if trades == 0:
        return _empty_metrics(), True, 0.0

    win_rate = _finite(value.get("win_rate"))
    net = _finite(value.get("net_profit"))
    expectancy = _finite(value.get("expectancy"))
    drawdown = _finite(value.get("max_drawdown"))
    reported_pf = _finite(value.get("profit_factor"))
    if (
        win_rate is None
        or not 0.0 <= win_rate <= 1.0
        or net is None
        or expectancy is None
        or drawdown is None
        or drawdown < 0.0
    ):
        return _empty_metrics(), False, 0.0

    if losses == 0:
        pf = None
        pf_state = "no_losses"
        comparable_pf = math.inf if net > 0 else 0.0
    elif reported_pf is None or reported_pf < 0.0:
        return _empty_metrics(), False, 0.0
    else:
        pf = reported_pf
        pf_state = "finite"
        comparable_pf = reported_pf
    return (
        {
            "trades": trades,
            "win_rate": win_rate,
            "closed_net_pnl_aud": net,
            "expectancy_aud": expectancy,
            "profit_factor": pf,
            "profit_factor_state": pf_state,
            "max_drawdown_aud": drawdown,
        },
        True,
        comparable_pf,
    )


def _load_shadow_report(path: Path) -> tuple[dict[str, Any] | None, bool]:
    try:
        if not path.is_file() or path.stat().st_size > 2_000_000:
            return None, False
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None, False
    return (dict(payload), True) if isinstance(payload, Mapping) else (None, False)


def _candidate_result(
    candidate: Mapping[str, Any],
    *,
    baseline: dict[str, Any],
    baseline_pf: float,
    min_train: int,
    min_validation: int,
    min_coverage: float,
    stale: bool,
    revision_mismatch: bool,
    evidence_scope: str,
    protocol: Mapping[str, Any],
    total_clean_trades: int,
) -> tuple[dict[str, Any] | None, bool]:
    candidate_id = str(candidate.get("name") or "").strip()
    if candidate_id not in REGISTERED_CANDIDATES:
        return None, False
    train, train_valid, train_pf = _shadow_metrics(candidate.get("train"))
    validation, validation_valid, validation_pf = _shadow_metrics(
        candidate.get("validation")
    )
    coverage = _finite(candidate.get("validation_coverage"))
    if coverage is None or not 0.0 <= coverage <= 1.0:
        coverage = 0.0
        validation_valid = False

    passed: list[str] = []
    failed: list[str] = []
    if stale:
        failed.append("evidence_stale")
    if revision_mismatch:
        failed.append("revision_mismatch")
    if not train_valid or not validation_valid:
        failed.append("report_invalid")

    gates = (
        (train["trades"] >= min_train, "sufficient_train_sample", "insufficient_train_sample"),
        (
            validation["trades"] >= min_validation,
            "sufficient_validation_sample",
            "insufficient_validation_sample",
        ),
        (
            coverage >= min_coverage,
            "sufficient_validation_coverage",
            "insufficient_validation_coverage",
        ),
        (
            train["expectancy_aud"] is not None
            and train["expectancy_aud"] > 0.0,
            "training_expectancy_positive",
            "training_expectancy_not_positive",
        ),
        (
            train_pf >= 1.0,
            "training_profit_factor_at_least_one",
            "training_profit_factor_below_one",
        ),
    )
    for condition, pass_code, failure_code in gates:
        (passed if condition else failed).append(pass_code if condition else failure_code)

    baseline_expectancy = baseline["expectancy_aud"]
    expectancy_margin = (
        max(0.05, abs(baseline_expectancy) * 0.15)
        if baseline_expectancy is not None
        else math.inf
    )
    validation_expectancy_ok = (
        validation["expectancy_aud"] is not None
        and baseline_expectancy is not None
        and validation["expectancy_aud"] >= baseline_expectancy + expectancy_margin
    )
    validation_pf_ok = validation_pf >= max(1.10, baseline_pf)
    baseline_drawdown = baseline["max_drawdown_aud"]
    validation_drawdown_ok = (
        validation["max_drawdown_aud"] is not None
        and baseline_drawdown is not None
        and validation["max_drawdown_aud"] <= baseline_drawdown
    )
    for condition, pass_code, failure_code in (
        (
            validation_expectancy_ok,
            "validation_expectancy_gate_passed",
            "validation_expectancy_gate_failed",
        ),
        (
            validation_pf_ok,
            "validation_profit_factor_gate_passed",
            "validation_profit_factor_gate_failed",
        ),
        (
            validation_drawdown_ok,
            "validation_drawdown_gate_passed",
            "validation_drawdown_gate_failed",
        ),
    ):
        (passed if condition else failed).append(pass_code if condition else failure_code)

    if evidence_scope != "prospective_shadow_observations":
        failed.extend(
            ["prospective_validation_required", "forward_shadow_required"]
        )
    else:
        candidate_protocol = candidate.get("governance_evidence")
        if not isinstance(candidate_protocol, Mapping):
            candidate_protocol = {}

        def protocol_value(name: str, default: object = None) -> object:
            if name in candidate_protocol:
                return candidate_protocol.get(name)
            return protocol.get(name, default)

        fdr_alpha = _finite(protocol_value("false_discovery_alpha"))
        governance_gates = (
            (
                _as_bool(protocol_value("immutable_registration")),
                "immutable_candidate_registered",
                "immutable_registration_required",
            ),
            (
                _as_bool(protocol_value("post_registration_evidence")),
                "post_registration_evidence",
                "post_registration_evidence_required",
            ),
            (
                _as_bool(protocol_value("frozen_referee")),
                "frozen_referee",
                "frozen_referee_required",
            ),
            (
                _as_bool(protocol_value("false_discovery_controlled"))
                and fdr_alpha is not None
                and 0.0 < fdr_alpha <= 0.05,
                "false_discovery_control_passed",
                "false_discovery_control_required",
            ),
            (
                total_clean_trades >= 250 and validation["trades"] >= 100,
                "minimum_trade_count_passed",
                "minimum_trade_count_required",
            ),
            (
                _safe_int(protocol_value("validation_pair_count"), 0) >= 3,
                "minimum_pair_coverage_passed",
                "minimum_pair_coverage_required",
            ),
            (
                _as_bool(protocol_value("forward_shadow_passed")),
                "forward_shadow_passed",
                "forward_shadow_required",
            ),
        )
        for condition, pass_code, failure_code in governance_gates:
            (passed if condition else failed).append(
                pass_code if condition else failure_code
            )

    # V1 has no independent immutable registration store or frozen referee.
    # Assertions embedded in the worker-produced report are advisory, not
    # sufficient promotion proof.  No candidate may become READY until that
    # separate protocol exists.
    failed.append("promotion_protocol_not_implemented")

    unavailable = any(
        reason in {"report_invalid", "revision_mismatch"} for reason in failed
    )
    collecting = any(
        reason
        in {
            "insufficient_train_sample",
            "insufficient_validation_sample",
            "insufficient_validation_coverage",
            "prospective_validation_required",
            "forward_shadow_required",
            "immutable_registration_required",
            "post_registration_evidence_required",
            "frozen_referee_required",
            "false_discovery_control_required",
            "minimum_trade_count_required",
            "minimum_pair_coverage_required",
            "promotion_protocol_not_implemented",
        }
        for reason in failed
    )
    if unavailable:
        status = "unavailable"
    elif stale:
        status = "stale"
    elif collecting:
        status = "collecting"
    elif not failed:
        status = "ready_for_luke_review"
    else:
        status = "rejected"
    return (
        {
            "candidate_id": candidate_id,
            "definition_version": DEFINITION_VERSION,
            "status": status,
            "train": train,
            "validation": validation,
            "validation_coverage": coverage,
            "passed_gates": passed,
            "failed_gates": list(dict.fromkeys(failed)),
        },
        True,
    )


def _adaptive_status(
    snapshot: object,
    *,
    enabled: bool,
    lookback: int,
    minimum_sample: int,
) -> dict[str, Any]:
    if not enabled:
        return {
            "state": "disabled",
            "mode": "reduce_only",
            "lookback_closed_trades": lookback,
            "minimum_sample": minimum_sample,
            "sample_count": 0,
            "risk_multiplier": None,
            "reason_code": "unavailable",
        }
    if snapshot is None:
        return {
            "state": "error",
            "mode": "reduce_only",
            "lookback_closed_trades": lookback,
            "minimum_sample": minimum_sample,
            "sample_count": 0,
            "risk_multiplier": None,
            "reason_code": "unavailable",
        }
    sample_count = _safe_int(_snapshot_value(snapshot, "session_closed_trades"), 0)
    multiplier = _finite(_snapshot_value(snapshot, "risk_multiplier"))
    if multiplier is None or not 0.0 <= multiplier <= 1.0:
        state = "error"
        multiplier = None
        reason = "unavailable"
    elif sample_count < minimum_sample:
        state = "collecting"
        reason = "small_sample"
    else:
        state = "active"
        supplied_reason = str(_snapshot_value(snapshot, "reason_code", "")).strip()
        if supplied_reason in ADAPTIVE_REASON_CODES - {"small_sample", "unavailable"}:
            reason = supplied_reason
        elif _safe_int(_snapshot_value(snapshot, "loss_streak"), 0) >= 2:
            reason = "loss_streak"
        else:
            reason = "normal"
    return {
        "state": state,
        "mode": "reduce_only",
        "lookback_closed_trades": lookback,
        "minimum_sample": minimum_sample,
        "sample_count": sample_count,
        "risk_multiplier": multiplier,
        "reason_code": reason,
    }


def _observation_capture(snapshot: object) -> dict[str, Any]:
    """Sanitize aggregate observer health without exposing decision records."""

    unavailable = {
        "state": "degraded",
        "mode": "best_effort",
        "submitted": 0,
        "persisted_events": 0,
        "duplicate_events": 0,
        "queue_drops": 0,
        "write_errors": 0,
        "pending_events": None,
    }
    required = {
        "submitted",
        "persisted_events",
        "duplicate_events",
        "queue_drops",
        "write_errors",
    }
    if not isinstance(snapshot, Mapping) or not required.issubset(snapshot):
        return unavailable
    if not set(snapshot).issubset(required | {"pending_events"}):
        return unavailable

    counters: dict[str, int] = {}
    for name in required:
        value = snapshot.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return unavailable
        counters[name] = value
    raw_pending = snapshot.get("pending_events")
    if raw_pending is None:
        pending: int | None = None
    elif (
        isinstance(raw_pending, bool)
        or not isinstance(raw_pending, int)
        or raw_pending < 0
    ):
        return unavailable
    else:
        pending = raw_pending

    degraded = counters["queue_drops"] > 0 or counters["write_errors"] > 0
    return {
        "state": "degraded" if degraded else "active",
        "mode": "best_effort",
        "submitted": counters["submitted"],
        "persisted_events": counters["persisted_events"],
        "duplicate_events": counters["duplicate_events"],
        "queue_drops": counters["queue_drops"],
        "write_errors": counters["write_errors"],
        "pending_events": pending,
    }


def build_learning_review(
    db_path: Path | str,
    *,
    shadow_report_path: Path | str | None = None,
    adaptive_snapshot: object = None,
    observation_snapshot: object = None,
    source_revision: str = "unknown",
    observed_at: datetime | None = None,
    adaptive_enabled: bool | None = None,
    setup_policy_enabled: bool | None = None,
    shadow_enabled: bool | None = None,
    adaptive_lookback: int | None = None,
    adaptive_minimum_sample: int | None = None,
    shadow_minimum_train: int | None = None,
    shadow_minimum_validation: int | None = None,
    shadow_minimum_coverage: float | None = None,
    shadow_max_age_seconds: float | None = None,
) -> dict[str, Any]:
    """Build a read-only, aggregate-only review of Mossy's learning evidence.

    No broker calls or trading actions occur here. Malformed or missing inputs
    produce explicit unavailable/collecting states rather than inferred facts.
    """

    timestamp = observed_at or datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    timestamp = timestamp.astimezone(timezone.utc)
    revision = _safe_revision(source_revision)
    journal_path = Path(db_path)
    report_path = (
        Path(shadow_report_path)
        if shadow_report_path is not None
        else journal_path.parent / "shadow_learning_report.json"
    )
    journal_ok = _journal_available(journal_path)

    adaptive_on = (
        _env_bool("ADAPTIVE_TUNING_ENABLED", True)
        if adaptive_enabled is None
        else bool(adaptive_enabled)
    )
    setup_on = (
        _env_bool("ADAPTIVE_POLICY_ENABLED", True)
        if setup_policy_enabled is None
        else bool(setup_policy_enabled)
    )
    shadow_on = (
        _env_bool("SHADOW_LEARNING_ENABLED", True)
        if shadow_enabled is None
        else bool(shadow_enabled)
    )
    lookback = max(
        1,
        adaptive_lookback
        if adaptive_lookback is not None
        else _safe_int(os.getenv("ADAPTIVE_LOOKBACK"), 80, minimum=1),
    )
    adaptive_min = max(
        1,
        adaptive_minimum_sample
        if adaptive_minimum_sample is not None
        else _safe_int(os.getenv("ADAPTIVE_MIN_SAMPLE"), 20, minimum=1),
    )
    min_train = max(
        1,
        shadow_minimum_train
        if shadow_minimum_train is not None
        else _safe_int(os.getenv("SHADOW_MIN_TRAIN"), 50, minimum=1),
    )
    min_validation = max(
        1,
        shadow_minimum_validation
        if shadow_minimum_validation is not None
        else _safe_int(os.getenv("SHADOW_MIN_VALIDATION"), 30, minimum=1),
    )
    if shadow_minimum_coverage is None:
        coverage_value = _finite(os.getenv("SHADOW_MIN_COVERAGE", "0.50"))
        min_coverage = coverage_value if coverage_value is not None else 0.5
    else:
        min_coverage = float(shadow_minimum_coverage)
    min_coverage = max(0.0, min(1.0, min_coverage))

    adaptive_run_tag = str(
        _snapshot_value(adaptive_snapshot, "filter_run_tag", "") or ""
    ).strip()
    expected_run_tag = (
        adaptive_run_tag
        if adaptive_run_tag.lower() not in {"", "all", "none", "unknown"}
        else (
            os.getenv("ADAPTIVE_RUN_TAG", DEFAULT_LEARNING_RUN_TAG).strip()
            or DEFAULT_LEARNING_RUN_TAG
        )
    )
    adaptive_window_start_raw = str(
        _snapshot_value(adaptive_snapshot, "filter_window_start_utc", "") or ""
    ).strip()
    if adaptive_window_start_raw.lower() in {"", "none", "unknown"}:
        adaptive_window_start_raw = (
            os.getenv(
                "ADAPTIVE_WINDOW_START_UTC",
                CLEAN_COHORT_START_UTC,
            ).strip()
            or CLEAN_COHORT_START_UTC
        )
    expected_cohort_start = _utc(adaptive_window_start_raw)
    if expected_cohort_start is None:
        journal_ok = False

    pair_performance = [
        _pair_metrics(
            journal_path,
            instrument,
            journal_ok=journal_ok,
            as_of_utc=timestamp,
            cohort_run_tag=expected_run_tag,
            cohort_start_utc=(
                _iso(expected_cohort_start)
                if expected_cohort_start is not None
                else adaptive_window_start_raw
            ),
        )
        for instrument in INSTRUMENTS
    ]
    adaptive = _adaptive_status(
        adaptive_snapshot,
        enabled=adaptive_on,
        lookback=lookback,
        minimum_sample=adaptive_min,
    )
    setup_policy = {
        "state": "active" if setup_on else "disabled",
        "mode": "reduce_or_block_only",
    }
    observation_capture = _observation_capture(observation_snapshot)

    report, report_ok = _load_shadow_report(report_path) if shadow_on else (None, False)
    generated_at = None if report is None else _utc(report.get("generated_utc"))
    max_age = shadow_max_age_seconds
    if max_age is None:
        interval = _finite(os.getenv("SHADOW_INTERVAL_SECONDS", "3600")) or 3600.0
        max_age = max(7_200.0, interval * 2.0)
    evidence_age = (
        (timestamp - generated_at).total_seconds() if generated_at is not None else None
    )
    evidence_stale = evidence_age is not None and (
        evidence_age > max(1.0, float(max_age)) or evidence_age < -300.0
    )
    reported_cohort_start = (
        None if report is None else _utc(report.get("cohort_start_utc"))
    )
    cohort_start = reported_cohort_start or _utc(CLEAN_COHORT_START_UTC)
    assert cohort_start is not None

    report_revision_raw = (
        None if report is None else report.get("evidence_revision")
    )
    # Evidence provenance is explicit. Missing/invalid legacy provenance,
    # unknown runtime provenance, and mismatches all fail closed; never relabel
    # a persisted report with the currently running revision.
    evidence_revision = _safe_revision(report_revision_raw)
    revision_mismatch = bool(
        report is not None
        and (
            revision == "unknown"
            or evidence_revision == "unknown"
            or evidence_revision != revision
        )
    )
    report_run_tag = (
        "" if report is None else str(report.get("cohort_run_tag") or "").strip()
    )
    raw_report_instruments = None if report is None else report.get("instruments")
    report_instruments = (
        tuple(str(item or "").strip().upper() for item in raw_report_instruments)
        if isinstance(raw_report_instruments, (list, tuple))
        else ()
    )
    cohort_lineage_mismatch = bool(
        report is not None
        and (
            report_run_tag != expected_run_tag
            or report_instruments != INSTRUMENTS
            or reported_cohort_start is None
            or expected_cohort_start is None
            or reported_cohort_start != expected_cohort_start
        )
    )
    total_clean = _safe_int(None if report is None else report.get("total_clean_trades"), 0)
    train_trades = _safe_int(None if report is None else report.get("train_trades"), 0)
    validation_trades = _safe_int(
        None if report is None else report.get("validation_trades"), 0
    )
    requested_scope = (
        "" if report is None else str(report.get("evidence_scope") or "").strip()
    )
    evidence_scope = (
        "prospective_shadow_observations"
        if requested_scope == "prospective_shadow_observations"
        else "executed_trade_subsets"
    )
    protocol = (
        report.get("validation_protocol")
        if isinstance(report, Mapping)
        and isinstance(report.get("validation_protocol"), Mapping)
        else {}
    )

    baseline_source = None
    if isinstance(report, Mapping) and isinstance(report.get("baseline"), Mapping):
        baseline_source = report["baseline"].get("validation")
    baseline, baseline_valid, baseline_pf = _shadow_metrics(baseline_source)
    known_candidates: list[dict[str, Any]] = []
    unknown_candidate_seen = False
    if isinstance(report, Mapping) and isinstance(report.get("candidates"), list):
        for raw_candidate in report["candidates"]:
            if not isinstance(raw_candidate, Mapping):
                unknown_candidate_seen = True
                continue
            result, known = _candidate_result(
                raw_candidate,
                baseline=baseline,
                baseline_pf=baseline_pf,
                min_train=min_train,
                min_validation=min_validation,
                min_coverage=min_coverage,
                stale=evidence_stale,
                revision_mismatch=revision_mismatch,
                evidence_scope=evidence_scope,
                protocol=protocol,
                total_clean_trades=total_clean,
            )
            if not known:
                unknown_candidate_seen = True
            elif result is not None:
                known_candidates.append(result)
    elif report_ok:
        report_ok = False

    # Duplicate IDs make selection ambiguous. Never publish both or pick one.
    candidate_ids = [candidate["candidate_id"] for candidate in known_candidates]
    if len(candidate_ids) != len(set(candidate_ids)):
        report_ok = False
        known_candidates = []
    report_ok = bool(
        report_ok
        and generated_at is not None
        and baseline_valid
        and not cohort_lineage_mismatch
        and not unknown_candidate_seen
        and known_candidates
    )
    if not journal_ok:
        readiness_state = "unavailable"
        readiness_reasons = ["journal_unavailable"]
        leading_candidate = None
        next_action = "none"
    elif not shadow_on:
        readiness_state = "unavailable"
        readiness_reasons = ["report_invalid"]
        leading_candidate = None
        next_action = "none"
    elif not report_ok or revision_mismatch:
        readiness_state = "unavailable"
        readiness_reasons = [
            "revision_mismatch"
            if revision_mismatch
            else (
                "unregistered_candidate"
                if unknown_candidate_seen
                else "report_invalid"
            )
        ]
        leading_candidate = None
        next_action = "none"
        for candidate in known_candidates:
            candidate["status"] = "unavailable"
            candidate["failed_gates"] = list(
                dict.fromkeys(candidate["failed_gates"] + readiness_reasons)
            )
    elif evidence_stale:
        readiness_state = "stale_evidence"
        readiness_reasons = ["evidence_stale"]
        leading_candidate = None
        next_action = "none"
    else:
        ready = [
            candidate
            for candidate in known_candidates
            if candidate["status"] == "ready_for_luke_review"
        ]
        if ready:
            ready.sort(
                key=lambda candidate: (
                    candidate["validation"]["expectancy_aud"],
                    candidate["validation"]["profit_factor"]
                    if candidate["validation"]["profit_factor"] is not None
                    else math.inf,
                    -candidate["validation"]["max_drawdown_aud"],
                    candidate["candidate_id"],
                ),
                reverse=True,
            )
            readiness_state = "ready_for_luke_review"
            leading_candidate = ready[0]["candidate_id"]
            readiness_reasons = []
            next_action = "luke_review_only"
        elif any(candidate["status"] == "collecting" for candidate in known_candidates):
            readiness_state = "collecting_data"
            leading_candidate = None
            readiness_reasons = list(
                dict.fromkeys(
                    reason
                    for candidate in known_candidates
                    for reason in candidate["failed_gates"]
                    if reason.startswith("insufficient_")
                    or reason
                    in {
                        "prospective_validation_required",
                        "forward_shadow_required",
                        "immutable_registration_required",
                        "post_registration_evidence_required",
                        "frozen_referee_required",
                        "false_discovery_control_required",
                        "minimum_trade_count_required",
                        "minimum_pair_coverage_required",
                        "promotion_protocol_not_implemented",
                    }
                )
            )
            next_action = "collect_more_evidence"
        else:
            readiness_state = "no_candidate_passed"
            leading_candidate = None
            readiness_reasons = list(
                dict.fromkeys(
                    reason
                    for candidate in known_candidates
                    for reason in candidate["failed_gates"]
                )
            )
            next_action = "none"

    promotion_protocol_pending = any(
        "promotion_protocol_not_implemented" in candidate["failed_gates"]
        for candidate in known_candidates
    )
    if not shadow_on:
        shadow_state = "disabled"
    elif not report_ok or not journal_ok or revision_mismatch:
        shadow_state = "error"
    elif evidence_stale:
        shadow_state = "stale"
    elif (
        evidence_scope != "prospective_shadow_observations"
        or promotion_protocol_pending
    ):
        shadow_state = "collecting"
    elif train_trades < min_train or validation_trades < min_validation:
        shadow_state = "collecting"
    else:
        shadow_state = "active"
    shadow_learning = {
        "state": shadow_state,
        "mode": "advisory_only",
        "evidence_scope": evidence_scope,
        "can_evaluate_more_trade_hypotheses": (
            evidence_scope == "prospective_shadow_observations"
        ),
        "cohort_start_utc": _iso(cohort_start),
        "clean_trades": total_clean,
        "train_trades": train_trades,
        "validation_trades": validation_trades,
        "minimum_train": min_train,
        "minimum_validation": min_validation,
        "minimum_coverage": min_coverage,
        "auto_apply": False,
    }

    component_states = {adaptive["state"], setup_policy["state"], shadow_state}
    if observation_capture["state"] == "degraded":
        overall = "degraded"
    elif component_states <= {"disabled"}:
        overall = "disabled"
    elif "error" in component_states or "stale" in component_states or not journal_ok:
        overall = "degraded"
    elif "collecting" in component_states or any(
        pair["evidence_status"] != "sufficient" for pair in pair_performance
    ):
        overall = "collecting"
    else:
        overall = "active"

    return {
        "schema_version": SCHEMA_VERSION,
        "observed_at_utc": _iso(timestamp),
        "source_revision": revision,
        "freshness": {"state": "fresh", "age_seconds": 0.0},
        "learning_status": {
            "overall": overall,
            "adaptive_tuner": adaptive,
            "setup_policy": setup_policy,
            "shadow_learning": shadow_learning,
            "observation_capture": observation_capture,
        },
        "pair_performance": pair_performance,
        "challenger_review": {
            "registry_version": REGISTRY_VERSION,
            "evidence_generated_at_utc": (
                _iso(generated_at) if generated_at is not None else UNKNOWN_EVIDENCE_TIME
            ),
            "evidence_revision": evidence_revision,
            "baseline_validation": baseline,
            "candidates": known_candidates,
            "readiness": {
                "state": readiness_state,
                "leading_candidate_id": leading_candidate,
                "reason_codes": readiness_reasons,
                "next_allowed_action": next_action,
                "luke_approval_required": True,
                "auto_promotion_permitted": False,
            },
        },
        "governance": {
            "read_only": True,
            "demo_only": True,
            "can_trade": False,
            "can_write_configuration": False,
            "can_run_optimisation": False,
            "can_deploy": False,
            "can_promote": False,
        },
    }


def _exact_keys(value: object, expected: set[str]) -> bool:
    return isinstance(value, Mapping) and set(value) == expected


def _all_finite(value: object) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, list):
        return all(_all_finite(item) for item in value)
    if isinstance(value, Mapping):
        return all(_all_finite(item) for item in value.values())
    return False


def _has_forbidden_key(value: object) -> bool:
    if isinstance(value, list):
        return any(_has_forbidden_key(item) for item in value)
    if not isinstance(value, Mapping):
        return False
    for key, nested in value.items():
        normalized = str(key).lower()
        if any(part in normalized for part in _FORBIDDEN_KEY_PARTS):
            return True
        if _has_forbidden_key(nested):
            return True
    return False


def validate_learning_review(snapshot: object) -> bool:
    """Strictly validate the worker-controlled payload before network egress."""

    if not _exact_keys(
        snapshot,
        {
            "schema_version",
            "observed_at_utc",
            "source_revision",
            "freshness",
            "learning_status",
            "pair_performance",
            "challenger_review",
            "governance",
        },
    ):
        return False
    assert isinstance(snapshot, Mapping)
    if snapshot.get("schema_version") != SCHEMA_VERSION:
        return False
    if _utc(snapshot.get("observed_at_utc")) is None:
        return False
    if _safe_revision(snapshot.get("source_revision")) != snapshot.get(
        "source_revision"
    ):
        return False
    if _has_forbidden_key(snapshot) or not _all_finite(snapshot):
        return False

    freshness = snapshot.get("freshness")
    if not _exact_keys(freshness, {"state", "age_seconds"}):
        return False
    if freshness["state"] not in {"fresh", "stale", "future", "unknown"}:
        return False
    age = freshness["age_seconds"]
    if age is not None and (_finite(age) is None or float(age) < 0.0):
        return False

    learning = snapshot.get("learning_status")
    if not _exact_keys(
        learning,
        {
            "overall",
            "adaptive_tuner",
            "setup_policy",
            "shadow_learning",
            "observation_capture",
        },
    ):
        return False
    if learning["overall"] not in {"active", "collecting", "degraded", "disabled"}:
        return False
    adaptive = learning["adaptive_tuner"]
    if not _exact_keys(
        adaptive,
        {
            "state",
            "mode",
            "lookback_closed_trades",
            "minimum_sample",
            "sample_count",
            "risk_multiplier",
            "reason_code",
        },
    ):
        return False
    if adaptive["state"] not in {"active", "collecting", "disabled", "error"}:
        return False
    if adaptive["mode"] != "reduce_only" or adaptive["reason_code"] not in ADAPTIVE_REASON_CODES:
        return False
    multiplier = adaptive["risk_multiplier"]
    if multiplier is not None and (
        _finite(multiplier) is None or not 0.0 <= float(multiplier) <= 1.0
    ):
        return False
    policy = learning["setup_policy"]
    if not _exact_keys(policy, {"state", "mode"}):
        return False
    if policy["state"] not in {"active", "disabled", "error"} or policy["mode"] != "reduce_or_block_only":
        return False
    shadow = learning["shadow_learning"]
    if not _exact_keys(
        shadow,
        {
            "state",
            "mode",
            "evidence_scope",
            "can_evaluate_more_trade_hypotheses",
            "cohort_start_utc",
            "clean_trades",
            "train_trades",
            "validation_trades",
            "minimum_train",
            "minimum_validation",
            "minimum_coverage",
            "auto_apply",
        },
    ):
        return False
    if shadow["state"] not in {"active", "collecting", "stale", "disabled", "error"}:
        return False
    if shadow["mode"] != "advisory_only" or shadow["evidence_scope"] not in {
        "executed_trade_subsets",
        "prospective_shadow_observations",
    }:
        return False
    if shadow["auto_apply"] is not False:
        return False

    observation = learning["observation_capture"]
    observation_keys = {
        "state",
        "mode",
        "submitted",
        "persisted_events",
        "duplicate_events",
        "queue_drops",
        "write_errors",
        "pending_events",
    }
    if not _exact_keys(observation, observation_keys):
        return False
    if observation["state"] not in {"active", "degraded"}:
        return False
    if observation["mode"] != "best_effort":
        return False
    for name in observation_keys - {"state", "mode", "pending_events"}:
        value = observation[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return False
    pending = observation["pending_events"]
    if pending is not None and (
        isinstance(pending, bool) or not isinstance(pending, int) or pending < 0
    ):
        return False
    if observation["state"] == "active" and (
        observation["queue_drops"] > 0 or observation["write_errors"] > 0
    ):
        return False
    if observation["state"] == "degraded" and learning["overall"] != "degraded":
        return False

    pairs = snapshot.get("pair_performance")
    if not isinstance(pairs, list) or len(pairs) != len(INSTRUMENTS):
        return False
    pair_keys = {
        "instrument",
        "window",
        "sample_count",
        "wins",
        "losses",
        "flat",
        "win_rate",
        "closed_net_pnl_aud",
        "expectancy_aud",
        "profit_factor",
        "profit_factor_state",
        "max_drawdown_aud",
        "evidence_status",
    }
    if {pair.get("instrument") for pair in pairs if isinstance(pair, Mapping)} != set(INSTRUMENTS):
        return False
    for pair in pairs:
        if not _exact_keys(pair, pair_keys) or pair["window"] != PAIR_WINDOW:
            return False
        if pair["profit_factor_state"] not in {"finite", "no_losses", "no_sample"}:
            return False
        if pair["evidence_status"] not in {"sufficient", "small_sample", "no_sample"}:
            return False

    review = snapshot.get("challenger_review")
    if not _exact_keys(
        review,
        {
            "registry_version",
            "evidence_generated_at_utc",
            "evidence_revision",
            "baseline_validation",
            "candidates",
            "readiness",
        },
    ):
        return False
    metric_keys = {
        "trades",
        "win_rate",
        "closed_net_pnl_aud",
        "expectancy_aud",
        "profit_factor",
        "profit_factor_state",
        "max_drawdown_aud",
    }
    if not _exact_keys(review["baseline_validation"], metric_keys):
        return False
    candidate_keys = {
        "candidate_id",
        "definition_version",
        "status",
        "train",
        "validation",
        "validation_coverage",
        "passed_gates",
        "failed_gates",
    }
    if not isinstance(review["candidates"], list):
        return False
    for candidate in review["candidates"]:
        if not _exact_keys(candidate, candidate_keys):
            return False
        if candidate["candidate_id"] not in REGISTERED_CANDIDATES:
            return False
        if candidate["status"] not in {
            "ready_for_luke_review",
            "rejected",
            "collecting",
            "stale",
            "unavailable",
        }:
            return False
        if not _exact_keys(candidate["train"], metric_keys) or not _exact_keys(
            candidate["validation"], metric_keys
        ):
            return False
        if not set(candidate["passed_gates"]).issubset(PASSED_GATE_CODES):
            return False
        if not set(candidate["failed_gates"]).issubset(FAILURE_CODES):
            return False
    readiness = review["readiness"]
    if not _exact_keys(
        readiness,
        {
            "state",
            "leading_candidate_id",
            "reason_codes",
            "next_allowed_action",
            "luke_approval_required",
            "auto_promotion_permitted",
        },
    ):
        return False
    if readiness["state"] not in {
        "ready_for_luke_review",
        "no_candidate_passed",
        "collecting_data",
        "stale_evidence",
        "unavailable",
    }:
        return False
    if readiness["leading_candidate_id"] not in (*REGISTERED_CANDIDATES, None):
        return False
    if not set(readiness["reason_codes"]).issubset(FAILURE_CODES):
        return False
    if readiness["next_allowed_action"] not in {
        "luke_review_only",
        "collect_more_evidence",
        "none",
    }:
        return False
    if readiness["luke_approval_required"] is not True or readiness["auto_promotion_permitted"] is not False:
        return False

    governance = snapshot.get("governance")
    expected_governance = {
        "read_only": True,
        "demo_only": True,
        "can_trade": False,
        "can_write_configuration": False,
        "can_run_optimisation": False,
        "can_deploy": False,
        "can_promote": False,
    }
    return dict(governance) == expected_governance if isinstance(governance, Mapping) else False


def _learning_url() -> str:
    explicit = os.getenv("MOSSY_MCP_LEARNING_URL", "").strip()
    if explicit:
        return explicit
    heartbeat = os.getenv("MOSSY_MCP_STATUS_URL", "").strip()
    suffix = "/internal/runtime-heartbeat"
    if heartbeat.endswith(suffix):
        return heartbeat[: -len(suffix)] + "/internal/learning-snapshot"
    return ""


def _publish_fingerprint(snapshot: Mapping[str, Any]) -> str:
    stable = dict(snapshot)
    stable.pop("observed_at_utc", None)
    freshness = stable.get("freshness")
    if isinstance(freshness, Mapping):
        stable["freshness"] = {"state": freshness.get("state")}
    return json.dumps(stable, sort_keys=True, separators=(",", ":"))


async def publish_learning_review(snapshot: object) -> tuple[bool, str]:
    """Publish sanitized evidence without ever raising into the trading loop."""

    global _last_publish_fingerprint, _last_publish_monotonic

    if not validate_learning_review(snapshot):
        return False, "invalid-payload"
    assert isinstance(snapshot, Mapping)
    url = _learning_url()
    key = os.getenv("MOSSY_MCP_STATUS_KEY", "").strip()
    if not url or not key:
        return False, "disabled"
    allow_http = _env_bool("MOSSY_MCP_ALLOW_INSECURE_HTTP", False)
    if not url.startswith("https://") and not allow_http:
        return False, "insecure-url-blocked"
    now_monotonic = time.monotonic()
    fingerprint = _publish_fingerprint(snapshot)
    if (
        _last_publish_monotonic is not None
        and _last_publish_fingerprint == fingerprint
        and now_monotonic - _last_publish_monotonic < PUBLISH_INTERVAL_SECONDS
    ):
        return False, "throttled"
    try:
        async with httpx.AsyncClient(timeout=3.0, follow_redirects=False) as client:
            response = await client.post(
                url,
                headers={"Authorization": f"Bearer {key}"},
                json=snapshot,
            )
            response.raise_for_status()
    except httpx.HTTPError as exc:
        return False, f"http-error:{type(exc).__name__}"
    except Exception as exc:  # pragma: no cover - defensive fail-open guard
        return False, f"error:{type(exc).__name__}"
    _last_publish_monotonic = now_monotonic
    _last_publish_fingerprint = fingerprint
    return True, "sent"


__all__ = [
    "FAILURE_CODES",
    "INSTRUMENTS",
    "PASSED_GATE_CODES",
    "REGISTERED_CANDIDATES",
    "SCHEMA_VERSION",
    "build_learning_review",
    "publish_learning_review",
    "validate_learning_review",
]
