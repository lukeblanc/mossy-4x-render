from __future__ import annotations

import os
import secrets
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Literal

from mcp.server import MCPServer
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import ToolAnnotations
from pydantic import BaseModel, ConfigDict, Field, model_validator
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from bridge_state import (
    load_learning_snapshot,
    load_runtime_heartbeat,
    save_learning_snapshot,
    save_runtime_heartbeat,
)


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
STATUS_KEY_ENV = "MOSSY_MCP_STATUS_KEY"
STATUS_STALE_SECONDS = 900
LEARNING_STALE_SECONDS = 7_200
FUTURE_TOLERANCE_SECONDS = 60
SUPERVISOR_EQUITY_FLOOR_AUD = 5_000.0
LEARNING_SCHEMA_VERSION = "mossy.learning-review.v1"
LEARNING_REGISTRY_VERSION = "mossy-shadow-registry.v1"
LEARNING_DEFINITION_VERSION = "shadow-filter.v1"
UNKNOWN_EVIDENCE_TIME = "1970-01-01T00:00:00+00:00"

CandidateId = Literal[
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
]
FailureReason = Literal[
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
]
PassedGate = Literal[
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
]

STATISTICAL_READY_GATES = frozenset(
    {
        "sufficient_train_sample",
        "sufficient_validation_sample",
        "sufficient_validation_coverage",
        "training_expectancy_positive",
        "training_profit_factor_at_least_one",
        "validation_expectancy_gate_passed",
        "validation_profit_factor_gate_passed",
        "validation_drawdown_gate_passed",
    }
)
GOVERNANCE_READY_GATES = frozenset(
    {
        "immutable_candidate_registered",
        "post_registration_evidence",
        "frozen_referee",
        "false_discovery_control_passed",
        "minimum_trade_count_passed",
        "minimum_pair_coverage_passed",
        "forward_shadow_passed",
    }
)
READY_GATES = STATISTICAL_READY_GATES | GOVERNANCE_READY_GATES
MISSING_READY_GATE_REASONS = {
    "sufficient_train_sample": "insufficient_train_sample",
    "sufficient_validation_sample": "insufficient_validation_sample",
    "sufficient_validation_coverage": "insufficient_validation_coverage",
    "training_expectancy_positive": "training_expectancy_not_positive",
    "training_profit_factor_at_least_one": "training_profit_factor_below_one",
    "validation_expectancy_gate_passed": "validation_expectancy_gate_failed",
    "validation_profit_factor_gate_passed": "validation_profit_factor_gate_failed",
    "validation_drawdown_gate_passed": "validation_drawdown_gate_failed",
    "immutable_candidate_registered": "immutable_registration_required",
    "post_registration_evidence": "post_registration_evidence_required",
    "frozen_referee": "frozen_referee_required",
    "false_discovery_control_passed": "false_discovery_control_required",
    "minimum_trade_count_passed": "minimum_trade_count_required",
    "minimum_pair_coverage_passed": "minimum_pair_coverage_required",
    "forward_shadow_passed": "forward_shadow_required",
}

READ_ONLY_ANNOTATIONS = ToolAnnotations(
    read_only_hint=True,
    destructive_hint=False,
    idempotent_hint=True,
    open_world_hint=False,
)


class RuntimeHeartbeat(BaseModel):
    """Sanitised worker telemetry. No account ID, equity, orders, or secrets."""

    model_config = ConfigDict(extra="forbid")

    observed_at: datetime
    service_status: Literal["starting", "running", "broker-unavailable"]
    mode: str = Field(min_length=1, max_length=16)
    oanda_environment: str = Field(min_length=1, max_length=16)
    scheduler_alive: bool
    decision_cycle_fresh: bool
    broker_sync_fresh: bool
    has_open_trades: bool | None = None
    supervisor_floor_breached: bool
    entry_window_state: Literal[
        "weekend_locked", "in_configured_session", "off_session"
    ]
    last_verified_entry_age_bucket: Literal[
        "never",
        "under_1h",
        "under_24h",
        "one_to_three_days",
        "over_three_days",
        "unknown",
    ]
    # An older worker has not reported whether its entry latch is clear.
    broker_entry_halted: bool | None = Field(default=None, strict=True)
    broker_entry_halt_reason: Literal[
        "account-currency-mismatch",
        "protective-stop-audit-unavailable",
        "unprotected-open-trade",
        "invalid-order-response",
        "no-confirmed-trade-opening",
        "fill-does-not-match-request",
        "protective-stop-not-confirmed",
        "order-http-state-uncertain",
        "order-transport-state-uncertain",
        "other",
    ] | None = None
    revision: str = Field(min_length=1, max_length=120)

    @model_validator(mode="after")
    def validate_broker_halt_state(self) -> "RuntimeHeartbeat":
        if self.broker_entry_halted is True:
            if self.broker_entry_halt_reason is None:
                raise ValueError("a confirmed broker halt requires a bounded reason")
        elif self.broker_entry_halt_reason is not None:
            raise ValueError("a broker halt reason requires a confirmed halt")
        return self


class StrictSnapshotModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Freshness(StrictSnapshotModel):
    state: Literal["fresh", "stale", "future", "unknown"]
    age_seconds: float | None = Field(default=None, ge=0, allow_inf_nan=False)


class AdaptiveTunerReview(StrictSnapshotModel):
    state: Literal["active", "collecting", "disabled", "error"]
    mode: Literal["reduce_only"]
    lookback_closed_trades: int = Field(ge=1, le=10_000)
    minimum_sample: int = Field(ge=1, le=10_000)
    sample_count: int = Field(ge=0, le=10_000_000)
    risk_multiplier: float | None = Field(
        default=None, ge=0.0, le=1.0, allow_inf_nan=False
    )
    reason_code: Literal[
        "small_sample",
        "loss_streak",
        "negative_expectancy",
        "profit_factor_below_one",
        "low_win_rate",
        "normal",
        "unavailable",
    ]


class SetupPolicyReview(StrictSnapshotModel):
    state: Literal["active", "disabled", "error"]
    mode: Literal["reduce_or_block_only"]


class ShadowLearningReview(StrictSnapshotModel):
    state: Literal["active", "collecting", "stale", "disabled", "error"]
    mode: Literal["advisory_only"]
    evidence_scope: Literal[
        "executed_trade_subsets", "prospective_shadow_observations"
    ]
    can_evaluate_more_trade_hypotheses: bool
    cohort_start_utc: datetime
    clean_trades: int = Field(ge=0, le=10_000_000)
    train_trades: int = Field(ge=0, le=10_000_000)
    validation_trades: int = Field(ge=0, le=10_000_000)
    minimum_train: int = Field(ge=1, le=10_000_000)
    minimum_validation: int = Field(ge=1, le=10_000_000)
    minimum_coverage: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    auto_apply: Literal[False]

    @model_validator(mode="after")
    def validate_evidence_scope(self) -> "ShadowLearningReview":
        prospective = self.evidence_scope == "prospective_shadow_observations"
        if self.can_evaluate_more_trade_hypotheses is not prospective:
            raise ValueError("shadow capability must match its evidence scope")
        if self.train_trades + self.validation_trades != self.clean_trades:
            raise ValueError("shadow split counts must equal clean trade count")
        return self


class ObservationCaptureReview(StrictSnapshotModel):
    state: Literal["active", "degraded"]
    mode: Literal["best_effort"]
    submitted: int = Field(ge=0, strict=True)
    persisted_events: int = Field(ge=0, strict=True)
    duplicate_events: int = Field(ge=0, strict=True)
    queue_drops: int = Field(ge=0, strict=True)
    write_errors: int = Field(ge=0, strict=True)
    pending_events: int | None = Field(default=None, ge=0, strict=True)

    @model_validator(mode="after")
    def validate_degraded_counters(self) -> "ObservationCaptureReview":
        if (self.queue_drops > 0 or self.write_errors > 0) and self.state != "degraded":
            raise ValueError("capture loss or write errors require degraded state")
        return self


class LearningStatus(StrictSnapshotModel):
    overall: Literal["active", "collecting", "degraded", "disabled"]
    adaptive_tuner: AdaptiveTunerReview
    setup_policy: SetupPolicyReview
    shadow_learning: ShadowLearningReview
    observation_capture: ObservationCaptureReview

    @model_validator(mode="after")
    def validate_capture_rollup(self) -> "LearningStatus":
        if (
            self.observation_capture.state == "degraded"
            and self.overall != "degraded"
        ):
            raise ValueError("degraded observation capture requires degraded learning")
        return self


class PairPerformance(StrictSnapshotModel):
    instrument: Literal["AUD_USD", "GBP_USD"]
    window: Literal["last_40_broker_confirmed_closed"]
    sample_count: int = Field(ge=0, le=40)
    wins: int = Field(ge=0, le=40)
    losses: int = Field(ge=0, le=40)
    flat: int = Field(ge=0, le=40)
    win_rate: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    closed_net_pnl_aud: float | None = Field(default=None, allow_inf_nan=False)
    expectancy_aud: float | None = Field(default=None, allow_inf_nan=False)
    profit_factor: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    profit_factor_state: Literal["finite", "no_losses", "no_sample"]
    max_drawdown_aud: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    evidence_status: Literal["sufficient", "small_sample", "no_sample"]

    @model_validator(mode="after")
    def validate_counts_and_empty_state(self) -> "PairPerformance":
        if self.wins + self.losses + self.flat != self.sample_count:
            raise ValueError("pair outcome counts must equal sample_count")
        expected_evidence = (
            "no_sample"
            if self.sample_count == 0
            else "sufficient"
            if self.sample_count == 40
            else "small_sample"
        )
        if self.evidence_status != expected_evidence:
            raise ValueError("pair evidence status does not match its bounded sample")
        if self.sample_count == 0:
            if self.evidence_status != "no_sample" or self.profit_factor_state != "no_sample":
                raise ValueError("zero-sample pair metrics must be marked no_sample")
            values = (
                self.win_rate,
                self.closed_net_pnl_aud,
                self.expectancy_aud,
                self.profit_factor,
                self.max_drawdown_aud,
            )
            if any(value is not None for value in values):
                raise ValueError("zero-sample pair metrics must be null")
        else:
            required = (
                self.win_rate,
                self.closed_net_pnl_aud,
                self.expectancy_aud,
                self.max_drawdown_aud,
            )
            if any(value is None for value in required):
                raise ValueError("aggregate pair metrics are required for samples")
            assert self.win_rate is not None
            assert self.closed_net_pnl_aud is not None
            assert self.expectancy_aud is not None
            if abs(self.win_rate - (self.wins / self.sample_count)) > 1e-9:
                raise ValueError("pair win_rate does not match its outcome counts")
            if (
                abs(
                    self.expectancy_aud
                    - (self.closed_net_pnl_aud / self.sample_count)
                )
                > 1e-9
            ):
                raise ValueError("pair expectancy does not match net P&L")
            expected_pf_state = "no_losses" if self.losses == 0 else "finite"
            if self.profit_factor_state != expected_pf_state:
                raise ValueError("pair profit factor state does not match losses")
        if self.profit_factor_state == "finite" and self.profit_factor is None:
            raise ValueError("finite profit factor must include a value")
        if self.profit_factor_state != "finite" and self.profit_factor is not None:
            raise ValueError("non-finite/no-sample profit factor must be null")
        return self


class EvaluationMetrics(StrictSnapshotModel):
    trades: int = Field(ge=0, le=10_000_000)
    win_rate: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    closed_net_pnl_aud: float | None = Field(default=None, allow_inf_nan=False)
    expectancy_aud: float | None = Field(default=None, allow_inf_nan=False)
    profit_factor: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    profit_factor_state: Literal["finite", "no_losses", "no_sample"]
    max_drawdown_aud: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_empty_state(self) -> "EvaluationMetrics":
        if self.trades == 0:
            values = (
                self.win_rate,
                self.closed_net_pnl_aud,
                self.expectancy_aud,
                self.profit_factor,
                self.max_drawdown_aud,
            )
            if self.profit_factor_state != "no_sample" or any(
                value is not None for value in values
            ):
                raise ValueError("zero-trade evaluation metrics must be null/no_sample")
        else:
            required = (
                self.win_rate,
                self.closed_net_pnl_aud,
                self.expectancy_aud,
                self.max_drawdown_aud,
            )
            if any(value is None for value in required):
                raise ValueError("aggregate evaluation metrics are required for trades")
            assert self.closed_net_pnl_aud is not None
            assert self.expectancy_aud is not None
            if (
                abs(
                    self.expectancy_aud
                    - (self.closed_net_pnl_aud / self.trades)
                )
                > 1e-9
            ):
                raise ValueError("evaluation expectancy does not match net P&L")
        if self.profit_factor_state == "finite" and self.profit_factor is None:
            raise ValueError("finite profit factor must include a value")
        if self.profit_factor_state != "finite" and self.profit_factor is not None:
            raise ValueError("non-finite/no-sample profit factor must be null")
        return self


class ChallengerCandidateReview(StrictSnapshotModel):
    candidate_id: CandidateId
    definition_version: Literal["shadow-filter.v1"]
    status: Literal[
        "ready_for_luke_review", "rejected", "collecting", "stale", "unavailable"
    ]
    train: EvaluationMetrics
    validation: EvaluationMetrics
    validation_coverage: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    passed_gates: list[PassedGate] = Field(default_factory=list, max_length=15)
    failed_gates: list[FailureReason] = Field(default_factory=list, max_length=22)


class ChallengerReadiness(StrictSnapshotModel):
    state: Literal[
        "ready_for_luke_review",
        "no_candidate_passed",
        "collecting_data",
        "stale_evidence",
        "unavailable",
    ]
    leading_candidate_id: CandidateId | None = None
    reason_codes: list[FailureReason] = Field(default_factory=list, max_length=22)
    next_allowed_action: Literal[
        "luke_review_only", "collect_more_evidence", "none"
    ]
    luke_approval_required: Literal[True]
    auto_promotion_permitted: Literal[False]

    @model_validator(mode="after")
    def validate_review_authority(self) -> "ChallengerReadiness":
        if self.state == "ready_for_luke_review":
            if self.leading_candidate_id is None:
                raise ValueError("ready review requires a registered leading candidate")
            if self.next_allowed_action != "luke_review_only":
                raise ValueError("ready review can only proceed to Luke review")
        else:
            if self.leading_candidate_id is not None:
                raise ValueError("non-ready review cannot name a leading candidate")
            if self.next_allowed_action == "luke_review_only":
                raise ValueError("non-ready review cannot request Luke review")
        return self


class ChallengerReview(StrictSnapshotModel):
    registry_version: Literal["mossy-shadow-registry.v1"]
    evidence_generated_at_utc: datetime
    evidence_revision: str = Field(
        min_length=7, max_length=64, pattern=r"^(?:[0-9a-f]{7,64}|unknown)$"
    )
    baseline_validation: EvaluationMetrics
    candidates: list[ChallengerCandidateReview] = Field(
        default_factory=list, max_length=12
    )
    readiness: ChallengerReadiness


class LearningGovernance(StrictSnapshotModel):
    read_only: Literal[True]
    demo_only: Literal[True]
    can_trade: Literal[False]
    can_write_configuration: Literal[False]
    can_run_optimisation: Literal[False]
    can_deploy: Literal[False]
    can_promote: Literal[False]


class LearningReviewSnapshot(StrictSnapshotModel):
    schema_version: Literal["mossy.learning-review.v1"]
    observed_at_utc: datetime
    source_revision: str = Field(
        min_length=7, max_length=64, pattern=r"^(?:[0-9a-f]{7,64}|unknown)$"
    )
    freshness: Freshness
    learning_status: LearningStatus
    pair_performance: list[PairPerformance] = Field(min_length=2, max_length=2)
    challenger_review: ChallengerReview
    governance: LearningGovernance

    @model_validator(mode="after")
    def validate_unique_registered_evidence(self) -> "LearningReviewSnapshot":
        timestamps = (
            self.observed_at_utc,
            self.learning_status.shadow_learning.cohort_start_utc,
            self.challenger_review.evidence_generated_at_utc,
        )
        if any(value.tzinfo is None for value in timestamps):
            raise ValueError("learning timestamps must include a timezone")
        observed = self.observed_at_utc.astimezone(timezone.utc)
        cohort = self.learning_status.shadow_learning.cohort_start_utc.astimezone(
            timezone.utc
        )
        evidence_generated = (
            self.challenger_review.evidence_generated_at_utc.astimezone(timezone.utc)
        )
        if cohort > observed:
            raise ValueError("learning cohort cannot start after its observation")
        if (evidence_generated - observed).total_seconds() > FUTURE_TOLERANCE_SECONDS:
            raise ValueError("challenger evidence cannot postdate its observation")
        instruments = [item.instrument for item in self.pair_performance]
        if set(instruments) != {"AUD_USD", "GBP_USD"}:
            raise ValueError("pair evidence must contain AUD_USD and GBP_USD exactly once")
        candidates = [item.candidate_id for item in self.challenger_review.candidates]
        if len(candidates) != len(set(candidates)):
            raise ValueError("candidate evidence must not contain duplicates")
        leading = self.challenger_review.readiness.leading_candidate_id
        if leading is not None and leading not in candidates:
            raise ValueError("leading candidate must be present in candidate evidence")
        return self


mcp = MCPServer(
    "Mossy 4X Read-Only Monitor",
    version="1.0.0",
    instructions=(
        "Read-only supervision for Mossy 4X. This server cannot place or close "
        "orders, change strategy parameters, run optimisation, write GitHub, or deploy."
    ),
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _bearer_token(request: Request) -> str:
    header = request.headers.get("authorization", "")
    scheme, separator, value = header.partition(" ")
    if not separator or scheme.lower() != "bearer":
        return ""
    return value.strip()


def _runtime_supervision(snapshot: dict | None) -> dict:
    if snapshot is None:
        return {
            "bridge_status": "ok",
            "telemetry_received": False,
            "telemetry_fresh": False,
            "supervisor_status": "BLOCKED",
            "blockers": ["No worker heartbeat has been received."],
            "read_only": True,
        }

    try:
        received_at = datetime.fromisoformat(snapshot["received_at"])
        payload_model = RuntimeHeartbeat.model_validate(snapshot["payload"])
        observed_at = payload_model.observed_at
        if received_at.tzinfo is None:
            received_at = received_at.replace(tzinfo=timezone.utc)
        if observed_at.tzinfo is None:
            observed_at = observed_at.replace(tzinfo=timezone.utc)
        received_at = received_at.astimezone(timezone.utc)
        observed_at = observed_at.astimezone(timezone.utc)
        payload = payload_model.model_dump(mode="json")
    except Exception:
        return {
            "bridge_status": "degraded",
            "telemetry_received": True,
            "telemetry_fresh": False,
            "supervisor_status": "BLOCKED",
            "blockers": ["Stored worker telemetry is invalid."],
            "read_only": True,
        }

    now = _utc_now()
    heartbeat_age_seconds = max(0.0, (now - observed_at).total_seconds())
    transport_age_seconds = max(0.0, (now - received_at).total_seconds())
    telemetry_fresh = (
        -60.0 <= (now - observed_at).total_seconds() <= STATUS_STALE_SECONDS
        and -60.0 <= (now - received_at).total_seconds() <= STATUS_STALE_SECONDS
    )

    blockers: list[str] = []
    if not telemetry_fresh:
        blockers.append("Worker telemetry is stale.")
    if payload["service_status"] != "running":
        blockers.append(f"Worker status is {payload['service_status']}.")
    if payload["mode"].lower() != "demo":
        blockers.append("MODE is not demo.")
    if payload["oanda_environment"].lower() != "practice":
        blockers.append("OANDA environment is not practice.")
    if not payload["scheduler_alive"]:
        blockers.append("Scheduler is not alive.")
    if not payload["decision_cycle_fresh"]:
        blockers.append("Decision cycle is stale.")
    if not payload["broker_sync_fresh"]:
        blockers.append("Broker sync is stale.")
    if payload["broker_entry_halted"] is None:
        blockers.append("Broker entry halt state is unknown; worker telemetry is incomplete.")
    elif payload["broker_entry_halted"]:
        blockers.append(
            f"Broker entries are halted: {payload['broker_entry_halt_reason']}."
        )
    if payload["supervisor_floor_breached"]:
        blockers.append(
            f"The AUD {SUPERVISOR_EQUITY_FLOOR_AUD:,.0f} supervisory equity floor is breached."
        )

    return {
        "bridge_status": "ok",
        "telemetry_received": True,
        "telemetry_fresh": telemetry_fresh,
        "heartbeat_age_seconds": round(heartbeat_age_seconds, 1),
        "transport_age_seconds": round(transport_age_seconds, 1),
        "last_observed_utc": observed_at.isoformat(),
        "last_heartbeat_received_utc": received_at.isoformat(),
        "runtime": payload,
        "supervisor_status": "BLOCKED" if blockers else "READY_FOR_REVIEW",
        "blockers": blockers,
        "read_only": True,
        "authority_note": (
            "READY_FOR_REVIEW is not permission to trade. The deterministic Mossy 4X "
            "algorithm remains the only execution path, and Luke retains change authority."
        ),
    }


def _learning_governance() -> dict[str, bool]:
    """Return bridge-owned authority flags; publisher values never widen authority."""

    return {
        "read_only": True,
        "demo_only": True,
        "can_trade": False,
        "can_write_configuration": False,
        "can_run_optimisation": False,
        "can_deploy": False,
        "can_promote": False,
    }


def _unavailable_learning_review(reason_code: FailureReason) -> dict[str, Any]:
    """Describe an empty/lost cache without inventing learning evidence."""

    return {
        "schema_version": LEARNING_SCHEMA_VERSION,
        "observed_at_utc": UNKNOWN_EVIDENCE_TIME,
        "source_revision": "unknown",
        "freshness": {"state": "unknown", "age_seconds": None},
        "learning_status": {
            "overall": "degraded",
            "adaptive_tuner": {
                "state": "error",
                "mode": "reduce_only",
                "lookback_closed_trades": 80,
                "minimum_sample": 20,
                "sample_count": 0,
                "risk_multiplier": None,
                "reason_code": "unavailable",
            },
            "setup_policy": {
                "state": "error",
                "mode": "reduce_or_block_only",
            },
            "shadow_learning": {
                "state": "error",
                "mode": "advisory_only",
                "evidence_scope": "executed_trade_subsets",
                "can_evaluate_more_trade_hypotheses": False,
                "cohort_start_utc": UNKNOWN_EVIDENCE_TIME,
                "clean_trades": 0,
                "train_trades": 0,
                "validation_trades": 0,
                "minimum_train": 50,
                "minimum_validation": 30,
                "minimum_coverage": 0.5,
                "auto_apply": False,
            },
            "observation_capture": {
                "state": "degraded",
                "mode": "best_effort",
                "submitted": 0,
                "persisted_events": 0,
                "duplicate_events": 0,
                "queue_drops": 0,
                "write_errors": 0,
                "pending_events": None,
            },
        },
        "pair_performance": [
            {
                "instrument": instrument,
                "window": "last_40_broker_confirmed_closed",
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
            for instrument in ("AUD_USD", "GBP_USD")
        ],
        "challenger_review": {
            "registry_version": LEARNING_REGISTRY_VERSION,
            "evidence_generated_at_utc": UNKNOWN_EVIDENCE_TIME,
            "evidence_revision": "unknown",
            "baseline_validation": {
                "trades": 0,
                "win_rate": None,
                "closed_net_pnl_aud": None,
                "expectancy_aud": None,
                "profit_factor": None,
                "profit_factor_state": "no_sample",
                "max_drawdown_aud": None,
            },
            "candidates": [],
            "readiness": {
                "state": "unavailable",
                "leading_candidate_id": None,
                "reason_codes": [reason_code],
                "next_allowed_action": "none",
                "luke_approval_required": True,
                "auto_promotion_permitted": False,
            },
        },
        "governance": _learning_governance(),
    }


def _restrict_invalid_or_stale_evidence(
    payload: dict[str, Any], *, freshness_state: Literal["stale", "future"]
) -> None:
    """Prevent old or future-dated evidence from surfacing as review-ready."""

    payload["learning_status"]["overall"] = "degraded"
    shadow = payload["learning_status"]["shadow_learning"]
    if shadow["state"] not in {"disabled", "error"}:
        shadow["state"] = "stale" if freshness_state == "stale" else "error"
    for candidate in payload["challenger_review"]["candidates"]:
        if candidate["status"] != "unavailable":
            candidate["status"] = (
                "stale" if freshness_state == "stale" else "unavailable"
            )
        failed_gates = candidate["failed_gates"]
        reason = "evidence_stale" if freshness_state == "stale" else "report_invalid"
        if reason not in failed_gates:
            failed_gates.append(reason)
    readiness = payload["challenger_review"]["readiness"]
    if readiness["state"] == "unavailable":
        readiness.update(
            {
                "leading_candidate_id": None,
                "next_allowed_action": "none",
                "luke_approval_required": True,
                "auto_promotion_permitted": False,
            }
        )
        return
    readiness.update(
        {
            "state": "stale_evidence" if freshness_state == "stale" else "unavailable",
            "leading_candidate_id": None,
            "reason_codes": [
                "evidence_stale" if freshness_state == "stale" else "report_invalid"
            ],
            "next_allowed_action": "none",
            "luke_approval_required": True,
            "auto_promotion_permitted": False,
        }
    )


def _runtime_revision(snapshot: dict | None) -> str | None:
    if snapshot is None:
        return None
    try:
        heartbeat = RuntimeHeartbeat.model_validate(snapshot["payload"])
        received_at = datetime.fromisoformat(str(snapshot["received_at"]))
    except Exception:
        return None
    observed_at = heartbeat.observed_at
    if observed_at.tzinfo is None or received_at.tzinfo is None:
        return None
    now = _utc_now()
    observed_age = (now - observed_at.astimezone(timezone.utc)).total_seconds()
    received_age = (now - received_at.astimezone(timezone.utc)).total_seconds()
    if not (
        -FUTURE_TOLERANCE_SECONDS <= observed_age <= STATUS_STALE_SECONDS
        and -FUTURE_TOLERANCE_SECONDS <= received_age <= STATUS_STALE_SECONDS
    ):
        return None
    revision = heartbeat.revision.strip().lower()
    if not 7 <= len(revision) <= 64:
        return None
    if any(character not in "0123456789abcdef" for character in revision):
        return None
    return revision


def _enforce_learning_provenance(
    payload: dict[str, Any], runtime_snapshot: dict | None
) -> bool:
    """Require one known revision across worker, evidence, and runtime heartbeat."""

    source_revision = payload["source_revision"]
    evidence_revision = payload["challenger_review"]["evidence_revision"]
    runtime_revision = _runtime_revision(runtime_snapshot)
    revisions_match = (
        source_revision != "unknown"
        and evidence_revision != "unknown"
        and runtime_revision is not None
        and source_revision == evidence_revision == runtime_revision
    )
    if revisions_match:
        return True

    payload["learning_status"]["overall"] = "degraded"
    shadow = payload["learning_status"]["shadow_learning"]
    if shadow["state"] != "disabled":
        shadow["state"] = "error"
    for candidate in payload["challenger_review"]["candidates"]:
        candidate["status"] = "unavailable"
        if "revision_mismatch" not in candidate["failed_gates"]:
            candidate["failed_gates"].append("revision_mismatch")
    payload["challenger_review"]["readiness"].update(
        {
            "state": "unavailable",
            "leading_candidate_id": None,
            "reason_codes": ["revision_mismatch"],
            "next_allowed_action": "none",
            "luke_approval_required": True,
            "auto_promotion_permitted": False,
        }
    )
    return False


def _enforce_ready_governance(payload: dict[str, Any]) -> None:
    """Require prospective, registered, forward-shadow proof for any READY claim."""

    scope = payload["learning_status"]["shadow_learning"]["evidence_scope"]
    shadow = payload["learning_status"]["shadow_learning"]
    baseline = payload["challenger_review"]["baseline_validation"]
    registry_version = payload["challenger_review"]["registry_version"]

    def comparable_profit_factor(metrics: dict[str, Any]) -> float:
        if metrics["profit_factor_state"] == "no_losses":
            return (
                float("inf")
                if (metrics["closed_net_pnl_aud"] or 0.0) > 0.0
                else 0.0
            )
        return float(metrics["profit_factor"] or 0.0)

    def verified_gates(candidate: dict[str, Any]) -> dict[str, bool]:
        train = candidate["train"]
        validation = candidate["validation"]
        baseline_expectancy = baseline["expectancy_aud"]
        validation_expectancy = validation["expectancy_aud"]
        expectancy_margin = (
            max(0.05, abs(baseline_expectancy) * 0.15)
            if baseline_expectancy is not None
            else float("inf")
        )
        passed = set(candidate["passed_gates"])
        return {
            "sufficient_train_sample": train["trades"] >= shadow["minimum_train"],
            "sufficient_validation_sample": (
                validation["trades"] >= shadow["minimum_validation"]
            ),
            "sufficient_validation_coverage": (
                candidate["validation_coverage"] >= shadow["minimum_coverage"]
            ),
            "training_expectancy_positive": (
                train["expectancy_aud"] is not None
                and train["expectancy_aud"] > 0.0
            ),
            "training_profit_factor_at_least_one": (
                comparable_profit_factor(train) >= 1.0
            ),
            "validation_expectancy_gate_passed": (
                validation_expectancy is not None
                and baseline_expectancy is not None
                and validation_expectancy
                >= baseline_expectancy + expectancy_margin
            ),
            "validation_profit_factor_gate_passed": (
                comparable_profit_factor(validation)
                >= max(1.10, comparable_profit_factor(baseline))
            ),
            "validation_drawdown_gate_passed": (
                validation["max_drawdown_aud"] is not None
                and baseline["max_drawdown_aud"] is not None
                and validation["max_drawdown_aud"]
                <= baseline["max_drawdown_aud"]
            ),
            "immutable_candidate_registered": (
                registry_version == LEARNING_REGISTRY_VERSION
                and candidate["definition_version"] == LEARNING_DEFINITION_VERSION
            ),
            "post_registration_evidence": "post_registration_evidence" in passed,
            "frozen_referee": "frozen_referee" in passed,
            "false_discovery_control_passed": (
                "false_discovery_control_passed" in passed
            ),
            "minimum_trade_count_passed": (
                shadow["clean_trades"] >= 250 and validation["trades"] >= 100
            ),
            "minimum_pair_coverage_passed": (
                "minimum_pair_coverage_passed" in passed
            ),
            "forward_shadow_passed": (
                scope == "prospective_shadow_observations"
                and "forward_shadow_passed" in passed
            ),
        }

    unavailable_reasons = {
        "unregistered_candidate",
        "revision_mismatch",
        "journal_unavailable",
        "report_invalid",
        "promotion_protocol_not_implemented",
    }
    quality_failure_reasons = {
        "training_expectancy_not_positive",
        "training_profit_factor_below_one",
        "validation_expectancy_gate_failed",
        "validation_profit_factor_gate_failed",
        "validation_drawdown_gate_failed",
    }

    def restricted_status(reasons: list[str]) -> tuple[str, str, str]:
        reason_set = set(reasons)
        if reason_set & unavailable_reasons:
            return "unavailable", "unavailable", "none"
        if "evidence_stale" in reason_set:
            return "stale", "stale_evidence", "none"
        if reason_set & quality_failure_reasons:
            return "rejected", "no_candidate_passed", "none"
        return "collecting", "collecting_data", "collect_more_evidence"

    downgraded: dict[str, list[str]] = {}
    candidates = payload["challenger_review"]["candidates"]
    for candidate in candidates:
        if candidate["status"] != "ready_for_luke_review":
            continue
        passed = set(candidate["passed_gates"])
        reasons: list[str] = []
        if scope != "prospective_shadow_observations":
            reasons.extend(
                ["prospective_validation_required", "forward_shadow_required"]
            )
        train_trades = candidate["train"]["trades"]
        validation_trades = candidate["validation"]["trades"]
        validation_population = shadow["validation_trades"]
        expected_coverage = (
            validation_trades / validation_population
            if validation_population > 0
            else 0.0
        )
        if (
            baseline["trades"] != validation_population
            or train_trades > shadow["train_trades"]
            or validation_trades > validation_population
            or abs(candidate["validation_coverage"] - expected_coverage) > 1e-9
        ):
            reasons.append("report_invalid")
        checks = verified_gates(candidate)
        missing_gates = {
            gate for gate in READY_GATES if gate not in passed or not checks[gate]
        }
        reasons.extend(
            MISSING_READY_GATE_REASONS[gate]
            for gate in sorted(missing_gates)
        )
        reasons.extend(candidate["failed_gates"])
        # v1 can report evidence, but no independently implemented promotion
        # protocol exists yet. A worker READY label is therefore non-actionable.
        reasons.append("promotion_protocol_not_implemented")
        reasons = list(dict.fromkeys(reasons))
        if reasons:
            candidate["status"] = restricted_status(reasons)[0]
            candidate["failed_gates"] = reasons
            downgraded[candidate["candidate_id"]] = reasons

    readiness = payload["challenger_review"]["readiness"]
    if readiness["state"] != "ready_for_luke_review":
        for candidate in candidates:
            if candidate["status"] == "ready_for_luke_review":
                candidate["status"] = "unavailable"
                candidate["failed_gates"] = list(
                    dict.fromkeys(candidate["failed_gates"] + ["report_invalid"])
                )
        return
    leading = readiness["leading_candidate_id"]
    leading_evidence = next(
        (candidate for candidate in candidates if candidate["candidate_id"] == leading),
        None,
    )
    reasons = list(downgraded.get(leading, []))
    if leading_evidence is None or leading_evidence["status"] != "ready_for_luke_review":
        if not reasons:
            reasons.append("report_invalid")
    if readiness["reason_codes"]:
        reasons.extend(readiness["reason_codes"])
    reasons.append("promotion_protocol_not_implemented")
    reasons = list(dict.fromkeys(reasons))
    if reasons:
        _, readiness_state, next_action = restricted_status(reasons)
        readiness.update(
            {
                "state": readiness_state,
                "leading_candidate_id": None,
                "reason_codes": reasons,
                "next_allowed_action": next_action,
                "luke_approval_required": True,
                "auto_promotion_permitted": False,
            }
        )


def _learning_review(
    snapshot: dict | None, runtime_snapshot: dict | None = None
) -> dict[str, Any]:
    if snapshot is None:
        return _unavailable_learning_review("journal_unavailable")

    try:
        model = LearningReviewSnapshot.model_validate(snapshot["payload"])
        observed_at = model.observed_at_utc.astimezone(timezone.utc)
        evidence_generated_at = (
            model.challenger_review.evidence_generated_at_utc.astimezone(timezone.utc)
        )
        payload = model.model_dump(mode="json")
    except Exception:
        return _unavailable_learning_review("report_invalid")

    now = _utc_now()
    signed_ages = (
        (now - observed_at).total_seconds(),
        (now - evidence_generated_at).total_seconds(),
    )
    if min(signed_ages) < -FUTURE_TOLERANCE_SECONDS:
        freshness_state: Literal["fresh", "stale", "future"] = "future"
        age_seconds = max(abs(value) for value in signed_ages if value < 0.0)
    elif max(signed_ages) > LEARNING_STALE_SECONDS:
        freshness_state = "stale"
        age_seconds = max(signed_ages)
    else:
        freshness_state = "fresh"
        age_seconds = max(0.0, *signed_ages)

    payload["observed_at_utc"] = observed_at.isoformat()
    payload["freshness"] = {
        "state": freshness_state,
        "age_seconds": round(age_seconds, 1),
    }
    payload["governance"] = _learning_governance()
    payload["learning_status"]["shadow_learning"]["auto_apply"] = False
    readiness = payload["challenger_review"]["readiness"]
    readiness["luke_approval_required"] = True
    readiness["auto_promotion_permitted"] = False

    if freshness_state in {"stale", "future"}:
        _restrict_invalid_or_stale_evidence(
            payload,
            freshness_state=freshness_state,
        )
    _enforce_learning_provenance(payload, runtime_snapshot)
    _enforce_ready_governance(payload)
    return payload


@mcp.tool(
    title="Get Mossy runtime health",
    description=(
        "Return the latest sanitised Mossy 4X worker heartbeat and supervisory blockers. "
        "This is read-only and cannot affect trading."
    ),
    annotations=READ_ONLY_ANNOTATIONS,
    structured_output=True,
)
def get_runtime_health() -> dict[str, Any]:
    """Inspect sanitised runtime health without reading credentials or account values."""

    return _runtime_supervision(load_runtime_heartbeat())


@mcp.tool(
    title="Get Mossy learning review",
    description=(
        "Return the latest precomputed, sanitised Mossy 4X learning review. "
        "This reads only the bridge cache; it cannot query the broker, run analysis, "
        "change configuration, promote a Challenger, deploy, or trade."
    ),
    annotations=READ_ONLY_ANNOTATIONS,
    structured_output=True,
)
def get_learning_review() -> dict[str, Any]:
    """Read the latest learning evidence without invoking worker-side computation."""

    return _learning_review(load_learning_snapshot(), load_runtime_heartbeat())


@mcp.tool(
    title="Get Mossy guardrails",
    description="Return the non-negotiable Mossy 4X supervisory safety contract.",
    annotations=READ_ONLY_ANNOTATIONS,
    structured_output=True,
)
def get_algo_guardrails() -> dict[str, Any]:
    """Return the fixed advisory and execution boundaries for the Mossy agent."""

    return {
        "execution_authority": "Deterministic Mossy 4X algorithm only",
        "agent_role": "Read-only supervisor and analyst",
        "allowed_environment": {"mode": "demo", "oanda": "practice"},
        "live_trading_permitted": False,
        "supervisor_equity_floor_aud": SUPERVISOR_EQUITY_FLOOR_AUD,
        "normal_risk_limit_pct": 1.0,
        "absolute_risk_limit_pct": 2.0,
        "requires_luke_approval": [
            "strategy or parameter changes",
            "Champion changes",
            "deployment",
            "spend",
            "any live-trading enablement",
        ],
        "mcp_capabilities": [
            "read sanitised runtime health",
            "read the latest sanitised learning review",
            "read this safety contract",
            "read the latest public algorithm report",
        ],
        "mcp_forbidden_capabilities": [
            "place, amend, or close orders",
            "run or apply optimisation",
            "write GitHub or secrets",
            "trigger a deploy",
            "change strategy configuration",
        ],
    }


@mcp.tool(
    title="Get latest Mossy algorithm report",
    description=(
        "Read the newest public weekly algorithm report committed in the repository. "
        "Reports are research context only and never auto-apply changes."
    ),
    annotations=READ_ONLY_ANNOTATIONS,
    structured_output=True,
)
def get_latest_algo_report() -> dict[str, Any]:
    """Return the latest committed weekly report, bounded to a safe text size."""

    report_dir = REPOSITORY_ROOT / "reports" / "algo-weekly"
    reports = sorted(report_dir.glob("*.md"), reverse=True)
    if not reports:
        return {"available": False, "reason": "No weekly report is committed."}

    report_path = reports[0]
    text = report_path.read_text(encoding="utf-8")
    limit = 30_000
    try:
        report_date = date.fromisoformat(report_path.stem)
        report_age_days = max(0, (_utc_now().date() - report_date).days)
    except ValueError:
        report_date = None
        report_age_days = None
    stale = report_age_days is None or report_age_days > 14
    return {
        "available": True,
        "report_path": str(report_path.relative_to(REPOSITORY_ROOT)),
        "report": text[:limit],
        "truncated": len(text) > limit,
        "report_date": report_date.isoformat() if report_date else None,
        "report_age_days": report_age_days,
        "stale": stale,
        "freshness_warning": (
            "This report is stale and must not be treated as current evidence."
            if stale
            else None
        ),
        "governance": "Research context only; no automatic strategy changes.",
    }


@mcp.custom_route("/health", methods=["GET"])
async def health(_: Request) -> Response:
    key_configured = bool(os.getenv(STATUS_KEY_ENV, "").strip())
    snapshot = load_runtime_heartbeat()
    supervision = _runtime_supervision(snapshot)
    return JSONResponse(
        {
            "status": "ok" if key_configured else "degraded",
            "read_only": True,
            "status_key_configured": key_configured,
            "telemetry_received": supervision["telemetry_received"],
            "telemetry_fresh": supervision["telemetry_fresh"],
        },
        status_code=200 if key_configured else 503,
    )


@mcp.custom_route("/internal/runtime-heartbeat", methods=["GET"])
async def runtime_heartbeat_capabilities(request: Request) -> Response:
    expected = os.getenv(STATUS_KEY_ENV, "").strip()
    if not expected:
        return JSONResponse(
            {"detail": f"Server missing {STATUS_KEY_ENV}."}, status_code=503
        )
    supplied = _bearer_token(request)
    if not supplied or not secrets.compare_digest(supplied, expected):
        return JSONResponse({"detail": "Unauthorized."}, status_code=401)
    return JSONResponse({
        "optional_heartbeat_fields": [
            "broker_entry_halted", "broker_entry_halt_reason",
        ],
    })


@mcp.custom_route("/internal/runtime-heartbeat", methods=["POST"])
async def ingest_runtime_heartbeat(request: Request) -> Response:
    expected = os.getenv(STATUS_KEY_ENV, "").strip()
    if not expected:
        return JSONResponse(
            {"detail": f"Server missing {STATUS_KEY_ENV}."}, status_code=503
        )
    supplied = _bearer_token(request)
    if not supplied or not secrets.compare_digest(supplied, expected):
        return JSONResponse({"detail": "Unauthorized."}, status_code=401)

    try:
        payload = RuntimeHeartbeat.model_validate(await request.json())
    except Exception:
        return JSONResponse({"detail": "Invalid heartbeat payload."}, status_code=422)

    observed_at = payload.observed_at
    if observed_at.tzinfo is None:
        observed_at = observed_at.replace(tzinfo=timezone.utc)
    observed_age = (_utc_now() - observed_at).total_seconds()
    if observed_age < -60 or observed_age > STATUS_STALE_SECONDS:
        return JSONResponse({"detail": "Heartbeat timestamp is stale."}, status_code=422)

    payload_json = payload.model_dump(mode="json")
    payload_json["observed_at"] = observed_at.astimezone(timezone.utc).isoformat()
    stored = save_runtime_heartbeat(payload_json)
    return JSONResponse(
        {"accepted": True, "stored": stored},
        status_code=202,
    )


@mcp.custom_route("/internal/learning-snapshot", methods=["POST"])
async def ingest_learning_snapshot(request: Request) -> Response:
    expected = os.getenv(STATUS_KEY_ENV, "").strip()
    if not expected:
        return JSONResponse(
            {"detail": f"Server missing {STATUS_KEY_ENV}."}, status_code=503
        )
    supplied = _bearer_token(request)
    if not supplied or not secrets.compare_digest(supplied, expected):
        return JSONResponse({"detail": "Unauthorized."}, status_code=401)

    try:
        payload = LearningReviewSnapshot.model_validate(await request.json())
    except Exception:
        return JSONResponse(
            {"detail": "Invalid learning snapshot payload."}, status_code=422
        )

    observed_at = payload.observed_at_utc.astimezone(timezone.utc)
    if (_utc_now() - observed_at).total_seconds() < -FUTURE_TOLERANCE_SECONDS:
        return JSONResponse(
            {"detail": "Learning snapshot timestamp is in the future."},
            status_code=422,
        )

    payload_json = payload.model_dump(mode="json")
    payload_json["observed_at_utc"] = observed_at.isoformat()
    payload_json["freshness"] = {"state": "fresh", "age_seconds": 0.0}
    payload_json["governance"] = _learning_governance()
    payload_json["learning_status"]["shadow_learning"]["auto_apply"] = False
    readiness = payload_json["challenger_review"]["readiness"]
    readiness["luke_approval_required"] = True
    readiness["auto_promotion_permitted"] = False
    stored = save_learning_snapshot(payload_json)
    return JSONResponse({"accepted": True, "stored": stored}, status_code=202)


def _transport_security() -> TransportSecuritySettings:
    render_host = os.getenv("RENDER_EXTERNAL_HOSTNAME", "").strip()
    allowed_hosts = ["127.0.0.1", "127.0.0.1:*", "localhost", "localhost:*"]
    if render_host:
        allowed_hosts.extend([render_host, f"{render_host}:*"])
    return TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=allowed_hosts,
        allowed_origins=[
            "http://127.0.0.1:*",
            "http://localhost:*",
            "https://platform.openai.com",
            "https://chatgpt.com",
        ],
    )


app = mcp.streamable_http_app(
    json_response=True,
    stateless_http=True,
    transport_security=_transport_security(),
)
