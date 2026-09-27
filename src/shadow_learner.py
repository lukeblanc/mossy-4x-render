from __future__ import annotations

import json
import math
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Optional

from src.learning_cohort import (
    DEFAULT_LEARNING_COHORT_START_UTC,
    DEFAULT_LEARNING_RUN_TAG,
    load_clean_outcomes,
)


CLEAN_COHORT_START_UTC = DEFAULT_LEARNING_COHORT_START_UTC
ALLOWED_INSTRUMENTS = ("AUD_USD", "GBP_USD")
_SAFE_REVISION = re.compile(r"^[0-9a-f]{7,64}$")


@dataclass(frozen=True)
class ShadowMetrics:
    trades: int
    wins: int
    losses: int
    win_rate: float
    net_profit: float
    expectancy: float
    profit_factor: float
    max_drawdown: float


@dataclass(frozen=True)
class ShadowCandidateResult:
    name: str
    description: str
    train: ShadowMetrics
    validation: ShadowMetrics
    validation_coverage: float
    eligible: bool
    beats_baseline: bool
    reason: str


@dataclass(frozen=True)
class ShadowReport:
    generated_utc: str
    evidence_revision: str
    cohort_start_utc: str
    cohort_run_tag: str
    instruments: tuple[str, ...]
    total_clean_trades: int
    train_trades: int
    validation_trades: int
    baseline: ShadowCandidateResult
    candidates: tuple[ShadowCandidateResult, ...]
    recommendation: Optional[str]
    recommendation_reason: str
    auto_apply: bool = False


@dataclass(frozen=True)
class _Trade:
    timestamp: datetime
    instrument: str
    side: str
    session: str
    rsi: Optional[float]
    trend: str
    pnl: float


def _deployed_revision() -> str:
    """Return deployment-attested source provenance or an unusable sentinel."""

    candidate = str(
        os.getenv("RENDER_GIT_COMMIT") or os.getenv("GIT_COMMIT") or ""
    ).strip().lower()
    return candidate if _SAFE_REVISION.fullmatch(candidate) else "unknown"


def _parse_dt(value: object) -> Optional[datetime]:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _safe_float(value: object) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _session_bucket(value: object) -> str:
    label = str(value or "").upper()
    if label.startswith("LONDON"):
        return "LONDON"
    if label.startswith("NEWYORK") or label.startswith("NEW_YORK"):
        return "NEWYORK"
    if label.startswith("ADHOC"):
        return "ADHOC"
    return "OFF_SESSION"


def _trend_bucket(side: str, indicators: dict[str, object]) -> str:
    fast = _safe_float(indicators.get("ema_fast"))
    slow = _safe_float(indicators.get("ema_slow"))
    trend_fast = _safe_float(indicators.get("ema50"))
    trend_slow = _safe_float(indicators.get("ema200"))
    short_aligned = False
    long_aligned = False
    if fast is not None and slow is not None:
        short_aligned = fast > slow if side == "BUY" else fast < slow
    if trend_fast is not None and trend_slow is not None:
        long_aligned = trend_fast > trend_slow if side == "BUY" else trend_fast < trend_slow
    if short_aligned and long_aligned:
        return "FULL"
    if short_aligned or long_aligned:
        return "PARTIAL"
    return "COUNTER"


def _load_clean_trades(
    db_path: Path,
    *,
    cohort_start_utc: str,
    cohort_run_tag: str,
    instruments: tuple[str, ...],
) -> list[_Trade]:
    trades: list[_Trade] = []
    outcomes = load_clean_outcomes(
        db_path,
        instruments=instruments,
        run_tag=cohort_run_tag,
        entry_start_utc=cohort_start_utc,
    )
    for outcome in outcomes:
        # Order chronologically by when the result became knowable, not entry
        # time, so a future shadow evaluator cannot learn from open positions.
        timestamp = _parse_dt(outcome.exit_timestamp_utc)
        pnl = _safe_float(outcome.realized_pnl_ccy)
        if timestamp is None or pnl is None:
            continue
        side = outcome.side
        if side not in {"BUY", "SELL"}:
            continue
        try:
            indicators = json.loads(outcome.indicators_snapshot or "{}")
            if not isinstance(indicators, dict):
                indicators = {}
        except (TypeError, json.JSONDecodeError):
            indicators = {}
        trades.append(
            _Trade(
                timestamp=timestamp,
                instrument=outcome.instrument,
                side=side,
                session=_session_bucket(outcome.session_id),
                rsi=_safe_float(indicators.get("rsi")),
                trend=_trend_bucket(side, indicators),
                pnl=pnl,
            )
        )
    return trades


def _metrics(trades: Iterable[_Trade]) -> ShadowMetrics:
    items = list(trades)
    pnl = [trade.pnl for trade in items]
    wins = [value for value in pnl if value > 0]
    losses = [value for value in pnl if value < 0]
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (99.0 if gross_profit > 0 else 0.0)
    equity = 0.0
    peak = 0.0
    drawdown = 0.0
    for value in pnl:
        equity += value
        peak = max(peak, equity)
        drawdown = max(drawdown, peak - equity)
    count = len(items)
    return ShadowMetrics(
        trades=count,
        wins=len(wins),
        losses=len(losses),
        win_rate=len(wins) / count if count else 0.0,
        net_profit=sum(pnl),
        expectancy=sum(pnl) / count if count else 0.0,
        profit_factor=profit_factor,
        max_drawdown=drawdown,
    )


def _candidate_specs() -> list[tuple[str, str, Callable[[_Trade], bool]]]:
    return [
        ("baseline", "Current strategy: accept every clean recorded trade", lambda trade: True),
        ("trend_full_only", "Only trades with short and long trend fully aligned", lambda trade: trade.trend == "FULL"),
        ("avoid_countertrend", "Allow full or partial trend alignment; reject countertrend", lambda trade: trade.trend != "COUNTER"),
        ("momentum_50", "BUY only at RSI >= 50 and SELL only at RSI <= 50", lambda trade: trade.rsi is not None and ((trade.side == "BUY" and trade.rsi >= 50.0) or (trade.side == "SELL" and trade.rsi <= 50.0))),
        ("momentum_55_45", "BUY only at RSI >= 55 and SELL only at RSI <= 45", lambda trade: trade.rsi is not None and ((trade.side == "BUY" and trade.rsi >= 55.0) or (trade.side == "SELL" and trade.rsi <= 45.0))),
        ("london_only", "Only London-session trades", lambda trade: trade.session == "LONDON"),
        ("newyork_only", "Only New York-session trades", lambda trade: trade.session == "NEWYORK"),
        ("aud_only", "Only AUD/USD trades", lambda trade: trade.instrument == "AUD_USD"),
        ("gbp_only", "Only GBP/USD trades", lambda trade: trade.instrument == "GBP_USD"),
        ("aud_buy", "Only AUD/USD BUY trades", lambda trade: trade.instrument == "AUD_USD" and trade.side == "BUY"),
        ("aud_sell", "Only AUD/USD SELL trades", lambda trade: trade.instrument == "AUD_USD" and trade.side == "SELL"),
        ("gbp_buy", "Only GBP/USD BUY trades", lambda trade: trade.instrument == "GBP_USD" and trade.side == "BUY"),
        ("gbp_sell", "Only GBP/USD SELL trades", lambda trade: trade.instrument == "GBP_USD" and trade.side == "SELL"),
    ]


def _evaluate_candidate(
    name: str,
    description: str,
    predicate: Callable[[_Trade], bool],
    train: list[_Trade],
    validation: list[_Trade],
    baseline_validation: ShadowMetrics,
    *,
    min_train: int,
    min_validation: int,
    min_coverage: float,
) -> ShadowCandidateResult:
    train_selected = [trade for trade in train if predicate(trade)]
    validation_selected = [trade for trade in validation if predicate(trade)]
    train_metrics = _metrics(train_selected)
    validation_metrics = _metrics(validation_selected)
    coverage = validation_metrics.trades / len(validation) if validation else 0.0
    eligible = (
        train_metrics.trades >= min_train
        and validation_metrics.trades >= min_validation
        and coverage >= min_coverage
    )
    if not eligible:
        reason = "insufficient-walk-forward-sample"
        beats = False
    elif name == "baseline":
        reason = "current-strategy-benchmark"
        beats = False
    else:
        expectancy_margin = max(0.05, abs(baseline_validation.expectancy) * 0.15)
        beats = (
            validation_metrics.expectancy >= baseline_validation.expectancy + expectancy_margin
            and validation_metrics.profit_factor >= max(1.10, baseline_validation.profit_factor)
            and validation_metrics.max_drawdown <= baseline_validation.max_drawdown
            and train_metrics.expectancy > 0
            and train_metrics.profit_factor >= 1.0
        )
        reason = "advisory-improvement" if beats else "did-not-clear-advisory-gate"
    return ShadowCandidateResult(
        name=name,
        description=description,
        train=train_metrics,
        validation=validation_metrics,
        validation_coverage=coverage,
        eligible=eligible,
        beats_baseline=beats,
        reason=reason,
    )


def run_shadow_analysis(
    db_path: Path | str,
    *,
    output_path: Path | str | None = None,
    cohort_start_utc: str | None = None,
    cohort_run_tag: str | None = None,
    instruments: tuple[str, ...] = ALLOWED_INSTRUMENTS,
) -> ShadowReport:
    start = str(
        cohort_start_utc
        if cohort_start_utc is not None
        else os.getenv("ADAPTIVE_WINDOW_START_UTC", CLEAN_COHORT_START_UTC)
    ).strip()
    if not start:
        start = CLEAN_COHORT_START_UTC
    run_tag = str(
        cohort_run_tag
        if cohort_run_tag is not None
        else os.getenv("ADAPTIVE_RUN_TAG", DEFAULT_LEARNING_RUN_TAG)
    ).strip()
    if not run_tag:
        # An empty tag would silently broaden the cohort to legacy regimes.
        run_tag = DEFAULT_LEARNING_RUN_TAG
    trades = _load_clean_trades(
        Path(db_path),
        cohort_start_utc=start,
        cohort_run_tag=run_tag,
        instruments=instruments,
    )
    split_ratio = max(0.55, min(0.8, float(os.getenv("SHADOW_TRAIN_RATIO", "0.70"))))
    split_at = max(1, min(len(trades), int(len(trades) * split_ratio))) if trades else 0
    train = trades[:split_at]
    validation = trades[split_at:]
    min_train = max(8, int(os.getenv("SHADOW_MIN_TRAIN", "20")))
    min_validation = max(4, int(os.getenv("SHADOW_MIN_VALIDATION", "10")))
    min_coverage = max(0.2, min(1.0, float(os.getenv("SHADOW_MIN_COVERAGE", "0.35"))))

    specs = _candidate_specs()
    baseline_spec = specs[0]
    baseline_validation = _metrics(validation)
    baseline = _evaluate_candidate(
        *baseline_spec,
        train,
        validation,
        baseline_validation,
        min_train=min_train,
        min_validation=min_validation,
        min_coverage=min_coverage,
    )
    candidates = tuple(
        _evaluate_candidate(
            name,
            description,
            predicate,
            train,
            validation,
            baseline_validation,
            min_train=min_train,
            min_validation=min_validation,
            min_coverage=min_coverage,
        )
        for name, description, predicate in specs[1:]
    )
    leading_candidates = [
        candidate for candidate in candidates if candidate.beats_baseline
    ]
    leading_candidates.sort(
        key=lambda candidate: (
            candidate.validation.expectancy,
            candidate.validation.profit_factor,
            -candidate.validation.max_drawdown,
            candidate.validation.trades,
        ),
        reverse=True,
    )
    recommendation = leading_candidates[0].name if leading_candidates else None
    if recommendation:
        recommendation_reason = (
            "advisory executed-trade subset cleared local sample, profit-factor "
            "and drawdown gates; control-plane status remains collecting"
        )
    elif len(validation) < min_validation:
        recommendation_reason = f"collecting clean trades: need at least {min_validation} validation trades"
    elif len(train) < min_train:
        recommendation_reason = f"collecting clean trades: need at least {min_train} training trades"
    else:
        recommendation_reason = "no candidate has safely beaten the current strategy on unseen trades"

    report = ShadowReport(
        generated_utc=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        evidence_revision=_deployed_revision(),
        cohort_start_utc=start,
        cohort_run_tag=run_tag,
        instruments=instruments,
        total_clean_trades=len(trades),
        train_trades=len(train),
        validation_trades=len(validation),
        baseline=baseline,
        candidates=candidates,
        recommendation=recommendation,
        recommendation_reason=recommendation_reason,
        auto_apply=False,
    )
    destination = Path(output_path) if output_path is not None else Path(db_path).parent / "shadow_learning_report.json"
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(report)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(destination)
    except OSError:
        pass

    baseline_metrics = report.baseline.validation
    print(
        f"[SHADOW] clean_trades={report.total_clean_trades} train={report.train_trades} "
        f"validation={report.validation_trades} baseline_expectancy={baseline_metrics.expectancy:.3f} "
        f"baseline_pf={baseline_metrics.profit_factor:.3f} baseline_dd={baseline_metrics.max_drawdown:.2f} "
        f"recommendation={report.recommendation or 'none'} auto_apply=false "
        f"reason={report.recommendation_reason}",
        flush=True,
    )
    if report.recommendation:
        leading_candidate = next(
            candidate
            for candidate in report.candidates
            if candidate.name == report.recommendation
        )
        print(
            f"[SHADOW][CHALLENGER] name={leading_candidate.name} "
            f"validation_n={leading_candidate.validation.trades} "
            f"expectancy={leading_candidate.validation.expectancy:.3f} "
            f"pf={leading_candidate.validation.profit_factor:.3f} "
            f"drawdown={leading_candidate.validation.max_drawdown:.2f} "
            f"coverage={leading_candidate.validation_coverage:.2f} "
            "status=collecting auto_apply=false",
            flush=True,
        )
    return report


__all__ = [
    "ALLOWED_INSTRUMENTS",
    "CLEAN_COHORT_START_UTC",
    "ShadowCandidateResult",
    "ShadowMetrics",
    "ShadowReport",
    "run_shadow_analysis",
]
