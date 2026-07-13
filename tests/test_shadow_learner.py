from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json

import pytest

from src.shadow_learner import CLEAN_COHORT_START_UTC, run_shadow_analysis
from src.trade_journal import TradeJournal


def _closed_trade(
    journal: TradeJournal,
    *,
    trade_id: str,
    opened: datetime,
    instrument: str = "AUD_USD",
    side: str = "BUY",
    rsi: float = 60.0,
    pnl: float = 1.0,
    broker_confirmed: bool = True,
) -> None:
    journal.record_entry(
        trade_id=trade_id,
        timestamp_utc=opened,
        instrument=instrument,
        side=side,
        units=1000,
        entry_price=0.66,
        stop_loss_price=0.659,
        take_profit_price=0.661,
        spread_at_entry=0.8,
        session_id="LONDON",
        session_mode="SOFT",
        run_tag="MINI_RUN",
        gating_flags={"trend_ok": True},
        indicators_snapshot={
            "rsi": rsi,
            "ema_fast": 1.2,
            "ema_slow": 1.1,
            "ema50": 1.15,
            "ema200": 1.0,
        },
        equity_after=1500.0,
    )
    journal.record_exit(
        trade_id=trade_id,
        exit_timestamp_utc=opened + timedelta(minutes=10),
        exit_price=0.6605,
        spread_at_exit=0.8,
        max_profit_ccy=max(0.0, pnl),
        realized_pnl_ccy=pnl,
        exit_reason="TP" if pnl > 0 else "SL",
        duration_seconds=600,
        broker_confirmed=broker_confirmed,
        instrument=instrument,
        direction=side,
        entry_price=0.66,
        equity_after=1500.0 + pnl,
    )


def test_shadow_uses_only_clean_confirmed_current_pair_cohort(tmp_path, monkeypatch):
    journal = TradeJournal(tmp_path / "trade_journal.db")
    cohort = datetime.fromisoformat(CLEAN_COHORT_START_UTC)
    _closed_trade(journal, trade_id="valid", opened=cohort + timedelta(minutes=1))
    _closed_trade(journal, trade_id="old", opened=cohort - timedelta(days=1))
    _closed_trade(
        journal,
        trade_id="gold",
        opened=cohort + timedelta(minutes=2),
        instrument="XAU_USD",
    )
    _closed_trade(
        journal,
        trade_id="unconfirmed",
        opened=cohort + timedelta(minutes=3),
        broker_confirmed=False,
    )
    monkeypatch.setenv("SHADOW_MIN_TRAIN", "8")
    monkeypatch.setenv("SHADOW_MIN_VALIDATION", "4")

    output = tmp_path / "report.json"
    report = run_shadow_analysis(journal.path, output_path=output)

    assert report.total_clean_trades == 1
    assert report.recommendation is None
    assert report.auto_apply is False
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["total_clean_trades"] == 1
    assert saved["auto_apply"] is False


def test_shadow_recommends_only_after_unseen_validation_beats_baseline(tmp_path, monkeypatch):
    journal = TradeJournal(tmp_path / "trade_journal.db")
    cohort = datetime.fromisoformat(CLEAN_COHORT_START_UTC)
    for index in range(40):
        momentum_aligned = index % 2 == 0
        _closed_trade(
            journal,
            trade_id=f"T{index}",
            opened=cohort + timedelta(minutes=index * 15),
            rsi=60.0 if momentum_aligned else 40.0,
            pnl=2.0 if momentum_aligned else -2.0,
        )

    monkeypatch.setenv("SHADOW_MIN_TRAIN", "8")
    monkeypatch.setenv("SHADOW_MIN_VALIDATION", "4")
    monkeypatch.setenv("SHADOW_MIN_COVERAGE", "0.20")
    monkeypatch.setenv("SHADOW_TRAIN_RATIO", "0.70")

    report = run_shadow_analysis(journal.path, output_path=tmp_path / "report.json")

    assert report.train_trades == 28
    assert report.validation_trades == 12
    assert report.baseline.validation.expectancy == pytest.approx(0.0)
    assert report.recommendation in {"momentum_50", "momentum_55_45"}
    champion = next(candidate for candidate in report.candidates if candidate.name == report.recommendation)
    assert champion.beats_baseline is True
    assert champion.validation.trades == 6
    assert champion.validation.expectancy == pytest.approx(2.0)
    assert champion.validation.max_drawdown == pytest.approx(0.0)
    assert report.auto_apply is False


def test_shadow_never_recommends_overfit_training_only_result(tmp_path, monkeypatch):
    journal = TradeJournal(tmp_path / "trade_journal.db")
    cohort = datetime.fromisoformat(CLEAN_COHORT_START_UTC)
    for index in range(40):
        in_training = index < 28
        aligned = index % 2 == 0
        if in_training:
            pnl = 2.0 if aligned else -2.0
        else:
            pnl = -2.0 if aligned else 2.0
        _closed_trade(
            journal,
            trade_id=f"T{index}",
            opened=cohort + timedelta(minutes=index * 15),
            rsi=60.0 if aligned else 40.0,
            pnl=pnl,
        )

    monkeypatch.setenv("SHADOW_MIN_TRAIN", "8")
    monkeypatch.setenv("SHADOW_MIN_VALIDATION", "4")
    monkeypatch.setenv("SHADOW_MIN_COVERAGE", "0.20")
    monkeypatch.setenv("SHADOW_TRAIN_RATIO", "0.70")

    report = run_shadow_analysis(journal.path, output_path=tmp_path / "report.json")

    momentum = next(candidate for candidate in report.candidates if candidate.name == "momentum_50")
    assert momentum.train.expectancy > 0
    assert momentum.validation.expectancy < 0
    assert momentum.beats_baseline is False
    assert report.recommendation not in {"momentum_50", "momentum_55_45"}
    assert report.auto_apply is False
