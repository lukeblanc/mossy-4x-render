from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src import adaptive_policy
from src.trade_journal import TradeJournal


INDICATORS = {
    "rsi": 57.0,
    "ema_fast": 1.2,
    "ema_slow": 1.1,
    "ema50": 1.15,
    "ema200": 1.0,
}


def _closed_trade(
    journal: TradeJournal,
    *,
    trade_id: str,
    pnl: float,
    opened: datetime,
) -> None:
    journal.record_entry(
        trade_id=trade_id,
        timestamp_utc=opened,
        instrument="AUD_USD",
        side="BUY",
        units=1000,
        entry_price=0.66,
        stop_loss_price=0.659,
        take_profit_price=0.661,
        spread_at_entry=0.8,
        session_id="LONDON-2026-07-10-1500",
        session_mode="SOFT",
        run_tag="MINI_RUN",
        gating_flags={"trend_ok": True},
        indicators_snapshot=INDICATORS,
        equity_after=1500.0,
    )
    journal.record_exit(
        trade_id=trade_id,
        exit_timestamp_utc=opened + timedelta(minutes=15),
        exit_price=0.6605,
        spread_at_exit=0.8,
        max_profit_ccy=max(pnl, 0.0),
        realized_pnl_ccy=pnl,
        exit_reason="TP" if pnl > 0 else "SL",
        duration_seconds=900,
        broker_confirmed=True,
        instrument="AUD_USD",
        direction="BUY",
        entry_price=0.66,
        equity_after=1500.0 + pnl,
    )


def _publish(now: datetime) -> None:
    adaptive_policy.publish_market_context(
        "AUD_USD",
        INDICATORS,
        now,
        side="BUY",
        session="LONDON",
    )


def test_policy_keeps_full_risk_until_setup_has_enough_evidence(tmp_path, monkeypatch):
    adaptive_policy.clear_policy_caches()
    monkeypatch.setenv("ADAPTIVE_POLICY_MIN_EXACT", "6")
    journal = TradeJournal(tmp_path / "trade_journal.db")
    now = datetime(2026, 7, 11, 8, 0, tzinfo=timezone.utc)
    for index, pnl in enumerate((1.0, -0.5, 0.8)):
        _closed_trade(journal, trade_id=f"T{index}", pnl=pnl, opened=now - timedelta(days=3 - index))

    _publish(now)
    decision = adaptive_policy.evaluate_instrument_policy(
        "AUD_USD",
        db_path=journal.path,
        now_utc=now,
    )

    assert decision.blocked is False
    assert decision.risk_scale == pytest.approx(1.0)
    assert decision.reason == "insufficient-setup-history"


def test_policy_reduces_repeated_losing_setup(tmp_path, monkeypatch):
    adaptive_policy.clear_policy_caches()
    monkeypatch.setenv("ADAPTIVE_POLICY_MIN_EXACT", "6")
    monkeypatch.setenv("ADAPTIVE_POLICY_CACHE_SECONDS", "1")
    journal = TradeJournal(tmp_path / "trade_journal.db")
    now = datetime(2026, 7, 11, 8, 0, tzinfo=timezone.utc)
    pnls = (1.0, 0.8, 0.4, -0.5, -0.6, -0.7)
    for index, pnl in enumerate(pnls):
        _closed_trade(
            journal,
            trade_id=f"T{index}",
            pnl=pnl,
            opened=now - timedelta(days=len(pnls) - index),
        )

    _publish(now)
    decision = adaptive_policy.evaluate_instrument_policy(
        "AUD_USD",
        db_path=journal.path,
        now_utc=now,
    )

    assert decision.blocked is False
    assert decision.exact_samples == 6
    assert decision.loss_streak == 3
    assert decision.risk_scale == pytest.approx(0.5)
    assert decision.reason == "setup-three-loss-streak"


def test_policy_temporarily_blocks_four_loss_setup(tmp_path, monkeypatch):
    adaptive_policy.clear_policy_caches()
    monkeypatch.setenv("ADAPTIVE_POLICY_MIN_EXACT", "6")
    monkeypatch.setenv("ADAPTIVE_POLICY_BLOCK_MINUTES", "240")
    journal = TradeJournal(tmp_path / "trade_journal.db")
    now = datetime(2026, 7, 11, 8, 0, tzinfo=timezone.utc)
    pnls = (1.0, 0.8, -0.3, -0.4, -0.5, -0.6)
    for index, pnl in enumerate(pnls):
        opened = now - timedelta(minutes=(len(pnls) - index) * 20)
        _closed_trade(journal, trade_id=f"T{index}", pnl=pnl, opened=opened)

    _publish(now)
    decision = adaptive_policy.evaluate_instrument_policy(
        "AUD_USD",
        db_path=journal.path,
        now_utc=now,
    )

    assert decision.blocked is True
    assert decision.risk_scale == pytest.approx(0.0)
    assert decision.loss_streak == 4
    assert decision.reason == "setup-four-loss-streak"
