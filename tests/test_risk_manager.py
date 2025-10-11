from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import json

import pytest

from src.risk_manager import RiskManager, AWST


@pytest.fixture()
def state_dir(tmp_path, monkeypatch):
    state_path = tmp_path / "state"
    monkeypatch.setenv("MOSSY_STATE_PATH", str(state_path))
    return state_path


def _utc(year, month, day, hour=0, minute=0):
    return datetime(year, month, day, hour, minute, tzinfo=timezone.utc)


def test_equity_floor_halts_and_resets(state_dir):
    manager = RiskManager({"equity_floor": 1000.0}, mode="live")
    closed = {"count": 0}

    def close_all():
        closed["count"] += 1

    now = _utc(2024, 1, 1, 0, 0)
    manager.enforce_equity_floor(now, 900.0, close_all)
    assert closed["count"] == 1

    ok, reason = manager.should_open(now, 900.0, [], "EUR_USD", 0.1)
    assert ok is False
    assert reason == "equity-floor"

    manager.enforce_equity_floor(now, 1100.0, close_all)
    assert closed["count"] == 1
    ok, reason = manager.should_open(now, 1100.0, [], "EUR_USD", 0.1)
    assert ok is True


def test_weekly_profit_bank_no_giveback(state_dir):
    manager = RiskManager(
        {
            "weekly_profit_target": 250.0,
            "no_giveback_below_target": True,
        },
        mode="paper",
    )
    monday = _utc(2024, 1, 1, 1, 0)

    manager.should_open(monday, 10_000.0, [], "EUR_USD", 0.1)
    manager.register_exit(300.0)
    ok, reason = manager.should_open(monday + timedelta(minutes=1), 10_000.0, [], "EUR_USD", 0.1)
    assert ok is True
    assert reason == "ok"

    manager.register_exit(-120.0)
    ok, reason = manager.should_open(monday + timedelta(minutes=2), 10_000.0, [], "EUR_USD", 0.1)
    assert ok is False
    assert reason == "weekly-target-locked"


def test_cooldown_and_spread_limits(state_dir):
    manager = RiskManager(
        {
            "cooldown_minutes": 45,
            "spread_pips_limit": {"EUR_USD": 1.0},
        },
        mode="paper",
    )
    now = _utc(2024, 1, 1, 2, 0)
    manager.register_entry(now)

    ok, reason = manager.should_open(now + timedelta(minutes=30), 10_000.0, [], "EUR_USD", 0.5)
    assert ok is False
    assert reason == "cooldown"

    ok, reason = manager.should_open(now + timedelta(minutes=60), 10_000.0, [], "EUR_USD", 2.0)
    assert ok is False
    assert reason == "spread-too-wide"

    ok, reason = manager.should_open(now + timedelta(minutes=60), 10_000.0, [], "EUR_USD", 0.5)
    assert ok is True


def test_daily_loss_cap_blocks_after_drawdown(state_dir):
    manager = RiskManager({"daily_loss_cap_pct": 0.01}, mode="paper")
    today = _utc(2024, 1, 1, 4, 0)

    ok, reason = manager.should_open(today, 10_000.0, [], "EUR_USD", 0.2)
    assert ok is True

    later = today + timedelta(hours=1)
    ok, reason = manager.should_open(later, 9_800.0, [], "EUR_USD", 0.2)
    assert ok is False
    assert reason == "daily-loss-cap"


def test_rollover_window_blocks(state_dir):
    manager = RiskManager(
        {
            "rollover_quiet_awst": {"start": "04:55", "end": "05:05"},
        },
        mode="paper",
    )
    awst_time = datetime(2024, 1, 1, 4, 56, tzinfo=AWST)
    now = awst_time.astimezone(timezone.utc)
    ok, reason = manager.should_open(now, 10_000.0, [], "EUR_USD", 0.1)
    assert ok is False
    assert reason == "rollover-window"


def test_state_persistence_across_restart(state_dir):
    cfg = {"weekly_profit_target": 250.0}
    first = RiskManager(cfg, mode="paper")
    monday = _utc(2024, 1, 1, 1, 0)
    first.should_open(monday, 10_000.0, [], "EUR_USD", 0.1)
    first.register_exit(200.0)
    first.should_open(monday + timedelta(minutes=1), 10_000.0, [], "EUR_USD", 0.1)

    state_file = state_dir / "risk_state.json"
    assert state_file.exists()

    with state_file.open("r", encoding="utf-8") as handle:
        snapshot = json.load(handle)

    second = RiskManager(cfg, mode="paper")
    assert second.state.weekly_realized_pl == pytest.approx(200.0)
    assert second.state.week_id == snapshot["week_id"]


def test_rollover_preserves_realized_pl_when_equity_missing(state_dir):
    manager = RiskManager({}, mode="paper")

    start = _utc(2024, 1, 1, 0, 0)
    manager.should_open(start, 10_000.0, [], "EUR_USD", 0.1)
    manager.register_exit(75.0)

    previous_day_id = manager.state.day_id

    glitch_time = start + timedelta(days=1)
    manager._rollover(glitch_time, None)

    assert manager.state.day_id != previous_day_id
    assert manager.state.day_start_equity is None
    assert manager.state.daily_realized_pl == pytest.approx(75.0)

    manager._rollover(glitch_time, 10_100.0)

    assert manager.state.day_start_equity == pytest.approx(10_100.0)
    assert manager.state.daily_realized_pl == pytest.approx(0.0)
