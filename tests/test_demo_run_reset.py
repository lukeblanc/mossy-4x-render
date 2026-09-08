from datetime import datetime, timedelta, timezone
import json
import os

import pytest

from src.risk_manager import RiskManager


NOW = datetime(2026, 9, 8, 4, tzinfo=timezone.utc)
EQUITY = 1381.5154


@pytest.fixture
def halted(tmp_path):
    manager = RiskManager(
        {"max_drawdown_cap_pct": 0.05, "daily_loss_cap_pct": 0.01,
         "weekly_loss_cap_pct": 0.03},
        mode="paper", demo_mode=True, state_dir=tmp_path,
    )
    manager.startup_daily_reset(EQUITY, now_utc=NOW)
    manager.state.peak_equity = 1455.3837
    manager.state.max_drawdown_halt = True
    manager.state.daily_entry_count = 3
    manager.state.daily_realized_pl = -2
    manager.state.weekly_realized_pl = -4
    manager.state.last_trades = [{"pl": -2, "instrument": "AUD_USD"}]
    manager.state.cooldown_until = {"AUD_USD": NOW + timedelta(minutes=15)}
    manager._save_state()
    return manager


def start(manager, run_id="luke-approved-demo", **overrides):
    args = dict(equity=EQUITY, open_positions_count=0, oanda_env="practice", now_utc=NOW)
    args.update(overrides)
    return manager.start_demo_run(run_id, **args)


def test_reset_preserves_risk_history_and_journal_with_atomic_audit(halted, tmp_path):
    journal = tmp_path / "trade_journal.db"
    journal.write_bytes(b"existing trade history must remain untouched")
    before = halted.state.to_dict()
    assert start(halted) == (True, "applied")
    saved = json.loads((tmp_path / "risk_state.json").read_text())
    runs = saved.pop("demo_runs")
    before.pop("demo_runs")
    assert runs["luke-approved-demo"] == {
        "started_at_utc": NOW.isoformat(), "start_equity": EQUITY, "previous_state": before,
    }
    assert saved == {**before, "peak_equity": EQUITY, "max_drawdown_halt": False}
    assert journal.read_bytes() == b"existing trade history must remain untouched"


def test_restart_cannot_replay_reset_or_clear_a_later_drawdown_halt(halted, tmp_path):
    assert start(halted)[0]
    # The unchanged 5% brake still latches against the fresh peak.
    halted.enforce_equity_floor(NOW, EQUITY * 0.94, lambda: pytest.fail("demo must not close all"))
    assert halted.state.max_drawdown_halt
    restarted = RiskManager(halted.config, mode="paper", demo_mode=True, state_dir=tmp_path)
    restarted.startup_daily_reset(EQUITY * 0.94, now_utc=NOW)
    original = restarted._state_file.read_bytes()
    assert start(restarted) == (False, "already-applied")
    assert restarted.state.max_drawdown_halt
    assert restarted.state.peak_equity == EQUITY
    assert restarted._state_file.read_bytes() == original


def test_daily_and_weekly_caps_remain_effective(halted):
    assert start(halted)[0]
    assert halted.should_open(NOW, EQUITY * 0.98, [], "GBP_USD", 0.1) == (False, "daily-loss-cap")
    # A new day may roll its baseline; the weekly baseline must remain protected.
    assert halted.should_open(NOW + timedelta(days=1), EQUITY * 0.96, [], "GBP_USD", 0.1) == (False, "weekly-loss-cap")


@pytest.mark.parametrize("overrides,reason", [
    ({"equity": None}, "equity-unavailable"),
    ({"equity": float("nan")}, "equity-unavailable"),
    ({"equity": float("inf")}, "equity-unavailable"),
    ({"equity": 0}, "equity-unavailable"),
    ({"open_positions_count": None}, "confirmed-flat-account-required"),
    ({"open_positions_count": 1}, "confirmed-flat-account-required"),
    ({"open_positions_count": False}, "confirmed-flat-account-required"),
    ({"oanda_env": "live"}, "practice-demo-required"),
    ({"oanda_env": ""}, "practice-demo-required"),
])
def test_unverified_broker_state_cannot_reset(halted, overrides, reason):
    original = halted._state_file.read_bytes()
    assert start(halted, **overrides) == (False, reason)
    assert halted._state_file.read_bytes() == original
    assert halted.state.max_drawdown_halt


@pytest.mark.parametrize("mode,demo", [("live", True), ("live", False), ("paper", False)])
def test_only_demo_mode_may_reset(halted, mode, demo):
    halted.mode, halted.demo_mode = mode, demo
    assert start(halted) == (False, "practice-demo-required")
    assert halted.state.max_drawdown_halt


@pytest.mark.parametrize("run_id", ["", "../reset", "line\nbreak", "x" * 97])
def test_invalid_request_does_not_reset(halted, run_id):
    assert start(halted, run_id) == (False, "invalid-run-id")
    assert halted.state.max_drawdown_halt


def test_failed_save_rolls_back_reset_and_does_not_consume_id(halted, monkeypatch):
    original = halted._state_file.read_bytes()
    before = halted.state.to_dict()
    def unavailable(*args):
        raise OSError("disk unavailable")
    with monkeypatch.context() as patch:
        patch.setattr(os, "replace", unavailable)
        assert start(halted) == (False, "risk-state-unwritable")
        assert halted.state.to_dict() == before
        assert halted._state_file.read_bytes() == original
        assert halted.should_open(NOW, EQUITY, [], "GBP_USD", 0.1) == (False, "risk-state-unwritable")
    # Ordinary disk recovery must persist the old halt, not the failed reset.
    assert halted.should_open(NOW, EQUITY, [], "GBP_USD", 0.1) == (False, "max-drawdown")
    assert start(halted) == (True, "applied")


def test_corrupt_state_is_never_replaced_by_reset(tmp_path):
    path = tmp_path / "risk_state.json"
    path.write_text('{"peak_equity":')
    manager = RiskManager({}, demo_mode=True, state_dir=tmp_path)
    assert start(manager) == (False, "risk-state-unavailable")
    assert path.read_text() == '{"peak_equity":'


def test_all_consumed_ids_and_previous_runs_survive(halted, tmp_path):
    assert start(halted, "first")[0]
    assert start(halted, "second")[0]
    restarted = RiskManager({}, demo_mode=True, state_dir=tmp_path)
    restarted.state.max_drawdown_halt = True
    restarted._save_state()
    assert start(restarted, "first") == (False, "already-applied")
    assert start(restarted, "second") == (False, "already-applied")
    assert restarted.state.max_drawdown_halt
    assert set(restarted.state.demo_runs) == {"first", "second"}
