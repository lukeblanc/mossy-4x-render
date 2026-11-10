from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from src.risk_manager import RiskManager, RiskState


NOW = datetime(2026, 11, 10, 4, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def clean_demo_floor_env(monkeypatch):
    monkeypatch.delenv("DEMO_EQUITY_FLOOR_AUD", raising=False)


def manager(path, *, demo_mode=True, mode="paper", **overrides):
    config = {
        "weekly_profit_target": 0,
        "daily_profit_target_usd": 0,
        "daily_loss_cap_pct": 1,
        "weekly_loss_cap_pct": 1,
        "max_drawdown_cap_pct": 1,
        **overrides,
    }
    return RiskManager(config, demo_mode=demo_mode, mode=mode, state_dir=path)


def entry(risk, equity, now=NOW):
    return risk.should_open(now, equity, [], "AUD_USD", 0.1)


@pytest.mark.parametrize("equity,blocked", [(500.01, False), (500.0, True), (499.99, True)])
def test_demo_floor_entry_boundary_and_durable_latch(tmp_path, equity, blocked):
    risk = manager(tmp_path)
    ok, reason = entry(risk, equity)
    assert ok is not blocked
    assert reason == ("demo-equity-floor" if blocked else "ok")
    assert risk.state.demo_halted_on_equity_floor is blocked
    if blocked:
        assert json.loads(risk._state_file.read_text())["demo_halted_on_equity_floor"] is True


@pytest.mark.parametrize("equity,expected_closes", [(500.01, 0), (500.0, 1), (499.99, 1)])
def test_enforcement_persists_before_close(tmp_path, equity, expected_closes):
    risk = manager(tmp_path)
    calls = []

    def close():
        persisted = json.loads(risk._state_file.read_text())
        assert persisted["demo_halted_on_equity_floor"] is True
        calls.append(True)
        return True

    risk.enforce_equity_floor(NOW, equity, close)
    assert len(calls) == expected_closes
    assert risk.state.demo_halted_on_equity_floor is bool(expected_closes)


def test_demo_halt_survives_rebound_restart_rollover_and_other_resets(tmp_path):
    risk = manager(tmp_path)
    risk.enforce_equity_floor(NOW, 499, lambda: True)
    risk.enforce_equity_floor(NOW + timedelta(minutes=1), 900, lambda: True)
    assert entry(risk, 900) == (False, "demo-equity-floor")

    restarted = manager(tmp_path)
    later = NOW + timedelta(days=8)
    restarted.startup_daily_reset(900, now_utc=later)
    restarted.clear_weekly_loss_cap(900)
    restarted.clear_max_drawdown_halt(900)
    assert restarted.start_demo_run(
        "separate-drawdown-reset", 900, open_positions_count=0,
        oanda_env="practice", now_utc=later,
    ) == (True, "applied")
    assert entry(restarted, 900, later) == (False, "demo-equity-floor")
    again = manager(tmp_path)
    assert again.state.demo_halted_on_equity_floor is True
    assert entry(again, 900, later) == (False, "demo-equity-floor")


def test_existing_drawdown_and_live_latches_do_not_suppress_demo_protection(tmp_path):
    risk = manager(tmp_path)
    risk.state.max_drawdown_halt = True
    risk.state.live_halted_on_equity_floor = True
    risk._save_state()
    calls = []
    risk.enforce_equity_floor(NOW, 499, lambda: calls.append(True) or True)
    assert calls == [True]
    assert risk.state.max_drawdown_halt is True
    assert risk.state.live_halted_on_equity_floor is True
    assert entry(risk, 900) == (False, "demo-equity-floor")


@pytest.mark.parametrize("result", [None, False, 1, "confirmed", True])
def test_close_result_never_clears_halt_and_callback_remains_idempotent(tmp_path, capsys, result):
    risk = manager(tmp_path)
    calls = []

    def close():
        calls.append(result)
        return result

    risk.enforce_equity_floor(NOW, 499, close)
    risk.enforce_equity_floor(NOW + timedelta(seconds=60), 900, close)
    assert calls == [result, result]
    assert entry(risk, 900) == (False, "demo-equity-floor")
    assert ("Flat account unconfirmed" in capsys.readouterr().out) is (result is not True)


def test_close_exception_is_retried_without_fresh_nav(tmp_path):
    risk = manager(tmp_path)
    calls = []

    def close():
        calls.append(True)
        raise RuntimeError("simulated unavailable broker")

    risk.enforce_equity_floor(NOW, 499, close)
    for equity in (None, float("nan"), float("inf"), 0, -1, "invalid"):
        risk.enforce_equity_floor(NOW, equity, close)
        assert entry(risk, equity) == (False, "demo-equity-floor")
    assert len(calls) == 7


def test_restart_retries_verified_close_even_without_nav(tmp_path):
    risk = manager(tmp_path)
    risk.enforce_equity_floor(NOW, 499, lambda: False)
    restarted = manager(tmp_path)
    calls = []
    restarted.enforce_equity_floor(NOW, None, lambda: calls.append(True) or True)
    assert calls == [True]
    assert restarted.state.demo_halted_on_equity_floor is True
    assert entry(restarted, 900) == (False, "demo-equity-floor")


@pytest.mark.parametrize("equity", [None, float("nan"), float("inf"), float("-inf"), 0, -1, "invalid"])
def test_invalid_unlatched_nav_blocks_entries_without_inventing_a_floor_breach(tmp_path, equity):
    risk = manager(tmp_path)
    assert entry(risk, equity) == (False, "equity-unavailable")
    risk.enforce_equity_floor(NOW, equity, lambda: pytest.fail("no valid floor breach"))
    assert risk.state.demo_halted_on_equity_floor is False


@pytest.mark.parametrize("invalid", [499.99, 0, -1, float("nan"), float("inf"), float("-inf"), None, "", "invalid", True])
def test_invalid_demo_floor_config_fails_startup(tmp_path, invalid):
    with pytest.raises(ValueError, match="finite and at least AUD 500"):
        manager(tmp_path, demo_equity_floor_aud=invalid)


@pytest.mark.parametrize("invalid", ["499", "0", "nan", "inf", "-inf", "", "invalid"])
def test_invalid_demo_floor_environment_fails_startup(tmp_path, monkeypatch, invalid):
    monkeypatch.setenv("DEMO_EQUITY_FLOOR_AUD", invalid)
    with pytest.raises(ValueError, match="finite and at least AUD 500"):
        manager(tmp_path)


def test_demo_floor_default_config_and_environment_precedence(tmp_path, monkeypatch):
    assert manager(tmp_path / "default").demo_equity_floor_aud == 500
    assert manager(tmp_path / "config", demo_equity_floor_aud=600).demo_equity_floor_aud == 600
    monkeypatch.setenv("DEMO_EQUITY_FLOOR_AUD", "700")
    risk = manager(tmp_path / "environment", demo_equity_floor_aud=600)
    assert risk.demo_equity_floor_aud == 700
    assert entry(risk, 700) == (False, "demo-equity-floor")


@pytest.mark.parametrize("mode,demo_mode", [("paper", False), ("live", False), ("live", True)])
def test_inactive_demo_configuration_does_not_change_live_or_paper(tmp_path, monkeypatch, mode, demo_mode):
    monkeypatch.setenv("DEMO_EQUITY_FLOOR_AUD", "invalid")
    risk = manager(tmp_path, mode=mode, demo_mode=demo_mode, demo_equity_floor_aud="invalid")
    calls = []
    risk.enforce_equity_floor(NOW, 400, lambda: calls.append(True))
    if mode == "live":
        assert calls == [True]
        assert entry(risk, 400) == (False, "equity-floor")
        risk.enforce_equity_floor(NOW, 1100, lambda: calls.append(True))
        assert entry(risk, 1100) == (True, "ok")
    else:
        assert calls == []
        assert entry(risk, 400) == (True, "ok")
    assert risk.state.demo_halted_on_equity_floor is False


def test_old_state_defaults_demo_latch_false_and_round_trips_true():
    assert RiskState.from_dict({}).demo_halted_on_equity_floor is False
    state = RiskState(demo_halted_on_equity_floor=True)
    assert RiskState.from_dict(state.to_dict()).demo_halted_on_equity_floor is True


def test_disk_failure_still_attempts_protection_and_keeps_entries_blocked(tmp_path, monkeypatch):
    risk = manager(tmp_path)
    calls = []

    def failed_replace(*args, **kwargs):
        raise OSError("simulated disk unavailable")

    monkeypatch.setattr("src.risk_manager.os.replace", failed_replace)
    risk.enforce_equity_floor(NOW, 499, lambda: calls.append(True) or True)
    assert calls == [True]
    assert risk._state_save_failed is True
    assert risk.state.demo_halted_on_equity_floor is True
    assert entry(risk, 900) == (False, "risk-state-unwritable")
    risk.enforce_equity_floor(NOW, None, lambda: calls.append(True) or True)
    assert calls == [True, True]
    assert risk._state_save_failed is True

    monkeypatch.undo()
    risk.enforce_equity_floor(NOW, None, lambda: calls.append(True) or True)
    assert calls == [True, True, True]
    assert risk._state_save_failed is False
    assert json.loads(risk._state_file.read_text())["demo_halted_on_equity_floor"] is True
    assert entry(manager(tmp_path), 900) == (False, "demo-equity-floor")
