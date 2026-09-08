from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from src.risk_manager import RiskManager
from src import adaptive_policy, position_sizer


def test_same_day_restart_preserves_loss_baseline_counts_and_halt(tmp_path):
    now = datetime(2026, 9, 8, 4, tzinfo=timezone.utc)
    config = {"daily_loss_cap_pct": 0.01}
    first = RiskManager(config, mode="paper", demo_mode=True, state_dir=tmp_path)
    first.startup_daily_reset(1000, now_utc=now)
    first.register_entry(now, "AUD_USD")
    first.state.max_drawdown_halt = True
    first.state.daily_realized_pl = -15
    first.state.daily_profit_cap_hit = True
    first._save_state()

    second = RiskManager(config, mode="paper", demo_mode=True, state_dir=tmp_path)
    second.startup_daily_reset(985, now_utc=now + timedelta(hours=1))
    assert second.state.day_start_equity == 1000
    assert second.state.day_start_equity_utc == 1000
    assert second.state.week_start_equity == 1000
    assert second.state.daily_entry_count == 1
    assert second.state.daily_realized_pl == -15
    assert second.state.daily_profit_cap_hit
    assert second.state.max_drawdown_halt
    assert second.should_open(now, 985, [], "GBP_USD", 0.1) == (False, "daily-loss-cap")


def test_restart_on_new_day_rolls_only_daily_limits(tmp_path):
    now = datetime(2026, 9, 8, 4, tzinfo=timezone.utc)
    first = RiskManager({}, mode="paper", demo_mode=True, state_dir=tmp_path)
    first.startup_daily_reset(1000, now_utc=now)
    first.register_entry(now, "AUD_USD")
    first.state.max_drawdown_halt = True
    first._save_state()
    second = RiskManager({}, mode="paper", demo_mode=True, state_dir=tmp_path)
    second.startup_daily_reset(980, now_utc=now + timedelta(days=1))
    assert second.state.day_start_equity == 980
    assert second.state.day_start_equity_utc == 980
    assert second.state.week_start_equity == 1000
    assert second.state.peak_equity == 1000
    assert second.state.daily_entry_count == 0
    assert second.state.max_drawdown_halt


def test_corrupt_state_is_preserved_and_blocks_entries(tmp_path):
    path = tmp_path / "risk_state.json"
    path.write_text('{"day_start_equity":')
    manager = RiskManager({}, state_dir=tmp_path, demo_mode=True)
    manager.startup_daily_reset(1000)
    assert manager.should_open(datetime.now(timezone.utc), 1000, [], "AUD_USD", 0.1) == (False, "risk-state-unreadable")
    assert path.read_text() == '{"day_start_equity":'


def test_failed_atomic_write_preserves_previous_state_and_blocks_entries(tmp_path, monkeypatch):
    manager = RiskManager({}, state_dir=tmp_path, demo_mode=True)
    manager.startup_daily_reset(1000)
    path = tmp_path / "risk_state.json"
    original = path.read_bytes()
    def unavailable(*args):
        raise OSError("disk unavailable")
    monkeypatch.setattr(os, "replace", unavailable)
    manager.register_entry(datetime.now(timezone.utc), "AUD_USD")
    assert path.read_bytes() == original
    assert manager.should_open(datetime.now(timezone.utc), 1000, [], "AUD_USD", 0.1) == (False, "risk-state-unwritable")


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), -100])
def test_invalid_equity_cannot_open_trade(tmp_path, value):
    manager = RiskManager({}, state_dir=tmp_path)
    assert manager.should_open(datetime.now(timezone.utc), value, [], "AUD_USD", 0.1) == (False, "equity-unavailable")


@pytest.mark.parametrize("first_import", ["src", "app.config"])
def test_safe_profile_keeps_stricter_values_and_real_risk_config(tmp_path, first_import):
    env = os.environ.copy()
    env.update({"RENDER_GIT_COMMIT": "test", "MOSSY_STATE_PATH": str(tmp_path),
                "MODE": "demo", "OANDA_ENV": "practice", "OANDA_API_KEY": "",
                "OANDA_ACCOUNT_ID": "", "ALGO_WEEKLY_REPORT_ENABLED": "false",
                "ADAPTIVE_MIN_SAMPLE": "40", "SHADOW_MIN_TRAIN": "80",
                "SHADOW_MIN_VALIDATION": "45", "SHADOW_MIN_COVERAGE": "0.75",
                "DAILY_LOSS_CAP_PCT": "0.005", "MAX_CONCURRENT_POSITIONS": "1",
                "MAX_TRADES_PER_DAY": "4", "RESET_MAX_DRAWDOWN_HALT": "true"})
    code = f"""
import {first_import}
import src.main as main
import os, json
print(json.dumps(dict(sample=main.adaptive_tuner.min_sample,
    daily=main.risk.daily_loss_cap_pct, positions=main.risk.max_concurrent_positions,
    entries=main.risk.max_trades_per_day, reset=os.environ['RESET_MAX_DRAWDOWN_HALT'],
    train=os.environ['SHADOW_MIN_TRAIN'], validation=os.environ['SHADOW_MIN_VALIDATION'],
    coverage=os.environ['SHADOW_MIN_COVERAGE'])))
"""
    result = subprocess.run([sys.executable, "-c", code], env=env,
                            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, check=True)
    actual = json.loads(result.stdout.strip().splitlines()[-1])
    assert actual == dict(sample=40, daily=0.005, positions=1, entries=4,
                         reset="false", train="80", validation="45", coverage="0.75")


def test_modest_cash_cap_increase_and_minimum_unit_protection(monkeypatch):
    monkeypatch.setattr(adaptive_policy, "evaluate_instrument_policy", lambda instrument:
                        adaptive_policy.PolicyDecision(instrument=instrument, setup_key="test", risk_scale=1,
                                                       blocked=False, reason="test"))
    class Broker:
        def conversion_rate(self, *args):
            return 1.5
    monkeypatch.setenv("MAX_RISK_PER_TRADE_CCY", "1.50")
    before, _ = position_sizer.units_for_risk(1381.52, "AUD_USD", 0.001, 0.0025, broker=Broker())
    monkeypatch.setenv("MAX_RISK_PER_TRADE_CCY", "1.80")
    after, diag = position_sizer.units_for_risk(1381.52, "AUD_USD", 0.001, 0.0025, broker=Broker())
    assert after == 1200 and before == 1000
    assert diag["risk_amount"] == 1.80
    units, diag = position_sizer.units_for_risk(1381.52, "AUD_USD", 0.001, 0.0025,
                                              broker=Broker(), min_trade_units=2000)
    assert units == 0
    assert diag["reason"] == "minimum-units-exceed-risk-budget"
