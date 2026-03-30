from __future__ import annotations

from types import SimpleNamespace
from typing import Dict

import src.main as main_mod


def test_trailing_config_prefers_min_check_override(monkeypatch):
    config: Dict = {"trailing": {"min_check_interval_sec": 5.0}}
    monkeypatch.setenv("TRAIL_MIN_CHECK_INTERVAL", "12")
    monkeypatch.setenv("MIN_CHECK_INTERVAL_SEC", "7")

    trailing = main_mod._build_trailing_config(config)

    assert trailing["min_check_interval_sec"] == 7.0


def test_trailing_config_soft_scalp_from_env(monkeypatch):
    config: Dict = {"trailing": {"soft_scalp_mode": False}}
    monkeypatch.setenv("SOFT_SCALP_MODE", "true")

    trailing = main_mod._build_trailing_config(config)

    assert trailing["soft_scalp_mode"] is True


def test_trailing_config_soft_scalp_from_config(monkeypatch):
    config: Dict = {"trailing": {"soft_scalp_mode": True}}
    monkeypatch.delenv("SOFT_SCALP_MODE", raising=False)

    trailing = main_mod._build_trailing_config(config)

    assert trailing["soft_scalp_mode"] is True


def test_trailing_config_warns_on_non_positive_thresholds(monkeypatch, capsys):
    config: Dict = {}
    monkeypatch.setenv("TRAIL_ARM_CCY", "0")
    monkeypatch.setenv("TRAIL_GIVEBACK_CCY", "-1")

    trailing = main_mod._build_trailing_config(config)

    captured = capsys.readouterr()
    assert "arm_ccy=0.0" in captured.out
    assert "giveback_ccy=-1.0" in captured.out
    assert trailing["arm_ccy"] == 0.0
    assert trailing["giveback_ccy"] == -1.0


def test_format_trading_summary_uses_explicit_trade_counter_names():
    snapshot = SimpleNamespace(
        source="trades",
        lifetime_closed_trades=42,
        session_closed_trades=7,
        wins=4,
        losses=3,
        loss_streak=2,
        risk_multiplier=0.75,
        filter_run_tag="MINI_RUN",
        filter_window_start_utc="2026-03-30T00:00:00+00:00",
        filter_window_end_utc="2026-03-30T01:00:00+00:00",
    )

    summary = main_mod._format_trading_summary(snapshot)

    assert "lifetime_closed_trades=42" in summary
    assert "session_closed_trades=7" in summary
    assert " closed=" not in summary
    assert "run_tag=MINI_RUN" in summary
    assert "window_start_utc=2026-03-30T00:00:00+00:00" in summary
    assert "window_end_utc=2026-03-30T01:00:00+00:00" in summary
