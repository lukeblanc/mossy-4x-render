from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

import src.main as main
from app.broker import PRACTICE
from src.risk_manager import RiskManager


@pytest.mark.parametrize("remaining,expected", [([], True), (None, False), ([{"id": "1"}], False)])
def test_demo_flatten_requires_fresh_flat_snapshot(monkeypatch, remaining, expected):
    calls = []
    broker = SimpleNamespace(
        mode="demo", base_url=PRACTICE, entry_halt_reason="protective-stop-not-confirmed",
        close_all_positions=lambda: calls.append("close"),
    )
    monkeypatch.setattr(main, "mode_env", "demo")
    monkeypatch.setattr(main, "oanda_env", "practice")
    monkeypatch.setattr(main, "broker", broker)

    def fresh_snapshot(*, force_refresh):
        calls.append(("verify", force_refresh))
        return remaining

    monkeypatch.setattr(main, "_open_trades_state", fresh_snapshot)
    assert main._close_demo_positions_and_verify() is expected
    assert calls == ["close", ("verify", True)]
    assert broker.entry_halt_reason == "protective-stop-not-confirmed"


@pytest.mark.parametrize("mode,environment,base_url", [
    ("live", "practice", PRACTICE),
    ("demo", "live", PRACTICE),
    ("demo", "practice", "https://api-fxtrade.oanda.com"),
])
def test_demo_flatten_refuses_unverified_practice_context(monkeypatch, mode, environment, base_url):
    calls = []
    monkeypatch.setattr(main, "mode_env", mode)
    monkeypatch.setattr(main, "oanda_env", environment)
    monkeypatch.setattr(main, "broker", SimpleNamespace(
        mode=mode, base_url=base_url,
        close_all_positions=lambda: calls.append("close"),
    ))
    assert main._close_demo_positions_and_verify() is False
    assert calls == []


def test_demo_flatten_keeps_failure_distinct_from_verified_close(monkeypatch):
    def fail():
        raise RuntimeError("unavailable")

    monkeypatch.setattr(main, "mode_env", "demo")
    monkeypatch.setattr(main, "oanda_env", "practice")
    monkeypatch.setattr(main, "broker", SimpleNamespace(mode="demo", base_url=PRACTICE, close_all_positions=fail))
    assert main._close_demo_positions_and_verify() is False


@pytest.mark.parametrize("equity,already_latched", [(499.0, False), (None, True)])
def test_heartbeat_enforces_demo_stop_off_session_and_retries_without_nav(
    monkeypatch, tmp_path, equity, already_latched
):
    risk = RiskManager({}, mode="paper", demo_mode=True, state_dir=tmp_path / "risk")
    if already_latched:
        risk.state.demo_halted_on_equity_floor = True
        risk._save_state()
    calls = []
    positions = [{"id": "7", "instrument": "AUD_USD"}]

    def close_all():
        calls.append("close")
        positions.clear()

    broker = SimpleNamespace(
        mode="demo", base_url=PRACTICE, entry_halt_reason=None,
        account_equity=lambda: equity,
        close_all_positions=close_all,
        list_open_trades=lambda: list(positions),
    )
    monkeypatch.setattr(main, "risk", risk)
    monkeypatch.setattr(main, "broker", broker)
    monkeypatch.setattr(main, "mode_env", "demo")
    monkeypatch.setattr(main, "oanda_env", "practice")
    monkeypatch.setenv("MODE", "demo")
    monkeypatch.setenv("OANDA_ENV", "practice")
    monkeypatch.setattr(main, "BOT_STATE", {})
    monkeypatch.setattr(main, "_SCHEDULER_REF", SimpleNamespace(running=True))
    monkeypatch.setattr(main, "_LAST_OPEN_TRADES_TS", None)
    monkeypatch.setattr(main, "_LAST_BROKER_SYNC_TS", None)
    monkeypatch.setattr(main, "_LAST_OPEN_TRADES_SNAPSHOT", [])
    monkeypatch.setattr(main, "_entry_window_state", lambda now: "weekend_locked")
    monkeypatch.setattr(main, "_safe_adaptive_snapshot", lambda context: None)
    monkeypatch.setattr(main, "_runtime_revision", lambda: "test-revision")
    monkeypatch.setattr(main, "journal", SimpleNamespace(
        path=tmp_path / "journal.sqlite", count_trade_events=lambda: 0,
        latest_verified_entry_timestamp=lambda: None,
    ))
    monkeypatch.delenv("MOSSY_MCP_STATUS_KEY", raising=False)
    published = []

    async def capture(payload, **kwargs):
        published.append(payload)
        return False, "disabled"

    monkeypatch.setattr(main, "publish_runtime_heartbeat", capture)
    asyncio.run(main.heartbeat())

    assert calls == ["close"]
    assert risk.state.demo_halted_on_equity_floor is True
    assert published[0]["demo_equity_floor_aud"] == 500.0
    assert published[0]["demo_equity_floor_halted"] is True
    assert published[0]["supervisor_floor_breached"] is True
    assert published[0]["entry_window_state"] == "weekend_locked"


def test_decision_cycle_retries_latched_stop_before_missing_nav_return(monkeypatch, tmp_path):
    risk = RiskManager({}, mode="paper", demo_mode=True, state_dir=tmp_path)
    risk.state.demo_halted_on_equity_floor = True
    risk._save_state()
    calls = []
    monkeypatch.setattr(main, "risk", risk)
    monkeypatch.setattr(main, "mode_env", "demo")
    monkeypatch.setattr(main, "broker", SimpleNamespace(account_equity=lambda: None))
    monkeypatch.setattr(main, "_close_demo_positions_and_verify", lambda: calls.append("close") or False)
    monkeypatch.setattr(main, "_utc_now", lambda: datetime(2026, 11, 10, 7, 0, tzinfo=timezone.utc))
    monkeypatch.setattr(main, "_ACTIVE_CYCLE_TICKS", set())
    monkeypatch.setattr(main, "_SUMMARY_EMITTED_TICKS", set())

    asyncio.run(main.decision_cycle())

    assert calls == ["close"]
    assert risk.state.demo_halted_on_equity_floor is True


def test_startup_retries_existing_demo_stop_even_when_nav_read_fails(monkeypatch, tmp_path):
    risk = RiskManager({}, mode="paper", demo_mode=True, state_dir=tmp_path)
    risk.state.demo_halted_on_equity_floor = True
    risk._save_state()
    calls = []

    def unavailable():
        raise RuntimeError("broker unavailable")

    monkeypatch.setattr(main, "risk", risk)
    monkeypatch.setattr(main, "mode_env", "demo")
    monkeypatch.setattr(main, "broker", SimpleNamespace(
        connectivity_check=lambda: {"ok": False}, account_equity=unavailable,
    ))
    monkeypatch.setattr(main, "_close_demo_positions_and_verify", lambda: calls.append("close") or False)

    main._startup_checks()

    assert calls == ["close"]
    assert risk.state.demo_halted_on_equity_floor is True
