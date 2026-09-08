from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app.broker import Broker
from app.config import settings
from src import main
from src.decision_engine import Evaluation


@pytest.fixture
def runtime(monkeypatch, tmp_path):
    broker = Mock()
    broker.account_equity.return_value = 10000.0
    broker.list_open_trades.return_value = []
    broker.current_spread.return_value = 0.1
    risk = Mock()
    guard = Mock()
    guard.process_open_trades.return_value = []
    engine = Mock()
    engine.evaluate_all.return_value = [Evaluation(
        instrument="EUR_USD", signal="BUY", diagnostics={}, reason="trend",
        market_active=True, candles=[],
    )]
    for name, value in {
        "broker": broker, "risk": risk, "profit_guard": guard, "engine": engine,
        "BOT_STATE": {}, "_ACTIVE_CYCLE_TICKS": set(),
        "_SUMMARY_EMITTED_TICKS": set(), "_LAST_BROKER_SYNC_TS": None,
        "_LAST_OPEN_TRADES_TS": None, "_LAST_OPEN_TRADES_SNAPSHOT": [],
    }.items():
        monkeypatch.setattr(main, name, value)
    monkeypatch.setattr(main, "_safe_adaptive_snapshot", lambda context: None)
    monkeypatch.setattr(main, "_log_projector", lambda *args: None)
    monkeypatch.setattr(main.session_filter, "current_session", lambda *args, **kw: None)
    monkeypatch.setattr(main.session_filter, "session_decision", lambda *args, **kw:
                        SimpleNamespace(allowed=True, session=None, in_session=False))
    journal = Mock()
    journal.path = tmp_path / "journal.sqlite"
    journal.count_trade_events.return_value = 0
    monkeypatch.setattr(main, "journal", journal)
    return broker, risk, guard, engine


@pytest.mark.parametrize("failed_read", ["account_equity", "list_open_trades", "current_spread"])
def test_failed_read_blocks_entry_and_next_cycle_can_run(runtime, failed_read):
    broker, risk, guard, engine = runtime
    getattr(broker, failed_read).return_value = None
    asyncio.run(main.decision_cycle())
    broker.place_order.assert_not_called()
    risk.should_open.assert_not_called()
    assert not main._ACTIVE_CYCLE_TICKS
    if failed_read == "list_open_trades":
        guard.process_open_trades.assert_not_called()

    broker.account_equity.return_value = 10000.0
    broker.list_open_trades.return_value = []
    broker.current_spread.return_value = 0.1
    engine.evaluate_all.return_value = []
    asyncio.run(main.decision_cycle())
    assert engine.evaluate_all.call_count == (1 if failed_read == "account_equity" else 2)
    assert not main._ACTIVE_CYCLE_TICKS


@pytest.mark.parametrize("failure", [None, RuntimeError("offline"), AttributeError("no reader")])
def test_failed_refresh_preserves_last_success_and_invalidates_cache(runtime, monkeypatch, failure):
    broker, *_ = runtime
    last_success = datetime.now(timezone.utc) - timedelta(seconds=5)
    monkeypatch.setattr(main, "_LAST_BROKER_SYNC_TS", last_success)
    monkeypatch.setattr(main, "_LAST_OPEN_TRADES_TS", last_success)
    monkeypatch.setattr(main, "_LAST_OPEN_TRADES_SNAPSHOT", [{"instrument": "EUR_USD"}])
    if isinstance(failure, Exception):
        broker.list_open_trades.side_effect = failure
    else:
        broker.list_open_trades.return_value = None
    assert main._open_trades_state() is None
    assert main._LAST_BROKER_SYNC_TS == last_success
    assert main._LAST_OPEN_TRADES_TS is None
    assert main._health_status()["open_trades_count"] is None
    assert main._instrument_open_on_broker("EUR_USD") is True


def test_heartbeat_survives_missing_equity_and_positions(runtime):
    broker, *_ = runtime
    broker.account_equity.return_value = None
    broker.list_open_trades.return_value = None
    asyncio.run(main.heartbeat())
    assert main.BOT_STATE["status"] == "broker-unavailable"
    assert main.BOT_STATE["equity"] is None
    assert main.BOT_STATE["open_trades"] is None
    assert main.BOT_STATE["last_heartbeat"]


@pytest.mark.parametrize("failed_read", ["account_equity", "list_open_trades"])
def test_startup_does_not_reset_risk_with_unknown_broker_state(runtime, monkeypatch, failed_read):
    broker, risk, *_ = runtime
    monkeypatch.setenv("RESET_MAX_DRAWDOWN_HALT", "true")
    monkeypatch.setenv("RESET_WEEKLY_LOSS_CAP", "true")
    monkeypatch.setenv("MOSSY_DEMO_RUN_ID", "explicit-demo-test")
    getattr(broker, failed_read).return_value = None
    main._startup_checks()
    risk.startup_daily_reset.assert_not_called()
    risk.clear_max_drawdown_halt.assert_not_called()
    risk.clear_weekly_loss_cap.assert_not_called()
    risk.start_demo_run.assert_not_called()


def test_startup_applies_explicit_demo_run_after_broker_reads(runtime, monkeypatch):
    broker, risk, *_ = runtime
    monkeypatch.setenv("MOSSY_DEMO_RUN_ID", "explicit-demo-test")
    monkeypatch.setattr(main, "oanda_env", "practice")
    risk.start_demo_run.return_value = (True, "applied")
    main._startup_checks()
    broker.list_open_trades.assert_called_once()
    risk.start_demo_run.assert_called_once_with(
        "explicit-demo-test", 10000.0, open_positions_count=0, oanda_env="practice",
    )


@pytest.mark.parametrize("method,payload,expected", [
    ("list_open_trades", {}, None),
    ("list_open_trades", {"trades": [None]}, None),
    ("list_open_trades", {"trades": []}, []),
    ("account_equity", {"account": {"balance": "10000"}}, None),
    ("account_equity", {"account": {"NAV": "NaN"}}, None),
    ("account_equity", {"account": {"NAV": "Infinity"}}, None),
    ("account_equity", {"account": {"NAV": "0", "balance": "10000"}}, None),
    ("account_equity", {"account": {"NAV": "10000"}}, 10000.0),
    ("current_spread", {"prices": [{"bids": [{"price": "NaN"}], "asks": [{"price": "1.2"}]}]}, None),
    ("current_spread", {"prices": [{"bids": [{"price": "1.3"}], "asks": [{"price": "1.2"}]}]}, None),
])
def test_malformed_success_responses_are_unavailable(monkeypatch, method, payload, expected):
    monkeypatch.setattr(settings, "MODE", "demo")
    monkeypatch.setattr(settings, "OANDA_ENV", "practice")
    monkeypatch.setattr(settings, "OANDA_API_KEY", "test-only")
    monkeypatch.setattr(settings, "OANDA_ACCOUNT_ID", "test-only")
    broker = Broker()
    response = SimpleNamespace(status_code=200, json=lambda: payload)
    client = SimpleNamespace(get=lambda *args, **kwargs: response)
    from contextlib import nullcontext
    monkeypatch.setattr(broker, "_client", lambda: nullcontext(client))
    args = ("EUR_USD",) if method == "current_spread" else ()
    assert getattr(broker, method)(*args) == expected


def test_missing_credentials_do_not_confirm_no_positions(monkeypatch):
    monkeypatch.setattr(settings, "MODE", "demo")
    monkeypatch.setattr(settings, "OANDA_ENV", "practice")
    monkeypatch.setattr(settings, "OANDA_API_KEY", "")
    assert Broker().list_open_trades() is None


@pytest.mark.parametrize("failed_read", ["account_equity", "list_open_trades", "current_spread"])
def test_legacy_entrypoint_blocks_unknown_reads(runtime, monkeypatch, failed_read):
    from app import main as legacy
    broker, risk, guard, _ = runtime
    monkeypatch.setattr(legacy, "broker", broker)
    monkeypatch.setattr(legacy, "risk", risk)
    monkeypatch.setattr(legacy, "profit_guard", guard)
    decide = Mock(side_effect=AssertionError("must stop before strategy"))
    monkeypatch.setattr(legacy, "decide", decide)
    getattr(broker, failed_read).return_value = None
    asyncio.run(legacy.decision_tick())
    broker.place_order.assert_not_called()
    decide.assert_not_called()
