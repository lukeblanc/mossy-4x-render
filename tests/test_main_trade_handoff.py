from __future__ import annotations

from datetime import datetime, timezone
import json
from types import SimpleNamespace

import pytest

import src.main as main


NOW = datetime(2026, 11, 10, 7, 0, tzinfo=timezone.utc)
REVISION = "a" * 40


def _detail():
    return {
        "id": "7", "instrument": "EUR_USD", "state": "OPEN",
        "currentUnits": "100", "price": "1.10",
        "openTime": "2026-11-10T06:59:00Z",
        "stopLossOrder": {"id": "8", "tradeID": "7", "type": "STOP_LOSS", "state": "PENDING", "price": "1.09"},
        "takeProfitOrder": {"id": "9", "tradeID": "7", "type": "TAKE_PROFIT", "state": "PENDING", "price": "1.12"},
    }


def _setup(monkeypatch, evidence):
    monkeypatch.setattr(main, "mode_env", "demo")
    monkeypatch.setattr(main, "oanda_env", "practice")
    monkeypatch.setattr(main, "_runtime_revision", lambda: REVISION)
    monkeypatch.setattr(main, "risk", SimpleNamespace(
        demo_equity_floor_aud=500.0,
        state=SimpleNamespace(demo_halted_on_equity_floor=False),
    ))
    monkeypatch.setattr(main, "broker", SimpleNamespace(demo_trade_handoff_evidence=lambda: evidence))


def test_private_logger_uses_exact_audited_evidence_and_source_timestamp(monkeypatch, capsys):
    evidence = {
        "broker_observed_at_utc": "2026-11-10T06:59:59Z", "account_currency": "AUD",
        "trade_details": {"7": _detail()},
        "account_id": "must-not-be-logged", "api_key": "must-not-be-logged",
    }
    _setup(monkeypatch, evidence)
    monkeypatch.setattr(main, "BOT_STATE", {"status": "running"})
    before = dict(main.BOT_STATE)
    payload = main._log_private_demo_trade_handoff(
        [{"id": "7", "instrument": "EUR_USD"}], NOW, broker_entry_halted=False,
    )
    line = capsys.readouterr().out.strip()
    assert line.startswith("[DEMO_TRADE_HANDOFF] ")
    assert json.loads(line.split(" ", 1)[1]) == payload
    assert payload["handoff_ready"] is True
    assert payload["destination_order_ready"] is False
    assert payload["broker_observed_at_utc"] == "2026-11-10T06:59:59Z"
    assert payload["trades"][0]["source_trade_id"] == "7"
    assert "must-not-be-logged" not in line
    assert main.BOT_STATE == before


@pytest.mark.parametrize("snapshot,expected", [([], "flat"), (None, "unavailable")])
def test_private_logger_distinguishes_confirmed_flat_from_unavailable(monkeypatch, snapshot, expected):
    evidence = None if snapshot is None else {
        "broker_observed_at_utc": NOW.isoformat(), "account_currency": "AUD", "trade_details": {},
    }
    _setup(monkeypatch, evidence)
    payload = main._log_private_demo_trade_handoff(snapshot, NOW, broker_entry_halted=False)
    assert payload["read_status"] == expected
    assert payload["handoff_ready"] is False
    assert payload["trades"] == []


def test_getter_failure_logs_unavailable_instead_of_reusing_a_trade(monkeypatch, capsys):
    _setup(monkeypatch, None)

    def fail():
        raise RuntimeError("do-not-log-sensitive-error-content")

    monkeypatch.setattr(main.broker, "demo_trade_handoff_evidence", fail)
    payload = main._log_private_demo_trade_handoff(
        [{"id": "7", "instrument": "EUR_USD"}], NOW, broker_entry_halted=False,
    )
    assert payload["read_status"] == "unavailable"
    assert payload["handoff_ready"] is False
    assert "do-not-log-sensitive-error-content" not in capsys.readouterr().out


@pytest.mark.parametrize("mode,environment", [("live", "practice"), ("demo", "live"), ("simulation", "practice")])
def test_private_handoff_logging_only_runs_for_explicit_demo_practice(monkeypatch, capsys, mode, environment):
    _setup(monkeypatch, None)
    monkeypatch.setattr(main, "mode_env", mode)
    monkeypatch.setattr(main, "oanda_env", environment)
    assert main._log_private_demo_trade_handoff([], NOW, broker_entry_halted=False) is None
    assert capsys.readouterr().out == ""


def test_old_audited_evidence_is_not_restamped_as_a_fresh_trade(monkeypatch):
    _setup(monkeypatch, {
        "broker_observed_at_utc": "2026-11-10T06:58:00Z", "account_currency": "AUD",
        "trade_details": {"7": _detail()},
    })
    payload = main._log_private_demo_trade_handoff(
        [{"id": "7", "instrument": "EUR_USD"}], NOW, broker_entry_halted=False,
    )
    assert payload["freshness"] == "stale"
    assert payload["handoff_ready"] is False
    assert payload["trades"] == []
