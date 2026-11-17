"""Recovery must prove this incident closed and never weaken later safeguards."""
import json
import os
import sqlite3
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from app.broker import Broker, PRACTICE
from app.config import settings
from src.protection_recovery import REASON, recover_reviewed_incident


@pytest.fixture
def recovery(tmp_path, monkeypatch):
    now = datetime(2026, 11, 17, 12, tzinfo=timezone.utc)
    incident = now - timedelta(days=1)
    monkeypatch.setattr(settings, "MODE", "demo")
    monkeypatch.setattr(settings, "OANDA_ENV", "practice")
    monkeypatch.setattr(settings, "OANDA_ACCOUNT_ID", "private-account")
    for key, value in {"MOSSY_STATE_PATH": str(tmp_path), "MAX_RISK_PER_TRADE_CCY": ".20",
                       "HARD_MAX_LOSS_CCY": ".20", "MAX_CONCURRENT_POSITIONS": "1",
                       "MAX_OPEN_TRADES": "1"}.items():
        monkeypatch.setenv(key, value)
    broker = Broker()
    broker._latch_entry_halt(REASON)
    os.utime(broker._entry_halt_path, (incident.timestamp(), incident.timestamp()))
    request = {"trade_id": "100", "stop_order_id": "102", "incident_at_utc": incident.isoformat(),
               "expires_at_utc": (now + timedelta(minutes=30)).isoformat()}
    opened = (incident - timedelta(minutes=2)).isoformat()
    closed = (incident + timedelta(seconds=1)).isoformat()
    opening = {"id": "100", "type": "ORDER_FILL", "accountID": broker.account,
               "instrument": "GBP_USD", "time": opened,
               "tradeOpened": {"tradeID": "100", "units": "-437", "price": "1.32053"}}
    closing = {"id": "105", "type": "ORDER_FILL", "accountID": broker.account,
               "instrument": "GBP_USD", "time": closed, "tradesClosed": [
                   {"tradeID": "100", "units": "437", "price": "1.32045", "realizedPL": "0.05"}]}
    payloads = {
        "summary": {"account": {"id": broker.account, "currency": "AUD", "openTradeCount": 0,
                                "openPositionCount": 0, "pendingOrderCount": 0}},
        "openTrades": {"trades": []}, "openPositions": {"positions": []},
        "pendingOrders": {"orders": []},
        "transactions/100": {"transaction": opening},
        "transactions/idrange": {"transactions": [opening, closing]},
        "orders/102": {"order": {"id": "102", "type": "STOP_LOSS", "state": "CANCELLED",
                                  "tradeID": "100", "price": "1.32083"}},
    }
    for payload in payloads.values():
        payload["lastTransactionID"] = "107"
    calls = []
    def handle(req):
        calls.append(req)
        assert req.method == "GET"  # Recovery can never create or close an order.
        assert req.url.host == "api-fxpractice.oanda.com"
        suffix = req.url.path.split(f"/accounts/{broker.account}/", 1)[1]
        return httpx.Response(200, json=payloads[suffix])
    monkeypatch.setattr(broker, "_client", lambda: httpx.Client(base_url=PRACTICE,
                        transport=httpx.MockTransport(handle)))
    dbpath = tmp_path / "journal.db"
    with sqlite3.connect(dbpath) as db:
        db.execute("CREATE TABLE trades (trade_id TEXT, instrument TEXT, units REAL, entry_price REAL, "
                   "stop_loss_price REAL, exit_timestamp_utc TEXT, exit_price REAL, realized_pnl_ccy REAL, "
                   "broker_confirmed INTEGER, side TEXT)")
        db.execute("INSERT INTO trades VALUES (?,?,?,?,?,?,?,?,?,?)",
                   ("100", "GBP_USD", 437, 1.32053, 1.32083, closed, 1.32045, .05, 1, "SELL"))
    risk = tmp_path / "risk_state.json"
    risk.write_text('{"daily_loss":0.1,"max_drawdown_halt":true}')
    return broker, dbpath, request, now, payloads, calls, risk


def run(fixture):
    broker, db, request, now, *_ = fixture
    return recover_reviewed_incident(broker, db, json.dumps(request), now=now)


def test_reviewed_recovery_archives_exact_incident_and_preserves_risk_and_journal(recovery):
    broker, db, request, now, payloads, calls, risk = recovery
    original_db, original_risk = db.read_bytes(), risk.read_bytes()
    assert run(recovery)["status"] == "recovered"
    assert broker.entry_halt_reason is None
    assert not broker._entry_halt_path.exists()
    assert list(db.parent.glob("broker_entry_halt.recovered-*.txt"))[0].read_text().strip() == REASON
    assert db.read_bytes() == original_db and risk.read_bytes() == original_risk
    assert len(calls) == 11
    # A retained environment request must never release a future incident.
    broker._latch_entry_halt(REASON)
    assert run(recovery)["reason"] == "different-halt-incident"
    assert broker.entry_halt_reason == REASON
    # Even a subsequent incident inside the timestamp tolerance cannot replay it.
    incident = datetime.fromisoformat(request["incident_at_utc"]).timestamp()
    os.utime(broker._entry_halt_path, (incident + 1, incident + 1))
    assert run(recovery)["reason"] == "request-already-consumed"


@pytest.mark.parametrize("suffix,key,value", [
    ("openTrades", "trades", [{"id": "200"}]),
    ("openPositions", "positions", [{"instrument": "AUD_USD"}]),
    ("pendingOrders", "orders", [{"id": "201"}]),
    ("openTrades", "lastTransactionID", "108"),
    ("transactions/idrange", "transactions", []),
    ("orders/102", "order", {"id": "102", "state": "PENDING"}),
])
def test_incomplete_or_conflicting_broker_evidence_preserves_halt(recovery, suffix, key, value):
    recovery[4][suffix][key] = value
    assert run(recovery)["status"] == "blocked"
    assert recovery[0]._entry_halt_path.read_text().strip() == REASON
    assert recovery[0].entry_halt_reason == REASON


@pytest.mark.parametrize("field,value", [("broker_confirmed", 0), ("side", "BUY"),
                                        ("units", 438), ("realized_pnl_ccy", 10),
                                        ("stop_loss_price", 1.32), ("exit_timestamp_utc", "invalid")])
def test_journal_mismatch_prevents_recovery(recovery, field, value):
    with sqlite3.connect(recovery[1]) as db:
        db.execute(f"UPDATE trades SET {field}=?", (value,))
    assert run(recovery)["status"] == "blocked"
    assert recovery[0].entry_halt_reason == REASON


@pytest.mark.parametrize("change", ["expired", "other-incident", "live", "cash-cap", "multiple", "other-reason"])
def test_request_and_configuration_guards_block_before_broker_access(recovery, monkeypatch, change):
    broker, _, request, now, _, calls, _ = recovery
    if change == "expired": request["expires_at_utc"] = now.isoformat()
    if change == "other-incident": request["incident_at_utc"] = (now - timedelta(hours=2)).isoformat()
    if change == "live": broker.base_url = "https://api-fxtrade.oanda.com"
    if change == "cash-cap": monkeypatch.setenv("MAX_RISK_PER_TRADE_CCY", ".50")
    if change == "multiple": monkeypatch.setenv("MAX_OPEN_TRADES", "2")
    if change == "other-reason": broker._latch_entry_halt("unprotected-open-trade")
    assert run(recovery)["status"] == "blocked"
    assert not calls
    assert broker._entry_halt_path.exists()


def test_no_request_cannot_touch_halt_or_broker(recovery):
    broker, db, *_ = recovery
    assert recover_reviewed_incident(broker, db, "")["status"] == "not-requested"
    assert recovery[5] == [] and broker.entry_halt_reason == REASON


def test_profit_guard_optional_exit_price_uses_exact_transaction_evidence(recovery):
    with sqlite3.connect(recovery[1]) as db:
        db.execute("UPDATE trades SET exit_price=NULL")
    result = run(recovery)
    assert result["status"] == "recovered" and not result["journal_exit_price_present"]


def test_halt_modified_during_checks_is_never_released(recovery, monkeypatch):
    import src.protection_recovery as module
    original = module._flat
    def mutate(client, account):
        result = original(client, account)
        recovery[0]._latch_entry_halt(REASON)
        return result
    monkeypatch.setattr(module, "_flat", mutate)
    assert run(recovery)["reason"] == "halt-changed-during-recovery"
    assert recovery[0]._entry_halt_path.exists()


def test_errors_cannot_leak_response_secrets(recovery, monkeypatch, capsys):
    def broken(): raise RuntimeError("Bearer private-key private-account")
    monkeypatch.setattr(recovery[0], "_client", broken)
    assert run(recovery)["status"] == "blocked"
    output = capsys.readouterr().out
    assert "private-key" not in output and "private-account" not in output
