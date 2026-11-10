from __future__ import annotations

from datetime import datetime, timezone
import threading

import pytest

from app.broker import Broker, PRACTICE


class Response:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code

    def json(self):
        return self.payload


class Client:
    def __init__(self, lists, details):
        self.lists = list(lists)
        self.details = details
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def get(self, path, params=None):
        self.calls.append(path.rsplit("/accounts/", 1)[-1].split("/", 1)[-1])
        if path.endswith("/openTrades"):
            payload = self.lists.pop(0)
            if isinstance(payload, Exception):
                raise payload
            if isinstance(payload, Response):
                return payload
            return Response({"trades": payload})
        trade_id = path.rsplit("/", 1)[-1]
        payload = self.details[trade_id]
        if isinstance(payload, Exception):
            raise payload
        return Response({"trade": payload})


def summary(trade_id="1"):
    return {"id": trade_id, "instrument": "AUD_USD", "stopLossOrderID": f"{trade_id}01"}


def detail(trade_id="1"):
    return {
        "id": trade_id, "instrument": "AUD_USD", "state": "OPEN",
        "currentUnits": "10", "price": "1.20000",
        "openTime": "2026-11-10T07:00:00Z",
        "stopLossOrder": {
            "id": f"{trade_id}01", "state": "PENDING", "type": "STOP_LOSS",
            "tradeID": trade_id, "price": "1.19000", "clientExtensions": {"secret": "excluded"},
        },
        "takeProfitOrder": {
            "id": f"{trade_id}02", "state": "PENDING", "type": "TAKE_PROFIT",
            "tradeID": trade_id, "price": "1.22000", "accountID": "excluded",
        },
        "accountID": "excluded", "clientExtensions": {"secret": "excluded"},
    }


@pytest.fixture
def broker(tmp_path, monkeypatch):
    # __new__ is also used by existing broker diagnostic fakes.
    value = Broker.__new__(Broker)
    value.mode = "demo"
    value.base_url = PRACTICE
    value.account = "test-account"
    value.key = "test-key"
    value._account_currency = "AUD"
    value._entry_halted_reason = None
    value._entry_halt_path = tmp_path / "halt.txt"
    monkeypatch.setenv("MAX_RISK_PER_TRADE_CCY", "0.50")
    monkeypatch.setattr(value, "conversion_rate", lambda source, target: 1.0)
    return value


def test_verified_detail_is_allowlisted_copied_and_requires_no_extra_http(broker, monkeypatch):
    original = detail()
    rows = [summary()]
    client = Client([rows], {"1": original})
    monkeypatch.setattr(broker, "_client", lambda: client)

    assert broker.list_open_trades() == rows
    snapshot = broker.demo_trade_handoff_evidence()
    assert client.calls == ["openTrades", "trades/1"]
    assert snapshot["account_currency"] == "AUD"
    observed = datetime.fromisoformat(snapshot["broker_observed_at_utc"])
    assert observed.tzinfo is not None
    assert observed.astimezone(timezone.utc) <= datetime.now(timezone.utc)
    trade = snapshot["trade_details"]["1"]
    assert set(trade) == {
        "id", "instrument", "state", "currentUnits", "price", "openTime",
        "stopLossOrder", "takeProfitOrder",
    }
    assert set(trade["stopLossOrder"]) == {"id", "state", "type", "tradeID", "price"}
    assert set(trade["takeProfitOrder"]) == {"id", "state", "type", "tradeID", "price"}
    assert trade["stopLossOrder"]["price"] == "1.19000"
    assert trade["takeProfitOrder"]["price"] == "1.22000"

    snapshot["trade_details"]["1"]["stopLossOrder"]["price"] = "changed"
    original["takeProfitOrder"]["price"] = "changed"
    rows[0]["instrument"] = "changed"
    second = broker.demo_trade_handoff_evidence()
    assert second["trade_details"]["1"]["stopLossOrder"]["price"] == "1.19000"
    assert second["trade_details"]["1"]["takeProfitOrder"]["price"] == "1.22000"
    assert second["trade_details"]["1"]["instrument"] == "AUD_USD"
    assert client.calls == ["openTrades", "trades/1"]


@pytest.mark.parametrize("take_profit,present", [("missing", False), (None, True), ({"price": []}, True)])
def test_missing_take_profit_is_distinct_from_invalid_present_value(broker, monkeypatch, take_profit, present):
    trade = detail()
    if take_profit == "missing":
        trade.pop("takeProfitOrder")
    else:
        trade["takeProfitOrder"] = take_profit
    client = Client([[summary()]], {"1": trade})
    monkeypatch.setattr(broker, "_client", lambda: client)
    broker.list_open_trades()
    recorded = broker.demo_trade_handoff_evidence()["trade_details"]["1"]
    assert ("takeProfitOrder" in recorded) is present
    if take_profit is None:
        assert recorded["takeProfitOrder"] is None
    if isinstance(take_profit, dict):
        assert recorded["takeProfitOrder"] == {"price": None}


def test_supported_guaranteed_stop_is_preserved(broker, monkeypatch):
    trade = detail()
    trade["guaranteedStopLossOrder"] = trade.pop("stopLossOrder")
    trade["guaranteedStopLossOrder"]["type"] = "GUARANTEED_STOP_LOSS"
    row = {"id": "1", "instrument": "AUD_USD", "guaranteedStopLossOrderID": "101"}
    client = Client([[row]], {"1": trade})
    monkeypatch.setattr(broker, "_client", lambda: client)
    assert broker.list_open_trades() == [row]
    evidence = broker.demo_trade_handoff_evidence()["trade_details"]["1"]
    assert "stopLossOrder" not in evidence
    assert evidence["guaranteedStopLossOrder"]["type"] == "GUARANTEED_STOP_LOSS"


def test_closed_during_audit_refresh_discards_earlier_details(broker, monkeypatch):
    closed = {"id": "2", "state": "CLOSED", "instrument": "AUD_USD", "currentUnits": "0"}
    client = Client([[summary("1"), summary("2")], [summary("3")]], {
        "1": detail("1"), "2": closed, "3": detail("3"),
    })
    monkeypatch.setattr(broker, "_client", lambda: client)
    assert broker.list_open_trades() == [summary("3")]
    assert set(broker.demo_trade_handoff_evidence()["trade_details"]) == {"3"}
    assert client.calls == ["openTrades", "trades/1", "trades/2", "openTrades", "trades/3"]


def test_flat_read_is_confirmed_empty_and_unknown_read_clears_it(broker, monkeypatch):
    client = Client([[], Response({}, 503)], {})
    monkeypatch.setattr(broker, "_client", lambda: client)
    assert broker.list_open_trades() == []
    assert broker.demo_trade_handoff_evidence()["trade_details"] == {}
    assert broker.list_open_trades() is None
    assert broker.demo_trade_handoff_evidence() is None
    assert client.calls == ["openTrades", "openTrades"]


@pytest.mark.parametrize("failure", ["missing-creds", "simulation", "http-error", "exception", "malformed"])
def test_every_new_list_read_clears_old_evidence(broker, monkeypatch, failure):
    client = Client([[summary()]], {"1": detail()})
    monkeypatch.setattr(broker, "_client", lambda: client)
    assert broker.list_open_trades() == [summary()]
    assert broker.demo_trade_handoff_evidence() is not None
    if failure == "missing-creds":
        broker.key = ""
    elif failure == "simulation":
        broker.mode = "simulation"
    else:
        next_payload = {
            "http-error": Response({}, 500),
            "exception": RuntimeError("unavailable"),
            "malformed": Response({"trades": "unknown"}),
        }[failure]
        client.lists.append(next_payload)
    broker.list_open_trades()
    assert broker.demo_trade_handoff_evidence() is None


def test_failed_protection_audit_does_not_publish_partial_or_old_evidence(broker, monkeypatch):
    first = detail("1")
    bad = detail("2")
    bad["stopLossOrder"]["state"] = "CANCELLED"
    client = Client([[summary("1")], [summary("1"), summary("2")]], {"1": first, "2": bad})
    monkeypatch.setattr(broker, "_client", lambda: client)
    closes = []
    monkeypatch.setattr(broker, "_close_exact_trade", lambda client, ticket: closes.append(ticket) or False)
    broker.list_open_trades()
    assert broker.demo_trade_handoff_evidence() is not None
    assert broker.list_open_trades() is None
    assert broker.demo_trade_handoff_evidence() is None
    assert closes == ["2"]
    assert broker.entry_halt_reason == "unprotected-open-trade"


@pytest.mark.parametrize("rows", [[], [summary()]])
def test_unaudited_return_cannot_publish_evidence(broker, monkeypatch, rows):
    client = Client([rows], {"1": detail()})
    monkeypatch.setattr(broker, "_client", lambda: client)
    monkeypatch.setattr(broker, "_audit_open_trade_protection", lambda client, rows: rows)
    assert broker.list_open_trades() == rows
    assert broker.demo_trade_handoff_evidence() is None


def test_other_audit_paths_do_not_publish_or_retain_list_evidence(broker, monkeypatch):
    client = Client([[summary()]], {"1": detail()})
    monkeypatch.setattr(broker, "_client", lambda: client)
    broker.list_open_trades()
    assert broker.demo_trade_handoff_evidence() is not None
    assert broker._audit_open_trade_protection(client, [summary()]) == [summary()]
    assert broker.demo_trade_handoff_evidence() is None


def test_account_currency_is_not_assumed_for_flat_read(broker, monkeypatch):
    broker._account_currency = None
    client = Client([[]], {})
    monkeypatch.setattr(broker, "_client", lambda: client)
    assert broker.list_open_trades() == []
    assert broker.demo_trade_handoff_evidence()["account_currency"] is None


def test_changed_account_scope_cannot_reuse_snapshot(broker, monkeypatch):
    client = Client([[]], {})
    monkeypatch.setattr(broker, "_client", lambda: client)
    broker.list_open_trades()
    assert broker.demo_trade_handoff_evidence() is not None
    broker.account = "different-test-account"
    assert broker.demo_trade_handoff_evidence() is None


def test_an_older_concurrent_read_cannot_replace_a_later_failure(broker, monkeypatch):
    first_client = Client([[summary()]], {"1": detail()})
    second_client = Client([Response({}, 503)], {})
    ready, release = threading.Event(), threading.Event()
    original_audit = broker._audit_open_trade_protection

    def audit(client, rows):
        result = original_audit(client, rows)
        if threading.current_thread().name == "older-evidence-read":
            ready.set()
            assert release.wait(3)
        return result

    monkeypatch.setattr(broker, "_audit_open_trade_protection", audit)
    monkeypatch.setattr(broker, "_client", lambda: (
        first_client if threading.current_thread().name == "older-evidence-read" else second_client
    ))
    returned = []
    thread = threading.Thread(target=lambda: returned.append(broker.list_open_trades()), name="older-evidence-read")
    thread.start()
    try:
        assert ready.wait(3)
        assert broker.list_open_trades() is None
    finally:
        release.set()
        thread.join(3)
    assert not thread.is_alive()
    assert returned == [[summary()]]
    assert broker.demo_trade_handoff_evidence() is None
