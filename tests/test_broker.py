from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.broker import Broker
from app.config import settings


class DummyResponse:
    status_code = 201

    @staticmethod
    def json():
        return {"orderCreateTransaction": {"id": "1"}}


class DummyClient:
    def __init__(self, recorder):
        self.recorder = recorder

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def post(self, path: str, json):
        self.recorder["path"] = path
        self.recorder["payload"] = json
        return DummyResponse()


def _configure_settings(monkeypatch):
    monkeypatch.setattr(settings, "OANDA_API_KEY", "token")
    monkeypatch.setattr(settings, "OANDA_ACCOUNT_ID", "acct-123")
    monkeypatch.setattr(settings, "OANDA_ENV", "practice")
    monkeypatch.setattr(settings, "MODE", "demo")


def test_place_order_uses_absolute_tp_price_for_buy(monkeypatch):
    _configure_settings(monkeypatch)
    recorded = {}
    monkeypatch.setattr(Broker, "_client", lambda self: DummyClient(recorded))

    broker = Broker()
    result = broker.place_order(
        "EUR_USD",
        "BUY",
        1000,
        sl_distance=0.00123,
        tp_distance=0.005,
        entry_price=1.2000,
    )

    assert result["status"] == "SENT"
    order = recorded["payload"]["order"]
    assert order["units"] == "1000"
    assert order["stopLossOnFill"]["distance"] == "0.00123"
    assert order["takeProfitOnFill"]["price"] == "1.20500"
    assert "distance" not in order["takeProfitOnFill"]


def test_place_order_uses_absolute_tp_price_for_sell(monkeypatch):
    _configure_settings(monkeypatch)
    recorded = {}
    monkeypatch.setattr(Broker, "_client", lambda self: DummyClient(recorded))

    broker = Broker()
    result = broker.place_order(
        "EUR_USD",
        "SELL",
        500,
        sl_distance=0.00100,
        tp_distance=0.005,
        entry_price=1.2000,
    )

    assert result["status"] == "SENT"
    order = recorded["payload"]["order"]
    assert order["units"] == "-500"
    assert order["stopLossOnFill"]["distance"] == "0.00100"
    assert order["takeProfitOnFill"]["price"] == "1.19500"
    assert "distance" not in order["takeProfitOnFill"]


def test_usd_jpy_tp_price_is_rounded(monkeypatch, capsys):
    _configure_settings(monkeypatch)
    recorded = {}
    monkeypatch.setattr(Broker, "_client", lambda self: DummyClient(recorded))

    broker = Broker()
    result = broker.place_order(
        "USD_JPY",
        "BUY",
        1000,
        sl_distance=0.123,
        tp_distance=0.00432,
        entry_price=156.16487,
    )

    assert result["status"] == "SENT"
    order = recorded["payload"]["order"]
    assert order["takeProfitOnFill"]["price"] == "156.169"

    logs = capsys.readouterr().out
    assert "[ORDER_FMT] instrument=USD_JPY raw_tp=156.16919 rounded_tp=156.169" in logs
