import types

import pytest

from app import strategy


class DummyResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


@pytest.fixture(autouse=True)
def restore_strategy_state(monkeypatch):
    original_instrument = strategy.settings.INSTRUMENT
    original_httpx = strategy.httpx
    yield
    monkeypatch.setattr(strategy.settings, "INSTRUMENT", original_instrument)
    monkeypatch.setattr(strategy, "httpx", original_httpx)


def test_fetch_candles_uses_primary_instrument(monkeypatch, capsys):
    captured_url = {}

    def fake_get(url, **kwargs):
        captured_url["url"] = url
        return DummyResponse({"candles": []})

    monkeypatch.setattr(strategy.settings, "INSTRUMENT", "EUR_USD,AUD_USD")
    monkeypatch.setattr(strategy, "httpx", types.SimpleNamespace(get=fake_get))

    candles = strategy._fetch_candles()

    assert candles == []
    assert captured_url["url"].endswith("/instruments/EUR_USD/candles")
    captured = capsys.readouterr()
    assert "legacy strategy using primary EUR_USD" in captured.out


def test_fetch_candles_handles_blank_instrument(monkeypatch, capsys):
    monkeypatch.setattr(strategy.settings, "INSTRUMENT", " ,  ")
    monkeypatch.setattr(strategy, "httpx", types.SimpleNamespace(get=None))

    candles = strategy._fetch_candles()

    assert candles == []
    captured = capsys.readouterr()
    assert "No valid instrument configured" in captured.out
