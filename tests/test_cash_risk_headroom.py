from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace

import pytest

from app.broker import Broker
from app.config import settings
from src import adaptive_policy, position_sizer
from src.cash_risk import configured_cash_risk_limit


@pytest.fixture(autouse=True)
def neutral_learning_and_cap(monkeypatch):
    monkeypatch.setenv("MAX_RISK_PER_TRADE_CCY", "0.50")
    monkeypatch.setattr(
        adaptive_policy,
        "evaluate_instrument_policy",
        lambda instrument: adaptive_policy.PolicyDecision(
            instrument=instrument,
            setup_key="test-neutral",
            risk_scale=1.0,
            blocked=False,
            reason="test-neutral",
        ),
    )


def _size(*, conversion=1.0, **overrides):
    arguments = dict(
        equity=1000.0,
        instrument="AUD_USD",
        stop_distance=0.001,
        risk_pct=0.005,
        broker=SimpleNamespace(conversion_rate=lambda *_: conversion),
    )
    arguments.update(overrides)
    return position_sizer.units_for_risk(**arguments)


@pytest.mark.parametrize(
    "instrument,account_currency",
    [("USD_AUD", "AUD"), ("AUD_USD", "USD"), ("AUD_USD", "usd")],
)
def test_same_currency_stop_exposure_needs_no_conversion_reserve(
    instrument, account_currency
):
    units, diagnostics = _size(
        instrument=instrument, account_currency=account_currency
    )

    assert units == 500
    assert diagnostics["risk_amount"] == 0.50
    assert diagnostics["cash_risk_sizing_limit"] == 0.50
    assert diagnostics["conversion_risk_reserve_pct"] == 0.0


def test_smaller_percentage_request_is_not_reduced_by_conversion_reserve():
    units, diagnostics = _size(risk_pct=0.0001)

    assert units == 100
    assert diagnostics["requested_risk_pct"] == 0.0001
    assert diagnostics["risk_amount"] == pytest.approx(0.10)
    assert diagnostics["cash_risk_sizing_limit"] == 0.49


def test_learning_reduction_still_limits_smaller_percentage_request(monkeypatch):
    monkeypatch.setattr(
        adaptive_policy,
        "evaluate_instrument_policy",
        lambda instrument: adaptive_policy.PolicyDecision(
            instrument=instrument,
            setup_key="test-reduced",
            risk_scale=0.5,
            blocked=False,
            reason="test-reduced",
        ),
    )
    units, diagnostics = _size(risk_pct=0.0001)

    assert units == 50
    assert diagnostics["requested_risk_pct"] == 0.0001
    assert diagnostics["risk_amount"] == pytest.approx(0.05)


def test_minimum_unit_cannot_consume_conversion_reserve():
    # One unit would fit the A$0.50 audit ceiling, but not the A$0.49 budget.
    units, diagnostics = _size(stop_distance=0.495, min_trade_units=1)

    assert units == 0
    assert diagnostics["reason"] == "minimum-units-exceed-risk-budget"
    assert diagnostics["cash_risk_sizing_limit"] == 0.49
    assert diagnostics["max_risk_per_trade_ccy"] == 0.50


def test_conversion_reserve_cannot_be_disabled_by_environment(monkeypatch):
    monkeypatch.setenv("CROSS_CURRENCY_SIZING_RESERVE_PCT", "0")
    units, diagnostics = _size()

    assert units == 490
    assert diagnostics["conversion_risk_reserve_pct"] == 0.02
    assert configured_cash_risk_limit() == 0.50


@pytest.mark.parametrize("conversion", [None, 0.0, -1.0, float("nan"), float("inf")])
def test_invalid_conversion_still_blocks_sizing(conversion):
    units, _ = _size(conversion=conversion)
    assert units == 0


class TradeAuditClient:
    """In-memory broker responses; no HTTP client or real order is created."""

    def __init__(self, units):
        entry = Decimal("0.70")
        distance = Decimal("0.00042")
        self.trade = {
            "id": "2",
            "state": "OPEN",
            "instrument": "AUD_USD",
            "price": str(entry),
            "currentUnits": str(units),
            "stopLossOrder": {
                "id": "3",
                "state": "PENDING",
                "type": "STOP_LOSS",
                "tradeID": "2",
                "price": str(entry - distance if units > 0 else entry + distance),
            },
        }
        self.closed = []

    def get(self, path):
        assert path == "/v3/accounts/test-only/trades/2"
        return SimpleNamespace(status_code=200, json=lambda: {"trade": self.trade})

    def put(self, path, json):
        assert path == "/v3/accounts/test-only/trades/2/close"
        assert json == {"units": "ALL"}
        self.closed.append("2")
        self.trade["state"] = "CLOSED"
        return SimpleNamespace(
            status_code=200,
            json=lambda: {
                "orderFillTransaction": {"tradesClosed": [{"tradeID": "2"}]}
            },
        )


@pytest.mark.parametrize(
    "cash_cap,expected_units", [(0.50, 811), (0.25, 405), (0.20, 324)]
)
@pytest.mark.parametrize("direction", [1, -1])
def test_conversion_headroom_survives_small_drift_but_preserves_hard_audit(
    monkeypatch, tmp_path, cash_cap, expected_units, direction
):
    monkeypatch.setenv("MAX_RISK_PER_TRADE_CCY", str(cash_cap))
    monkeypatch.setenv("MOSSY_STATE_PATH", str(tmp_path))
    monkeypatch.setattr(settings, "MODE", "demo")
    monkeypatch.setattr(settings, "OANDA_ENV", "practice")
    monkeypatch.setattr(settings, "OANDA_ACCOUNT_ID", "test-only")
    monkeypatch.setattr(settings, "OANDA_API_KEY", "")
    broker = Broker()
    conversion = {"rate": 1.4375}
    monkeypatch.setattr(broker, "conversion_rate", lambda *_: conversion["rate"])
    units, diagnostics = _size(broker=broker, stop_distance=0.00042)
    assert units == expected_units
    assert diagnostics["cash_risk_sizing_limit"] == pytest.approx(cash_cap * 0.98)
    assert diagnostics["planned_stop_risk"] <= cash_cap * 0.98

    client = TradeAuditClient(direction * units)
    summary = {"id": "2", "instrument": "AUD_USD", "stopLossOrderID": "3"}
    assert broker._audit_open_trade_protection(client, [summary]) == [summary]

    # A later refresh can use the reserved margin; the audit does not enforce
    # the reduced sizing budget against existing trades.
    conversion["rate"] = 1.4375 * 1.02
    assert diagnostics["planned_stop_risk"] * 1.02 > cash_cap * 0.98
    assert broker._audit_open_trade_protection(client, [summary]) == [summary]
    assert client.closed == []
    assert broker.entry_halt_reason is None
    assert configured_cash_risk_limit() == cash_cap

    # Headroom is finite. A genuine above-cap stop risk still closes and halts.
    conversion["rate"] = 1.4375 * 1.05
    assert broker._audit_open_trade_protection(client, [summary]) is None
    assert client.closed == ["2"]
    assert broker.entry_halt_reason == "unprotected-open-trade"
    assert (tmp_path / "broker_entry_halt.txt").read_text().strip() == (
        "unprotected-open-trade"
    )


@pytest.mark.parametrize("mode,environment", [("live", "practice"), ("demo", "live")])
def test_conversion_headroom_does_not_enable_live_broker(monkeypatch, mode, environment):
    monkeypatch.setattr(settings, "MODE", mode)
    monkeypatch.setattr(settings, "OANDA_ENV", environment)

    with pytest.raises(SystemExit):
        Broker()


def test_existing_trade_at_cash_cap_is_allowed_without_above_cap_tolerance(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("MOSSY_STATE_PATH", str(tmp_path))
    monkeypatch.setattr(settings, "MODE", "demo")
    monkeypatch.setattr(settings, "OANDA_ENV", "practice")
    monkeypatch.setattr(settings, "OANDA_ACCOUNT_ID", "test-only")
    monkeypatch.setattr(settings, "OANDA_API_KEY", "")
    broker = Broker()
    client = TradeAuditClient(500)
    client.trade["stopLossOrder"]["price"] = "0.699"
    summary = {"id": "2", "instrument": "AUD_USD", "stopLossOrderID": "3"}
    monkeypatch.setattr(broker, "conversion_rate", lambda *_: 1.0)

    # Existing A$0.50 protection stays valid even though new sizing uses A$0.49.
    assert broker._audit_open_trade_protection(client, [summary]) == [summary]
    assert client.closed == []

    monkeypatch.setattr(broker, "conversion_rate", lambda *_: 1.00000001)
    assert broker._audit_open_trade_protection(client, [summary]) is None
    assert client.closed == ["2"]
    assert broker.entry_halt_reason == "unprotected-open-trade"
