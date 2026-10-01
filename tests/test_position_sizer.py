from __future__ import annotations

from decimal import Decimal

import pytest

from src import adaptive_policy, position_sizer


class StubBroker:
    def __init__(self, rates):
        self._rates = rates

    def conversion_rate(self, from_ccy: str, to_ccy: str):
        return self._rates.get((from_ccy.upper(), to_ccy.upper()))


def _neutral_learning(monkeypatch):
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


def test_units_for_risk_non_jpy_applies_default_cash_cap(monkeypatch):
    _neutral_learning(monkeypatch)
    monkeypatch.delenv("MAX_RISK_PER_TRADE_CCY", raising=False)
    broker = StubBroker({("USD", "AUD"): 1.5})
    units, diag = position_sizer.units_for_risk(
        equity=1324.0,
        instrument="EUR_USD",
        stop_distance=0.0010,  # 10 pips
        risk_pct=0.025,
        broker=broker,
    )

    # Requested percentage risk is 33.10 AUD, but the broker-side stop exposure
    # is sized below 0.49 AUD, reserving 2% of the 0.50 AUD audit cap for
    # conversion drift. Pip value per unit is 0.00015 AUD; units round down.
    assert units == 326
    assert diag["requested_risk_amount"] == 33.1
    assert diag["risk_amount"] == 0.49
    assert diag["max_risk_per_trade_ccy"] == 0.5
    assert diag["cash_risk_sizing_limit"] == 0.49
    assert diag["conversion_risk_reserve_pct"] == 0.02
    assert diag["planned_stop_risk"] <= 0.49
    assert round(diag["stop_pips"], 5) == 10.0


def test_units_for_risk_dashboard_value_cannot_loosen_code_cap(monkeypatch):
    _neutral_learning(monkeypatch)
    monkeypatch.setenv("MAX_RISK_PER_TRADE_CCY", "2.25")
    broker = StubBroker({("USD", "AUD"): 1.5})
    units, diag = position_sizer.units_for_risk(
        equity=5000.0,
        instrument="AUD_USD",
        stop_distance=0.0010,
        risk_pct=0.0025,
        broker=broker,
    )

    assert units == 326
    assert diag["requested_risk_amount"] == 12.5
    assert diag["risk_amount"] == 0.49
    assert diag["max_risk_per_trade_ccy"] == 0.5


def test_units_for_risk_jpy_pair_uses_0_01_pip_size(monkeypatch):
    _neutral_learning(monkeypatch)
    monkeypatch.delenv("MAX_RISK_PER_TRADE_CCY", raising=False)
    broker = StubBroker({("JPY", "AUD"): 0.01})
    units, diag = position_sizer.units_for_risk(
        equity=1324.0,
        instrument="USD_JPY",
        stop_distance=0.10,  # 10 pips for JPY pair
        risk_pct=0.025,
        broker=broker,
    )

    assert round(diag["stop_pips"], 5) == 10.0
    assert diag["risk_amount"] == 0.49
    assert diag["conversion_risk_reserve_pct"] == 0.02
    assert units > 0


def test_units_for_risk_returns_zero_without_conversion_rate(monkeypatch):
    _neutral_learning(monkeypatch)
    monkeypatch.delenv("MAX_RISK_PER_TRADE_CCY", raising=False)
    broker = StubBroker({})
    units, diag = position_sizer.units_for_risk(
        equity=1324.0,
        instrument="EUR_USD",
        stop_distance=0.0010,
        risk_pct=0.025,
        broker=broker,
    )

    assert units == 0
    assert diag == {}


@pytest.mark.parametrize("configured", ["", "0", "-0.1", "nan", "inf", "bad"])
def test_invalid_explicit_cash_cap_blocks_position(monkeypatch, configured):
    _neutral_learning(monkeypatch)
    monkeypatch.setenv("MAX_RISK_PER_TRADE_CCY", configured)
    broker = StubBroker({("USD", "AUD"): 1.5})

    units, diag = position_sizer.units_for_risk(
        equity=5000.0,
        instrument="AUD_USD",
        stop_distance=0.001,
        risk_pct=0.0025,
        broker=broker,
    )

    assert units == 0
    assert diag["reason"] == "invalid-cash-risk-cap"


def test_stricter_cash_cap_is_preserved(monkeypatch):
    _neutral_learning(monkeypatch)
    monkeypatch.setenv("MAX_RISK_PER_TRADE_CCY", "0.25")
    broker = StubBroker({("USD", "AUD"): 1.5})

    units, diag = position_sizer.units_for_risk(
        equity=5000.0,
        instrument="AUD_USD",
        stop_distance=0.001,
        risk_pct=0.0025,
        broker=broker,
    )

    assert units == 163
    assert diag["risk_amount"] == 0.245
    assert diag["max_risk_per_trade_ccy"] == 0.25
    assert diag["planned_stop_risk"] <= 0.245


@pytest.mark.parametrize("stop_distance", [0.00001, 0.00005, 0.001, 0.01, 0.1])
@pytest.mark.parametrize("conversion_rate", [0.01, 0.5, 1.0, 1.51, 100.0])
def test_planned_stop_risk_never_exceeds_cash_cap(
    monkeypatch, stop_distance, conversion_rate
):
    _neutral_learning(monkeypatch)
    monkeypatch.setenv("MAX_RISK_PER_TRADE_CCY", "0.50")
    broker = StubBroker({("USD", "AUD"): conversion_rate})

    units, diag = position_sizer.units_for_risk(
        equity=1_000_000.0,
        instrument="EUR_USD",
        stop_distance=stop_distance,
        risk_pct=0.01,
        broker=broker,
    )

    if units > 0:
        assert diag["planned_stop_risk"] <= 0.49
        # Check the broker's decimal exposure, avoiding float associativity at
        # an exact sizing-budget boundary (e.g. 490 * 0.00001 * 100).
        broker_risk = (
            Decimal(units) * Decimal(str(stop_distance)) * Decimal(str(conversion_rate))
        )
        assert broker_risk <= Decimal("0.49")
        assert broker_risk * Decimal("1.02") <= Decimal("0.50")
    else:
        assert diag.get("reason") == "minimum-units-exceed-risk-budget"
