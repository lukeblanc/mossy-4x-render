from __future__ import annotations

from src import position_sizer


class StubBroker:
    def __init__(self, rates):
        self._rates = rates

    def conversion_rate(self, from_ccy: str, to_ccy: str):
        return self._rates.get((from_ccy.upper(), to_ccy.upper()))


def test_units_for_risk_non_jpy_applies_default_cash_cap(monkeypatch):
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
    # is capped at 1.50 AUD. Pip value per unit is 0.00015 AUD, so:
    # units = 1.50 / (10 * 0.00015) = 1000.
    assert units == 1000
    assert diag["requested_risk_amount"] == 33.1
    assert diag["risk_amount"] == 1.5
    assert diag["max_risk_per_trade_ccy"] == 1.5
    assert round(diag["stop_pips"], 5) == 10.0


def test_units_for_risk_cash_cap_can_be_configured(monkeypatch):
    monkeypatch.setenv("MAX_RISK_PER_TRADE_CCY", "2.25")
    broker = StubBroker({("USD", "AUD"): 1.5})
    units, diag = position_sizer.units_for_risk(
        equity=5000.0,
        instrument="AUD_USD",
        stop_distance=0.0010,
        risk_pct=0.0025,
        broker=broker,
    )

    assert units == 1500
    assert diag["requested_risk_amount"] == 12.5
    assert diag["risk_amount"] == 2.25


def test_units_for_risk_jpy_pair_uses_0_01_pip_size(monkeypatch):
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
    assert diag["risk_amount"] == 1.5
    assert units > 0


def test_units_for_risk_returns_zero_without_conversion_rate(monkeypatch):
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
