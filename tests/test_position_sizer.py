from __future__ import annotations

from src import position_sizer


class StubBroker:
    def __init__(self, rates):
        self._rates = rates

    def conversion_rate(self, from_ccy: str, to_ccy: str):
        return self._rates.get((from_ccy.upper(), to_ccy.upper()))


def test_units_for_risk_non_jpy_with_quote_to_aud_conversion():
    broker = StubBroker({("USD", "AUD"): 1.5})
    units, diag = position_sizer.units_for_risk(
        equity=1324.0,
        instrument="EUR_USD",
        stop_distance=0.0010,  # 10 pips
        risk_pct=0.025,
        broker=broker,
    )

    # risk_amount = 1324 * 0.025 = 33.1 AUD
    # pip value per unit = 0.0001 * 1.5 = 0.00015 AUD
    # units = 33.1 / (10 * 0.00015) = 22066.66 -> int(22066)
    assert units == 22066
    assert diag["risk_amount"] == 33.1
    assert round(diag["stop_pips"], 5) == 10.0


def test_units_for_risk_jpy_pair_uses_0_01_pip_size():
    broker = StubBroker({("JPY", "AUD"): 0.01})
    units, diag = position_sizer.units_for_risk(
        equity=1324.0,
        instrument="USD_JPY",
        stop_distance=0.10,  # 10 pips for JPY pair
        risk_pct=0.025,
        broker=broker,
    )

    assert round(diag["stop_pips"], 5) == 10.0
    assert units > 0


def test_units_for_risk_returns_zero_without_conversion_rate():
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
