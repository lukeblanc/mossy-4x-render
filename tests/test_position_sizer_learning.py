from __future__ import annotations

import pytest

from src import adaptive_policy, position_sizer


class ConversionBroker:
    @staticmethod
    def conversion_rate(from_ccy: str, to_ccy: str):
        return 1.0


def test_learning_can_only_reduce_requested_risk(monkeypatch):
    monkeypatch.setattr(
        adaptive_policy,
        "evaluate_instrument_policy",
        lambda instrument: adaptive_policy.PolicyDecision(
            instrument=instrument,
            setup_key="AUD_USD|BUY|LONDON|RSI_GE55|TREND_FULL",
            risk_scale=2.0,
            blocked=False,
            reason="bad-upscale-attempt",
        ),
    )

    units, diagnostics = position_sizer.units_for_risk(
        1000.0,
        "AUD_USD",
        0.001,
        0.005,
        broker=ConversionBroker(),
    )

    assert units == 5000
    assert diagnostics["requested_risk_pct"] == pytest.approx(0.005)
    assert diagnostics["risk_pct"] == pytest.approx(0.005)
    assert diagnostics["learning_scale"] == pytest.approx(1.0)


def test_learning_block_returns_zero_units(monkeypatch):
    monkeypatch.setattr(
        adaptive_policy,
        "evaluate_instrument_policy",
        lambda instrument: adaptive_policy.PolicyDecision(
            instrument=instrument,
            setup_key="AUD_USD|BUY|LONDON|RSI_GE55|TREND_FULL",
            risk_scale=0.0,
            blocked=True,
            reason="setup-four-loss-streak",
        ),
    )

    units, diagnostics = position_sizer.units_for_risk(
        1000.0,
        "AUD_USD",
        0.001,
        0.005,
        broker=ConversionBroker(),
    )

    assert units == 0
    assert diagnostics["learning_blocked"] is True
    assert diagnostics["risk_amount"] == pytest.approx(0.0)
