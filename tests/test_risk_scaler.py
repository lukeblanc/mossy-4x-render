from __future__ import annotations

import pytest

from src.risk_scaler import RiskScaler


@pytest.fixture()
def base_config() -> dict:
    return {"risk": {"risk_per_trade_pct": 0.0025}}


def test_default_tiers_when_enabled(base_config):
    config = {**base_config, "risk_scaler": {"enabled": True}}
    scaler = RiskScaler(config, default_risk_pct=0.0025)

    risk = scaler.get_risk_pct(1500)
    assert pytest.approx(risk, rel=1e-5) == 0.0025
    assert scaler.last_tier and scaler.last_tier.name == "Tier1"

    risk = scaler.get_risk_pct(4500)
    assert pytest.approx(risk, rel=1e-5) == 0.0075
    assert scaler.last_tier and scaler.last_tier.name == "Tier3"


def test_custom_tiers_respected():
    config = {
        "risk_scaler": {
            "enabled": True,
            "tiers": [
                {"min": 0, "max": 1000, "risk_pct": 0.25},
                {"min": 1000, "max": 2000, "risk_pct": 0.5},
                {"min": 2000, "max": None, "risk_pct": 1.25},
            ],
        }
    }
    scaler = RiskScaler(config, default_risk_pct=0.001)

    assert pytest.approx(scaler.get_risk_pct(500), rel=1e-5) == 0.0025
    assert scaler.last_tier and scaler.last_tier.range_label == "0-1k"

    assert pytest.approx(scaler.get_risk_pct(10_000), rel=1e-5) == 0.0125
    assert scaler.last_tier and scaler.last_tier.range_label == "2k+"


def test_disabled_falls_back_to_default(base_config):
    config = {**base_config, "risk_scaler": {"enabled": False}}
    scaler = RiskScaler(config, default_risk_pct=0.001)

    assert pytest.approx(scaler.get_risk_pct(5000), rel=1e-5) == 0.001
    assert scaler.last_tier is None

