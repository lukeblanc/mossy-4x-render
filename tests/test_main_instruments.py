from __future__ import annotations

import os
from typing import Dict

import pytest

import src.main as main_mod
from src.decision_engine import DecisionEngine
from src.decision_engine import DEFAULT_INSTRUMENTS


def test_resolve_instruments_defaults_when_missing():
    config: Dict = {}
    resolved = main_mod._resolve_instruments_config(config)

    assert resolved == DEFAULT_INSTRUMENTS


def test_resolve_instruments_respects_env_override(monkeypatch):
    monkeypatch.setenv("INSTRUMENTS", "eur_usd;gbp_usd")
    config: Dict = {"instruments": ["AUD_USD", "USD_JPY"]}

    resolved = main_mod._resolve_instruments_config(config)

    assert resolved == ["EUR_USD", "GBP_USD"]


def test_resolve_instruments_allows_empty_env(monkeypatch):
    monkeypatch.setenv("INSTRUMENTS", "   ")
    config: Dict = {"instruments": ["AUD_USD"], "merge_default_instruments": False}

    resolved = main_mod._resolve_instruments_config(config)

    assert resolved == []


def test_engine_does_not_merge_defaults_when_disabled(monkeypatch):
    monkeypatch.setenv("INSTRUMENTS", "AUD_USD,GBP_USD")
    monkeypatch.setenv("MERGE_DEFAULT_INSTRUMENTS", "false")

    config: Dict = {
        "instruments": main_mod._resolve_instruments_config({}),
        "merge_default_instruments": main_mod._as_bool(os.getenv("MERGE_DEFAULT_INSTRUMENTS", "false")),
    }

    engine = DecisionEngine(config)

    assert engine.instruments == ["AUD_USD", "GBP_USD"]


def test_env_instruments_disable_default_merge_when_merge_env_missing(monkeypatch):
    monkeypatch.setenv("INSTRUMENTS", "AUD_USD,GBP_USD")
    monkeypatch.delenv("MERGE_DEFAULT_INSTRUMENTS", raising=False)

    config: Dict = {"merge_default_instruments": True}

    merge_default = main_mod._resolve_merge_default_instruments(config)
    resolved = main_mod._resolve_instruments_config(config)
    engine = DecisionEngine(
        {
            "instruments": resolved,
            "merge_default_instruments": merge_default,
        }
    )

    assert merge_default is False
    assert engine.instruments == ["AUD_USD", "GBP_USD"]


def test_engine_env_instruments_override_config_defaults(monkeypatch):
    monkeypatch.setenv("INSTRUMENTS", "AUD_USD,GBP_USD")
    monkeypatch.delenv("MERGE_DEFAULT_INSTRUMENTS", raising=False)

    engine = DecisionEngine(
        {
            "instruments": DEFAULT_INSTRUMENTS,
            "merge_default_instruments": True,
        }
    )

    assert engine.instruments == ["AUD_USD", "GBP_USD"]
