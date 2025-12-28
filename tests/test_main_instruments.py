from __future__ import annotations

from typing import Dict

import pytest

import src.main as main_mod
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
