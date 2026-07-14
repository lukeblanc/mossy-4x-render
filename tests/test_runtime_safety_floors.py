from __future__ import annotations

import os

from src import apply_runtime_safety_floors


def test_runtime_safety_floors_override_stale_render_values(monkeypatch) -> None:
    monkeypatch.setenv("ADAPTIVE_MIN_SAMPLE", "8")
    monkeypatch.setenv("SHADOW_MIN_TRAIN", "20")
    monkeypatch.setenv("SHADOW_MIN_VALIDATION", "10")
    monkeypatch.setenv("SHADOW_MIN_COVERAGE", "0.35")
    monkeypatch.setenv("MAX_CONCURRENT_POSITIONS", "3")
    monkeypatch.setenv("MAX_TRADES_PER_DAY", "100")
    monkeypatch.setenv("SHADOW_AUTO_APPLY", "true")

    apply_runtime_safety_floors()

    assert os.environ["ADAPTIVE_MIN_SAMPLE"] == "20"
    assert os.environ["SHADOW_MIN_TRAIN"] == "50"
    assert os.environ["SHADOW_MIN_VALIDATION"] == "30"
    assert float(os.environ["SHADOW_MIN_COVERAGE"]) == 0.50
    assert os.environ["MAX_CONCURRENT_POSITIONS"] == "2"
    assert os.environ["MAX_TRADES_PER_DAY"] == "8"
    assert os.environ["SHADOW_AUTO_APPLY"] == "false"


def test_runtime_safety_floors_preserve_stricter_values(monkeypatch) -> None:
    monkeypatch.setenv("ADAPTIVE_MIN_SAMPLE", "40")
    monkeypatch.setenv("SHADOW_MIN_TRAIN", "80")
    monkeypatch.setenv("SHADOW_MIN_VALIDATION", "45")
    monkeypatch.setenv("SHADOW_MIN_COVERAGE", "0.75")
    monkeypatch.setenv("MAX_CONCURRENT_POSITIONS", "1")
    monkeypatch.setenv("MAX_TRADES_PER_DAY", "4")

    apply_runtime_safety_floors()

    assert os.environ["ADAPTIVE_MIN_SAMPLE"] == "40"
    assert os.environ["SHADOW_MIN_TRAIN"] == "80"
    assert os.environ["SHADOW_MIN_VALIDATION"] == "45"
    assert float(os.environ["SHADOW_MIN_COVERAGE"]) == 0.75
    assert os.environ["MAX_CONCURRENT_POSITIONS"] == "1"
    assert os.environ["MAX_TRADES_PER_DAY"] == "4"
