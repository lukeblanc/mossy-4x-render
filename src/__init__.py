from __future__ import annotations

import os


def _floor_int_env(name: str, minimum: int) -> None:
    """Keep a stricter value, but raise stale or invalid values to the minimum."""
    try:
        current = int(os.getenv(name, str(minimum)))
    except (TypeError, ValueError):
        current = minimum
    os.environ[name] = str(max(minimum, current))


def _cap_int_env(name: str, maximum: int) -> None:
    """Keep a stricter value, but lower stale or invalid values to the maximum."""
    try:
        current = int(os.getenv(name, str(maximum)))
    except (TypeError, ValueError):
        current = maximum
    os.environ[name] = str(max(0, min(maximum, current)))


def _floor_float_env(name: str, minimum: float, maximum: float = 1.0) -> None:
    """Clamp a floating-point safety threshold to a conservative range."""
    try:
        current = float(os.getenv(name, str(minimum)))
    except (TypeError, ValueError):
        current = minimum
    os.environ[name] = str(max(minimum, min(maximum, current)))


def apply_runtime_safety_floors() -> None:
    """Enforce non-negotiable demo learning and over-trading safeguards.

    Render can retain older dashboard environment values even after render.yaml
    changes. Applying these limits when the ``src`` package loads ensures stale
    service settings cannot silently weaken the safety gates.
    """

    _floor_int_env("ADAPTIVE_MIN_SAMPLE", 20)
    _floor_int_env("SHADOW_MIN_TRAIN", 50)
    _floor_int_env("SHADOW_MIN_VALIDATION", 30)
    _floor_float_env("SHADOW_MIN_COVERAGE", 0.50)

    _cap_int_env("MAX_CONCURRENT_POSITIONS", 2)
    _cap_int_env("MAX_TRADES_PER_DAY", 8)

    # Learning remains observe/recommend only. It must never alter strategy
    # settings automatically without a reviewed code change.
    os.environ["SHADOW_AUTO_APPLY"] = "false"


apply_runtime_safety_floors()


__all__ = ["apply_runtime_safety_floors"]
