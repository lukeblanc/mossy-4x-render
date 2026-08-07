"""Gate the legacy one-time demo drawdown recovery on an actual active halt.

Python imports ``sitecustomize`` before the application starts.  The merged
PR #155 recovery marker is therefore prepared before ``app.config`` applies
its safe-demo profile:

* active persisted max-drawdown halt -> remove marker, allowing one reset;
* no active halt -> ensure marker exists, preventing an unnecessary re-anchor.

This module is deliberately inactive outside demo/practice mode.
"""
from __future__ import annotations

import json
import os
from pathlib import Path


def _enabled_demo_practice() -> bool:
    mode = (os.getenv("MODE") or "demo").strip().lower()
    oanda_env = (os.getenv("OANDA_ENV") or "practice").strip().lower()
    return mode in {"demo", "paper", "practice"} and oanda_env == "practice"


def _state_root() -> Path:
    configured = os.getenv("MOSSY_STATE_PATH")
    return Path(configured) if configured else Path("data")


def _persisted_halt_active(state_file: Path) -> bool:
    try:
        payload = json.loads(state_file.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return False
    return bool(payload.get("max_drawdown_halt", False))


def _gate_demo_drawdown_recovery() -> None:
    if not _enabled_demo_practice():
        return

    root = _state_root()
    marker = root / ".safe_demo_drawdown_recovery_20260805_applied"
    state_file = root / "risk_state.json"

    try:
        root.mkdir(parents=True, exist_ok=True)
        if _persisted_halt_active(state_file):
            # Allow app.config to request the existing one-time reset.
            marker.unlink(missing_ok=True)
            print("[DRAWdown-RECOVERY-GATE] active halt detected; reset armed", flush=True)
        else:
            # Prevent the legacy migration from moving a healthy peak baseline.
            marker.touch(exist_ok=True)
            print("[DRAWDOWN-RECOVERY-GATE] no active halt; reset suppressed", flush=True)
    except OSError as exc:
        print(f"[DRAWDOWN-RECOVERY-GATE][WARN] {exc}", flush=True)


_gate_demo_drawdown_recovery()
