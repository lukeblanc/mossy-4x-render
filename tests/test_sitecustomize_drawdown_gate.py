from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


def _run_gate(tmp_path: Path, halted: bool) -> tuple[str, bool]:
    state_file = tmp_path / "risk_state.json"
    state_file.write_text(json.dumps({"max_drawdown_halt": halted}), encoding="utf-8")
    marker = tmp_path / ".safe_demo_drawdown_recovery_20260805_applied"
    marker.write_text("existing\n", encoding="utf-8")

    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
            "MODE": "demo",
            "OANDA_ENV": "practice",
            "MOSSY_STATE_PATH": str(tmp_path),
        }
    )
    completed = subprocess.run(
        [sys.executable, "-c", "print('gate-complete')"],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout, marker.exists()


def test_active_halt_arms_existing_recovery(tmp_path: Path) -> None:
    output, marker_exists = _run_gate(tmp_path, halted=True)
    assert "active halt detected; reset armed" in output
    assert marker_exists is False


def test_no_active_halt_suppresses_recovery(tmp_path: Path) -> None:
    output, marker_exists = _run_gate(tmp_path, halted=False)
    assert "no active halt; reset suppressed" in output
    assert marker_exists is True
