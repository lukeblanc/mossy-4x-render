from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


def test_live_mode_does_not_touch_recovery_marker(tmp_path: Path) -> None:
    state_file = tmp_path / "risk_state.json"
    state_file.write_text(json.dumps({"max_drawdown_halt": True}), encoding="utf-8")
    marker = tmp_path / ".safe_demo_drawdown_recovery_20260805_applied"
    marker.write_text("existing\n", encoding="utf-8")

    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
            "MODE": "live",
            "OANDA_ENV": "live",
            "MOSSY_STATE_PATH": str(tmp_path),
        }
    )
    subprocess.run(
        [sys.executable, "-c", "print('gate-complete')"],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert marker.exists()
    assert marker.read_text(encoding="utf-8") == "existing\n"
