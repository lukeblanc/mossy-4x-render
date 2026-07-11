from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


def _run_config_import(tmp_path: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
            "RENDER_GIT_COMMIT": "test-commit",
            "MOSSY_STATE_PATH": str(tmp_path),
            "MODE": "demo",
            "OANDA_ENV": "practice",
            "AGGRESSIVE_MODE": "true",
            "AGGRESSIVE_TEST_MODE": "true",
            "AGGRESSIVE_TEST_RISK_PCT": "2.5",
            "MAX_RISK_PER_TRADE_CAP_PCT": "1.0",
            "INSTRUMENTS": "EUR_USD,GBP_USD,AUD_USD,USD_JPY,XAU_USD",
            "MERGE_DEFAULT_INSTRUMENTS": "true",
            "SESSION_MODE": "ALWAYS",
        }
    )
    code = """
import json
import os
import app.config
keys = [
    'MODE', 'OANDA_ENV', 'AGGRESSIVE_MODE', 'AGGRESSIVE_TEST_MODE',
    'MAX_RISK_PER_TRADE_CAP_PCT', 'INSTRUMENTS',
    'MERGE_DEFAULT_INSTRUMENTS', 'SESSION_MODE',
    'RESET_MAX_DRAWDOWN_HALT'
]
print(json.dumps({key: os.environ.get(key, '') for key in keys}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout.strip().splitlines()[-1])


def test_render_safe_demo_profile_overrides_stale_dashboard_values(tmp_path: Path) -> None:
    first = _run_config_import(tmp_path)
    assert first == {
        "MODE": "demo",
        "OANDA_ENV": "practice",
        "AGGRESSIVE_MODE": "false",
        "AGGRESSIVE_TEST_MODE": "false",
        "MAX_RISK_PER_TRADE_CAP_PCT": "0.5",
        "INSTRUMENTS": "AUD_USD,GBP_USD",
        "MERGE_DEFAULT_INSTRUMENTS": "false",
        "SESSION_MODE": "SOFT",
        "RESET_MAX_DRAWDOWN_HALT": "true",
    }

    second = _run_config_import(tmp_path)
    assert second["RESET_MAX_DRAWDOWN_HALT"] == "false"
    assert (tmp_path / ".safe_demo_profile_20260711_applied").exists()
