from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


def _run_config_import(
    tmp_path: Path,
    *,
    first_import: str = "app.config",
    overrides: dict[str, str] | None = None,
    extra_keys: tuple[str, ...] = (),
) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": os.pathsep.join(filter(None, [
                str(Path(__file__).resolve().parents[1]), env.get("PYTHONPATH")
            ])),
            "RENDER_GIT_COMMIT": "test-commit",
            "MOSSY_SAFE_DEMO_PROFILE": "true",
            "MOSSY_STATE_PATH": str(tmp_path),
            "MODE": "demo",
            "OANDA_ENV": "practice",
            "AGGRESSIVE_MODE": "true",
            "AGGRESSIVE_TEST_MODE": "true",
            "AGGRESSIVE_TEST_RISK_PCT": "2.5",
            "MAX_RISK_PER_TRADE_CAP_PCT": "1.0",
            "MAX_RISK_PER_TRADE_CCY": "1.80",
            "HARD_MAX_LOSS_CCY": "1.50",
            "MAX_CONCURRENT_POSITIONS": "2",
            "INSTRUMENTS": "EUR_USD,GBP_USD,AUD_USD,USD_JPY,XAU_USD",
            "MERGE_DEFAULT_INSTRUMENTS": "true",
            "SESSION_MODE": "ALWAYS",
        }
    )
    env.update(overrides or {})
    code = f"import {first_import}\nextra_keys = {list(extra_keys)!r}\n" + """
import json
import os
import app.config
keys = [
    'MODE', 'OANDA_ENV', 'AGGRESSIVE_MODE', 'AGGRESSIVE_TEST_MODE',
    'MAX_RISK_PER_TRADE_CAP_PCT', 'MAX_RISK_PER_TRADE_CCY',
    'HARD_MAX_LOSS_CCY', 'MAX_CONCURRENT_POSITIONS', 'INSTRUMENTS',
    'MERGE_DEFAULT_INSTRUMENTS', 'SESSION_MODE',
    'RESET_MAX_DRAWDOWN_HALT'
] + extra_keys
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
        "MAX_RISK_PER_TRADE_CCY": "0.50",
        "HARD_MAX_LOSS_CCY": "0.50",
        "MAX_CONCURRENT_POSITIONS": "1",
        "INSTRUMENTS": "AUD_USD,GBP_USD",
        "MERGE_DEFAULT_INSTRUMENTS": "false",
        "SESSION_MODE": "SOFT",
        "RESET_MAX_DRAWDOWN_HALT": "false",
    }

    second = _run_config_import(tmp_path)
    assert second["RESET_MAX_DRAWDOWN_HALT"] == "false"
    assert not (tmp_path / ".safe_demo_drawdown_recovery_20260805_applied").exists()


@pytest.mark.parametrize("first_import", ["src", "app.config"])
def test_approved_twenty_entry_twenty_cent_profile_preserves_other_guards(
    tmp_path: Path, first_import: str
) -> None:
    actual = _run_config_import(
        tmp_path,
        first_import=first_import,
        overrides={
            "MAX_TRADES_PER_DAY": "20",
            "MAX_RISK_PER_TRADE_CCY": "0.20",
            "HARD_MAX_LOSS_CCY": "0.20",
            "MAX_RISK_PER_TRADE": "0.0025",
            "DAILY_LOSS_CAP_PCT": "0.01",
            "WEEKLY_LOSS_CAP_PCT": "0.03",
            "MAX_DRAWDOWN_CAP_PCT": "0.05",
            "SHADOW_AUTO_APPLY": "true",
            "RESET_WEEKLY_LOSS_CAP": "true",
            "RESET_MAX_DRAWDOWN_HALT": "true",
        },
        extra_keys=(
            "MAX_TRADES_PER_DAY", "MAX_OPEN_TRADES", "MAX_RISK_PER_TRADE",
            "DAILY_LOSS_CAP_PCT", "WEEKLY_LOSS_CAP_PCT", "MAX_DRAWDOWN_CAP_PCT",
            "COOLDOWN_CANDLES", "TP_ENABLED", "SHADOW_AUTO_APPLY",
            "RESET_WEEKLY_LOSS_CAP",
        ),
    )

    assert actual["MAX_TRADES_PER_DAY"] == "20"
    assert float(actual["MAX_RISK_PER_TRADE_CCY"]) == 0.20
    assert float(actual["HARD_MAX_LOSS_CCY"]) == 0.20
    for key, value in {
        "MODE": "demo", "OANDA_ENV": "practice",
        "MAX_CONCURRENT_POSITIONS": "1", "MAX_OPEN_TRADES": "1",
        "AGGRESSIVE_MODE": "false", "AGGRESSIVE_TEST_MODE": "false",
        "MAX_RISK_PER_TRADE_CAP_PCT": "0.5", "MAX_RISK_PER_TRADE": "0.0025",
        "DAILY_LOSS_CAP_PCT": "0.01", "WEEKLY_LOSS_CAP_PCT": "0.03",
        "MAX_DRAWDOWN_CAP_PCT": "0.05", "COOLDOWN_CANDLES": "9",
        "SESSION_MODE": "SOFT", "INSTRUMENTS": "AUD_USD,GBP_USD",
        "MERGE_DEFAULT_INSTRUMENTS": "false", "TP_ENABLED": "true",
        "SHADOW_AUTO_APPLY": "false", "RESET_WEEKLY_LOSS_CAP": "false",
        "RESET_MAX_DRAWDOWN_HALT": "false",
    }.items():
        assert actual[key] == value, key
