from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

from src.profit_protection import ProfitProtection
from src.risk_manager import RiskManager


DEFAULT_TRAILING_CONFIG = {
    "arm_pips": 0.0,
    "giveback_pips": 0.0,
    "arm_ccy": 1.0,
    "giveback_ccy": 0.5,
    "use_pips": False,
    "be_arm_pips": 0.0,
    "be_offset_pips": 0.0,
    "min_check_interval_sec": 0.0,
}

DEFAULT_TIME_STOP = {
    "minutes": 90.0,
    "min_pips": 2.0,
    "xau_atr_mult": 0.35,
}


def resolve_state_dir(fallback: Optional[Path] = None) -> Path:
    """Return the state directory honoring MOSSY_STATE_PATH when set."""

    root = os.getenv("MOSSY_STATE_PATH")
    base = Path(root) if root else (fallback or Path("data"))
    base.mkdir(parents=True, exist_ok=True)
    return base


def build_risk_manager(
    config: Dict,
    *,
    mode: str,
    demo_mode: bool = False,
    state_dir: Optional[Path] = None,
) -> RiskManager:
    """Instantiate a RiskManager with consistent state handling."""

    risk_config = config.get("risk") if "risk" in config else config
    state_path = resolve_state_dir(state_dir)
    return RiskManager(
        risk_config or {},
        mode=mode,
        state_dir=state_path,
        demo_mode=demo_mode,
    )


def build_profit_protection(
    mode: str,
    broker,
    aggressive: bool = False,
    *,
    trailing: Optional[Dict] = None,
    time_stop: Optional[Dict] = None,
) -> ProfitProtection:
    """Create ProfitProtection consistent with deployment mode."""

    trailing_cfg = DEFAULT_TRAILING_CONFIG | (trailing or {})
    ts_cfg = DEFAULT_TIME_STOP | (time_stop or {})
    ts_minutes = float(ts_cfg.get("minutes", DEFAULT_TIME_STOP["minutes"]))
    ts_min_pips = float(ts_cfg.get("min_pips", DEFAULT_TIME_STOP["min_pips"]))
    ts_xau_mult = float(ts_cfg.get("xau_atr_mult", DEFAULT_TIME_STOP["xau_atr_mult"]))
    arm_ccy = trailing_cfg.get("arm_ccy", trailing_cfg.get("arm_usd"))
    giveback_ccy = trailing_cfg.get("giveback_ccy", trailing_cfg.get("giveback_usd"))

    return ProfitProtection(
        broker,
        trigger=float(arm_ccy if arm_ccy is not None else DEFAULT_TRAILING_CONFIG["arm_ccy"]),
        trail=float(giveback_ccy if giveback_ccy is not None else DEFAULT_TRAILING_CONFIG["giveback_ccy"]),
        arm_ccy=arm_ccy,
        giveback_ccy=giveback_ccy,
        arm_pips=trailing_cfg["arm_pips"],
        giveback_pips=trailing_cfg["giveback_pips"],
        use_pips=False,
        be_arm_pips=trailing_cfg["be_arm_pips"],
        be_offset_pips=trailing_cfg["be_offset_pips"],
        min_check_interval_sec=trailing_cfg["min_check_interval_sec"],
        aggressive=aggressive,
        aggressive_max_hold_minutes=float(trailing_cfg.get("aggressive_max_hold_minutes", 45.0)),
        aggressive_max_loss_ccy=float(
            trailing_cfg.get("aggressive_max_loss_ccy", trailing_cfg.get("aggressive_max_loss_usd", 5.0))
        ),
        aggressive_max_loss_atr_mult=float(trailing_cfg.get("aggressive_max_loss_atr_mult", 1.2)),
        time_stop_minutes=ts_minutes,
        time_stop_min_pips=ts_min_pips,
        time_stop_xau_atr_mult=ts_xau_mult,
    )
