from __future__ import annotations

from typing import Any, Mapping, Optional, Tuple

from src.risk_scaler import RiskScaler


def resolve_risk_pct(
    config: Optional[Mapping[str, Any]],
    equity: float,
    *,
    fallback: Optional[float] = None,
) -> Tuple[float, RiskScaler]:
    scaler = RiskScaler(config, default_risk_pct=fallback)
    return scaler.get_risk_pct(equity), scaler


def units_for_risk(
    equity: float,
    entry_price: float,
    stop_distance: float,
    risk_pct: float,
) -> int:
    if equity <= 0:
        return 0
    if stop_distance <= 0:
        return 0
    if risk_pct <= 0:
        return 0
    if entry_price <= 0:
        return 0

    risk_amount = equity * risk_pct
    if risk_amount <= 0:
        return 0

    units = risk_amount / stop_distance
    return max(1, int(units))

