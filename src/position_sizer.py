from __future__ import annotations

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

