from __future__ import annotations

import math
import os


HARD_MAX_RISK_PER_TRADE_CCY = 0.50
# New cross-currency positions leave room for account-currency conversion drift.
# This only reduces sizing; the broker audit continues to enforce the full cap.
CROSS_CURRENCY_SIZING_RESERVE_PCT = 0.02


def configured_cash_risk_limit() -> float:
    """Return a positive configured limit capped by the audited hard ceiling.

    A missing value uses the safe code default. An explicitly malformed,
    non-finite, zero, or negative value returns zero so callers fail closed.
    """

    raw = os.getenv(
        "MAX_RISK_PER_TRADE_CCY", str(HARD_MAX_RISK_PER_TRADE_CCY)
    )
    try:
        configured = float(raw)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    if not math.isfinite(configured) or configured <= 0:
        return 0.0
    return min(configured, HARD_MAX_RISK_PER_TRADE_CCY)


__all__ = [
    "HARD_MAX_RISK_PER_TRADE_CCY",
    "CROSS_CURRENCY_SIZING_RESERVE_PCT",
    "configured_cash_risk_limit",
]
