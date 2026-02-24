from __future__ import annotations

import math
from typing import Optional, Tuple


ACCOUNT_CURRENCY = "AUD"


def _pip_size(instrument: str) -> float:
    if instrument.endswith("JPY"):
        return 0.01
    return 0.0001


def _instrument_currencies(instrument: str) -> Tuple[str, str]:
    base, quote = instrument.split("_", 1)
    return base.upper(), quote.upper()


def _pip_value_per_unit_in_account_ccy(
    instrument: str,
    *,
    broker,
    account_currency: str = ACCOUNT_CURRENCY,
) -> Optional[float]:
    _, quote_ccy = _instrument_currencies(instrument)
    pip_size = _pip_size(instrument)
    conversion_rate = broker.conversion_rate(quote_ccy, account_currency)
    if conversion_rate is None or conversion_rate <= 0:
        return None
    return pip_size * conversion_rate


def units_for_risk(
    equity: float,
    instrument: str,
    stop_distance: float,
    risk_pct: float,
    *,
    broker,
    account_currency: str = ACCOUNT_CURRENCY,
    min_trade_units: int = 1,
) -> tuple[int, dict]:
    """Return units sized to risk_pct of equity using stop distance and pip conversion."""

    if equity <= 0 or stop_distance <= 0 or risk_pct <= 0:
        return 0, {}

    pip_size = _pip_size(instrument)
    if pip_size <= 0:
        return 0, {}

    stop_pips = stop_distance / pip_size
    if stop_pips <= 0:
        return 0, {}

    risk_amount = equity * risk_pct
    pip_value_per_unit = _pip_value_per_unit_in_account_ccy(
        instrument,
        broker=broker,
        account_currency=account_currency,
    )
    if pip_value_per_unit is None or pip_value_per_unit <= 0:
        return 0, {}

    raw_units = risk_amount / (stop_pips * pip_value_per_unit)
    if not math.isfinite(raw_units) or raw_units <= 0:
        return 0, {}

    final_units = max(int(min_trade_units), int(raw_units))
    diagnostics = {
        "equity": equity,
        "risk_pct": risk_pct,
        "risk_amount": risk_amount,
        "stop_pips": stop_pips,
        "pip_value_per_unit": pip_value_per_unit,
        "final_units": final_units,
    }
    return final_units, diagnostics
