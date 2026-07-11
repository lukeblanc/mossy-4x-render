from __future__ import annotations

import math
from typing import Optional, Tuple

from src import adaptive_policy


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
    """Return units sized to risk_pct with journal-backed setup learning.

    The existing account and strategy risk limits remain authoritative. The
    adaptive policy may only reduce or block risk; it can never increase the
    requested percentage above the caller's value.
    """

    if equity <= 0 or stop_distance <= 0 or risk_pct <= 0:
        return 0, {}

    try:
        policy = adaptive_policy.evaluate_instrument_policy(instrument)
    except Exception as exc:
        print(f"[LEARNING][WARN] policy lookup failed instrument={instrument} error={exc}", flush=True)
        policy = adaptive_policy.PolicyDecision(
            instrument=instrument,
            setup_key="error",
            risk_scale=1.0,
            blocked=False,
            reason="policy-error",
        )

    learning_scale = max(0.0, min(1.0, float(policy.risk_scale)))
    effective_risk_pct = min(float(risk_pct), float(risk_pct) * learning_scale)
    if policy.blocked or effective_risk_pct <= 0:
        diagnostics = {
            "equity": equity,
            "risk_pct": 0.0,
            "risk_amount": 0.0,
            "stop_pips": 0.0,
            "pip_value_per_unit": 0.0,
            "final_units": 0,
            "learning_scale": learning_scale,
            "learning_reason": policy.reason,
            "learning_setup_key": policy.setup_key,
            "learning_blocked": True,
        }
        print(
            f"[LEARNING][BLOCK] instrument={instrument} setup={policy.setup_key} reason={policy.reason}",
            flush=True,
        )
        return 0, diagnostics

    pip_size = _pip_size(instrument)
    if pip_size <= 0:
        return 0, {}

    stop_pips = stop_distance / pip_size
    if stop_pips <= 0:
        return 0, {}

    risk_amount = equity * effective_risk_pct
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
        "risk_pct": effective_risk_pct,
        "requested_risk_pct": risk_pct,
        "risk_amount": risk_amount,
        "stop_pips": stop_pips,
        "pip_value_per_unit": pip_value_per_unit,
        "final_units": final_units,
        "learning_scale": learning_scale,
        "learning_reason": policy.reason,
        "learning_setup_key": policy.setup_key,
        "learning_blocked": False,
        "learning_exact_samples": policy.exact_samples,
        "learning_pair_side_samples": policy.pair_side_samples,
    }
    return final_units, diagnostics
