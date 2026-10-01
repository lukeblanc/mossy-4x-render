from __future__ import annotations

import math
from typing import Optional, Tuple

from src import adaptive_policy
from src.cash_risk import (
    CROSS_CURRENCY_SIZING_RESERVE_PCT,
    HARD_MAX_RISK_PER_TRADE_CCY,
    configured_cash_risk_limit,
)


ACCOUNT_CURRENCY = "AUD"
DEFAULT_MAX_RISK_PER_TRADE_CCY = HARD_MAX_RISK_PER_TRADE_CCY


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


def _max_risk_per_trade_ccy() -> float:
    return configured_cash_risk_limit()


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
    """Return units sized to percentage risk with an absolute cash-risk ceiling.

    The percentage risk remains the strategy request, but the final broker-side
    stop exposure is capped by MAX_RISK_PER_TRADE_CCY (default 0.50 in account
    currency). Adaptive learning may only reduce or block risk; it cannot raise
    the requested percentage or bypass the absolute cash ceiling.

    When the stop exposure needs currency conversion, new positions reserve 2%
    of the cash ceiling for conversion drift. Smaller percentage-risk requests
    stay unchanged. This buffer is not a guarantee against larger moves, and
    never changes the cash limit enforced by the broker audit.
    """

    try:
        equity = float(equity)
        stop_distance = float(stop_distance)
        risk_pct = float(risk_pct)
    except (TypeError, ValueError, OverflowError):
        return 0, {}
    if not all(math.isfinite(value) for value in (equity, stop_distance, risk_pct)):
        return 0, {}
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

    try:
        policy_scale = float(policy.risk_scale)
    except (TypeError, ValueError, OverflowError):
        policy_scale = 0.0
    learning_scale = (
        max(0.0, min(1.0, policy_scale)) if math.isfinite(policy_scale) else 0.0
    )
    effective_risk_pct = min(float(risk_pct), float(risk_pct) * learning_scale)
    if policy.blocked or effective_risk_pct <= 0:
        diagnostics = {
            "equity": equity,
            "risk_pct": 0.0,
            "requested_risk_amount": 0.0,
            "risk_amount": 0.0,
            "max_risk_per_trade_ccy": _max_risk_per_trade_ccy(),
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

    requested_risk_amount = equity * effective_risk_pct
    max_risk_ccy = _max_risk_per_trade_ccy()
    if max_risk_ccy <= 0:
        return 0, {
            "equity": equity,
            "risk_pct": effective_risk_pct,
            "requested_risk_pct": risk_pct,
            "requested_risk_amount": requested_risk_amount,
            "risk_amount": 0.0,
            "max_risk_per_trade_ccy": max_risk_ccy,
            "stop_pips": stop_pips,
            "pip_value_per_unit": 0.0,
            "final_units": 0,
            "reason": "invalid-cash-risk-cap",
        }
    _, quote_ccy = _instrument_currencies(instrument)
    conversion_risk_reserve_pct = (
        CROSS_CURRENCY_SIZING_RESERVE_PCT
        if quote_ccy != account_currency.upper()
        else 0.0
    )
    cash_risk_sizing_limit = max_risk_ccy * (1.0 - conversion_risk_reserve_pct)
    risk_amount = min(requested_risk_amount, cash_risk_sizing_limit)
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

    # Rounding up to the broker minimum must never exceed the cash-risk budget.
    final_units = math.floor(raw_units)
    risk_per_unit = stop_pips * pip_value_per_unit
    planned_stop_risk = final_units * risk_per_unit
    # Defensive postcondition: floating-point edge cases must never round the
    # final order above the cash-risk budget.
    while final_units > 0 and planned_stop_risk > risk_amount:
        final_units -= 1
        planned_stop_risk = final_units * risk_per_unit
    if final_units < max(1, int(min_trade_units)):
        return 0, {
            "risk_amount": risk_amount,
            "max_risk_per_trade_ccy": max_risk_ccy,
            "cash_risk_sizing_limit": cash_risk_sizing_limit,
            "conversion_risk_reserve_pct": conversion_risk_reserve_pct,
            "final_units": 0,
            "reason": "minimum-units-exceed-risk-budget",
        }
    diagnostics = {
        "equity": equity,
        "risk_pct": effective_risk_pct,
        "requested_risk_pct": risk_pct,
        "requested_risk_amount": requested_risk_amount,
        "risk_amount": risk_amount,
        "max_risk_per_trade_ccy": max_risk_ccy,
        "cash_risk_sizing_limit": cash_risk_sizing_limit,
        "conversion_risk_reserve_pct": conversion_risk_reserve_pct,
        "stop_pips": stop_pips,
        "pip_value_per_unit": pip_value_per_unit,
        "final_units": final_units,
        "planned_stop_risk": planned_stop_risk,
        "learning_scale": learning_scale,
        "learning_reason": policy.reason,
        "learning_setup_key": policy.setup_key,
        "learning_blocked": False,
        "learning_exact_samples": policy.exact_samples,
        "learning_pair_side_samples": policy.pair_side_samples,
    }
    if max_risk_ccy > 0 and requested_risk_amount > risk_amount:
        print(
            f"[POSITION-SIZE][CASH-CAP] instrument={instrument} "
            f"requested_risk={requested_risk_amount:.2f} capped_risk={risk_amount:.2f} "
            f"cash_limit={max_risk_ccy:.2f} conversion_reserve={conversion_risk_reserve_pct:.2%} "
            f"currency={account_currency}",
            flush=True,
        )
    return final_units, diagnostics
