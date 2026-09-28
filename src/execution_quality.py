from __future__ import annotations

import math
from typing import Any


def _finite(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def pip_size(instrument: str) -> float:
    name = str(instrument or "").upper()
    if name.endswith("_JPY"):
        return 0.01
    if name == "XAU_USD":
        return 0.1
    return 0.0001


def summarize_fill_execution(
    payload: object,
    *,
    side: str,
    instrument: str,
) -> dict[str, Any] | None:
    """Extract OANDA fill-quality evidence without altering execution.

    Positive depth_impact_pips means the fill was worse than the best executable
    price in OANDA's fullPrice snapshot; negative means price improvement.
    """

    if not isinstance(payload, dict):
        return None
    fill = payload.get("orderFillTransaction")
    if not isinstance(fill, dict):
        return None
    full_price = fill.get("fullPrice")
    if not isinstance(full_price, dict):
        return None

    bids = full_price.get("bids")
    asks = full_price.get("asks")
    if not isinstance(bids, list) or not isinstance(asks, list) or not bids or not asks:
        return None
    if not isinstance(bids[0], dict) or not isinstance(asks[0], dict):
        return None

    bid = _finite(bids[0].get("price"))
    ask = _finite(asks[0].get("price"))
    opened = fill.get("tradeOpened")
    fill_price = _finite(opened.get("price")) if isinstance(opened, dict) else _finite(fill.get("fullVWAP"))
    full_vwap = _finite(fill.get("fullVWAP"))
    half_spread_cost = _finite(fill.get("halfSpreadCost"))
    size = pip_size(instrument)
    direction = str(side or "").upper()

    if (
        bid is None
        or ask is None
        or fill_price is None
        or size <= 0
        or bid <= 0
        or ask < bid
        or fill_price <= 0
        or direction not in {"BUY", "SELL"}
    ):
        return None

    executable = ask if direction == "BUY" else bid
    impact = (fill_price - executable) if direction == "BUY" else (executable - fill_price)

    result: dict[str, Any] = {
        "benchmark": "oanda_full_price_at_fill",
        "fill_timestamp_utc": str(fill.get("time") or ""),
        "full_price_timestamp_utc": str(full_price.get("timestamp") or full_price.get("time") or ""),
        "spread_pips_at_fill": (ask - bid) / size,
        "depth_impact_pips": impact / size,
        "half_spread_cost_ccy": half_spread_cost,
        "full_vwap": full_vwap,
        "fill_price": fill_price,
        "best_executable_price": executable,
        "fill_reason": str(fill.get("reason") or ""),
    }

    liquidity = _finite((asks[0] if direction == "BUY" else bids[0]).get("liquidity"))
    if liquidity is not None:
        result["top_of_book_liquidity"] = liquidity
    return result


def aggregate_execution_quality(trades: list[dict[str, Any]]) -> dict[str, Any]:
    samples = [
        trade.get("execution_quality")
        for trade in trades
        if isinstance(trade.get("execution_quality"), dict)
    ]

    def values(key: str) -> list[float]:
        result: list[float] = []
        for sample in samples:
            value = _finite(sample.get(key))
            if value is not None:
                result.append(value)
        return result

    spreads = values("spread_pips_at_fill")
    impacts = values("depth_impact_pips")
    costs = values("half_spread_cost_ccy")
    return {
        "sample_count": len(samples),
        "spread_sample_count": len(spreads),
        "impact_sample_count": len(impacts),
        "cost_sample_count": len(costs),
        "avg_spread_pips_at_fill": (sum(spreads) / len(spreads)) if spreads else None,
        "avg_depth_impact_pips": (sum(impacts) / len(impacts)) if impacts else None,
        "worst_depth_impact_pips": max(impacts) if impacts else None,
        "avg_half_spread_cost_ccy": (sum(costs) / len(costs)) if costs else None,
        "adverse_impact_count": sum(1 for value in impacts if value > 0),
        "improved_impact_count": sum(1 for value in impacts if value < 0),
    }
