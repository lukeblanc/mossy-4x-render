from __future__ import annotations

from src.execution_quality import aggregate_execution_quality, summarize_fill_execution


def fill_payload(
    fill_price: str = "1.20030",
    *,
    bid: str = "1.20010",
    ask: str = "1.20020",
    cost: str = "0.07",
):
    return {
        "orderFillTransaction": {
            "time": "2026-09-28T03:00:01.123456789Z",
            "reason": "MARKET_ORDER",
            "fullVWAP": fill_price,
            "halfSpreadCost": cost,
            "fullPrice": {
                "timestamp": "2026-09-28T03:00:01.120000000Z",
                "bids": [{"price": bid, "liquidity": "1000000"}],
                "asks": [{"price": ask, "liquidity": "900000"}],
            },
            "tradeOpened": {"price": fill_price},
        }
    }


def test_buy_uses_ask_as_execution_benchmark():
    result = summarize_fill_execution(
        fill_payload(), side="BUY", instrument="AUD_USD"
    )
    assert result is not None
    assert round(result["spread_pips_at_fill"], 6) == 1.0
    assert round(result["depth_impact_pips"], 6) == 1.0
    assert result["half_spread_cost_ccy"] == 0.07
    assert result["top_of_book_liquidity"] == 900000


def test_sell_uses_bid_as_execution_benchmark():
    result = summarize_fill_execution(
        fill_payload(fill_price="1.20005"), side="SELL", instrument="GBP_USD"
    )
    assert result is not None
    assert round(result["depth_impact_pips"], 6) == 0.5
    assert result["top_of_book_liquidity"] == 1000000


def test_price_improvement_is_negative_impact():
    result = summarize_fill_execution(
        fill_payload(fill_price="1.20015"), side="BUY", instrument="AUD_USD"
    )
    assert result is not None
    assert round(result["depth_impact_pips"], 6) == -0.5


def test_missing_full_price_is_unknown_not_zero():
    assert summarize_fill_execution(
        {"orderFillTransaction": {"tradeOpened": {"price": "1.2"}}},
        side="BUY",
        instrument="AUD_USD",
    ) is None


def test_invalid_money_never_becomes_execution_evidence():
    assert summarize_fill_execution(
        fill_payload(fill_price="NaN"), side="BUY", instrument="AUD_USD"
    ) is None


def test_aggregate_reports_only_real_samples():
    first = summarize_fill_execution(
        fill_payload(), side="BUY", instrument="AUD_USD"
    )
    second = summarize_fill_execution(
        fill_payload(fill_price="1.20015"), side="BUY", instrument="AUD_USD"
    )
    report = aggregate_execution_quality(
        [
            {"execution_quality": first},
            {"execution_quality": second},
            {"execution_quality": None},
        ]
    )
    assert report["sample_count"] == 2
    assert report["impact_sample_count"] == 2
    assert round(report["avg_depth_impact_pips"], 6) == 0.25
    assert round(report["worst_depth_impact_pips"], 6) == 1.0
    assert report["adverse_impact_count"] == 1
    assert report["improved_impact_count"] == 1


def test_jpy_uses_one_hundredth_as_pip():
    result = summarize_fill_execution(
        fill_payload(fill_price="150.025", bid="150.010", ask="150.020"),
        side="BUY",
        instrument="USD_JPY",
    )
    assert result is not None
    assert round(result["spread_pips_at_fill"], 6) == 1.0
    assert round(result["depth_impact_pips"], 6) == 0.5
