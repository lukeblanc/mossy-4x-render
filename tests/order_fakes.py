def confirmed_order_result(instrument="EUR_USD", signal="BUY", units=100,
                           trade_id="2", price=1.2):
    """A realistic OANDA opening fill for decision tests, without network access."""
    return {"status": "SENT", "response": {
        "orderCreateTransaction": {"id": "1"},
        "orderFillTransaction": {
            "id": "2", "instrument": instrument, "time": "2026-09-08T06:00:00Z",
            "tradeOpened": {"tradeID": trade_id, "price": str(price),
                            "units": str(abs(units) if signal == "BUY" else -abs(units))},
        },
        "stopLossOrderTransaction": {"id": "3"}, "lastTransactionID": "3",
    }}
