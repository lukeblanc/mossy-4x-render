"""Recover a full close from exact opening/closing fills in a bounded ID range."""
from datetime import datetime
from decimal import Decimal, InvalidOperation


def _number(value):
    try:
        number = Decimal(str(value))
        return number if number.is_finite() else None
    except InvalidOperation:
        return None


def _time(value):
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed if parsed.tzinfo is not None else None
    except ValueError:
        return None


def read_full_close(client, account, ticket):
    response = client.get(f"/v3/accounts/{account}/transactions/{ticket}")
    if response.status_code == 404:
        return None
    if response.status_code != 200:
        raise RuntimeError(f"opening transaction unavailable: HTTP {response.status_code}")
    payload = response.json()
    opening = payload.get("transaction") if isinstance(payload, dict) else None
    if (not isinstance(opening, dict) or str(opening.get("id")) != ticket
            or opening.get("accountID") != account or opening.get("type") != "ORDER_FILL"):
        return None
    opened = opening.get("tradeOpened")
    if not isinstance(opened, dict) or str(opened.get("tradeID")) != ticket:
        return None
    units, price = _number(opened.get("units")), _number(opened.get("price"))
    opened_at = _time(opening.get("time"))
    instrument = opening.get("instrument")
    latest = str(payload.get("lastTransactionID") or "")
    if (units is None or units == 0 or price is None or price <= 0 or opened_at is None
            or not instrument or not latest.isascii() or not latest.isdigit() or int(latest) < int(ticket)):
        return None
    upper = min(int(latest), int(ticket) + 999)
    response = client.get(f"/v3/accounts/{account}/transactions/idrange",
                          params={"from": ticket, "to": str(upper), "type": "ORDER_FILL"})
    if response.status_code != 200:
        raise RuntimeError(f"closing transaction range unavailable: HTTP {response.status_code}")
    payload = response.json()
    transactions = payload.get("transactions") if isinstance(payload, dict) else None
    if not isinstance(transactions, list) or len(transactions) > 1000:
        return None
    matches = []
    seen = set()
    for transaction in transactions:
        if not isinstance(transaction, dict):
            return None
        tx_id = str(transaction.get("id") or "")
        if (not tx_id.isascii() or not tx_id.isdigit() or tx_id in seen
                or not int(ticket) <= int(tx_id) <= upper or transaction.get("accountID") != account):
            return None
        seen.add(tx_id)
        reduced = transaction.get("tradeReduced")
        if isinstance(reduced, dict) and str(reduced.get("tradeID")) == ticket:
            # A partial-close history needs a separate cumulative reconciliation.
            return None
        closed = transaction.get("tradesClosed", [])
        if not isinstance(closed, list):
            return None
        for trade in closed:
            if isinstance(trade, dict) and str(trade.get("tradeID")) == ticket:
                matches.append((transaction, trade))
    if len(matches) != 1:
        return None
    closing, closed = matches[0]
    closed_at = _time(closing.get("time"))
    close_units, close_price, pnl = (_number(closed.get(key)) for key in ("units", "price", "realizedPL"))
    if (closing.get("type") != "ORDER_FILL" or closing.get("instrument") != instrument
            or int(closing["id"]) <= int(ticket) or closed_at is None or closed_at < opened_at
            or close_units is None or abs(close_units) != abs(units)
            or close_price is None or close_price <= 0 or pnl is None):
        return None
    print(f"[JOURNAL][LOOKUP] trade_id={ticket} source=exact-transactions "
          f"closing_transaction={closing['id']} state=CLOSED", flush=True)
    return {"id": ticket, "instrument": instrument, "state": "CLOSED", "currentUnits": "0",
            "initialUnits": str(units), "price": str(price), "openTime": opening["time"],
            "averageClosePrice": str(close_price), "realizedPL": str(pnl), "closeTime": closing["time"],
            "_close_evidence": {"source": "exact-transactions", "account_id": account,
                "range_from": ticket, "range_to": str(upper), "opening": opening, "closing": closing}}
