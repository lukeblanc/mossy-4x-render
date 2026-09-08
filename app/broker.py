from __future__ import annotations

from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from datetime import datetime, timezone
import math
from typing import Dict, Optional

import httpx

from app.config import settings

PRACTICE = "https://api-fxpractice.oanda.com"
LIVE = "https://api-fxtrade.oanda.com"


def read_trade_details(client, account: str, trade_id: str) -> Optional[Dict]:
    """Read one exact trade, with a bounded list fallback for detail 404s."""
    ticket = str(trade_id)
    if not ticket.isascii() or not ticket.isdigit() or int(ticket) <= 0:
        return None
    response = client.get(f"/v3/accounts/{account}/trades/{ticket}")
    if response.status_code == 404:
        # Explicit IDs and ALL include closed trades without scanning history.
        response = client.get(f"/v3/accounts/{account}/trades",
                              params={"ids": ticket, "state": "ALL", "count": 1})
        if response.status_code != 200:
            raise RuntimeError(f"broker trade list unavailable: HTTP {response.status_code}")
        payload = response.json()
        candidates = payload.get("trades") if isinstance(payload, dict) else None
        if not isinstance(candidates, list) or len(candidates) != 1:
            print(f"[JOURNAL][LOOKUP] trade_id={ticket} exact_list_match=False", flush=True)
            return None
        trade = candidates[0]
        source = "exact-list"
    else:
        if response.status_code != 200:
            raise RuntimeError(f"broker trade details unavailable: HTTP {response.status_code}")
        payload = response.json()
        trade = payload.get("trade") if isinstance(payload, dict) else None
        source = "detail"
    if not isinstance(trade, dict) or str(trade.get("id")) != ticket:
        return None
    if source == "exact-list":
        print(f"[JOURNAL][LOOKUP] trade_id={ticket} source={source} state={trade.get('state')}", flush=True)
    return trade


def opened_trade_fill(payload: object) -> Optional[Dict]:
    """Return only a verified new trade, never an order or a reduced/closed trade."""
    if not isinstance(payload, dict):
        return None
    fill = payload.get("orderFillTransaction")
    if not isinstance(fill, dict):
        return None
    opened = fill.get("tradeOpened")
    if not isinstance(opened, dict):
        return None
    trade_id = str(opened.get("tradeID") or "")
    try:
        price = float(opened["price"])
        units = float(opened["units"])
        timestamp = datetime.fromisoformat(str(fill["time"]).replace("Z", "+00:00"))
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    if (not trade_id.isascii() or not trade_id.isdigit() or int(trade_id) <= 0
            or not math.isfinite(price) or price <= 0
            or not math.isfinite(units) or units == 0 or timestamp.tzinfo is None):
        return None
    return {"trade_id": trade_id, "price": price, "units": units,
            "timestamp": timestamp.astimezone(timezone.utc)}


def _precision_for(instrument: str) -> Decimal:
    return {
        "USD_JPY": Decimal("0.001"),
        "XAU_USD": Decimal("0.01"),
    }.get(instrument, Decimal("0.00001"))


def _quantize_value(value: float | Decimal | str, precision: Decimal) -> Decimal:
    try:
        dec_value = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        raise ValueError(f"Invalid numeric value for normalization: {value}")

    return dec_value.quantize(precision, rounding=ROUND_HALF_UP)


def normalize_price(instrument: str, price: float | Decimal | str) -> str:
    precision = _precision_for(instrument)
    dec_price = _quantize_value(price, precision)

    return str(dec_price)


def normalize_distance(instrument: str, distance: float | Decimal | str) -> str:
    precision = _precision_for(instrument)
    dec_distance = _quantize_value(distance, precision)

    return str(dec_distance)


class Broker:
    def __init__(self):
        # Guard against accidental live usage
        env_label = (getattr(settings, "OANDA_ENV", "practice") or "practice").lower()
        self.mode = (settings.MODE or "demo").lower()
        if env_label == "live" or self.mode == "live":
            print("[OANDA] Live trading is disabled in this deployment. Exiting.", flush=True)
            raise SystemExit(1)

        self.account = settings.OANDA_ACCOUNT_ID
        self.key = settings.OANDA_API_KEY
        if env_label == "practice" or self.mode == "demo":
            self.base_url = PRACTICE
        elif env_label == "live" or self.mode == "live":
            self.base_url = LIVE
        else:
            self.base_url = PRACTICE
        self._headers = {"Authorization": f"Bearer {self.key}"} if self.key else {}

    def _client(self) -> httpx.Client:
        return httpx.Client(base_url=self.base_url, headers=self._headers, timeout=15.0)

    def trade_details(self, trade_id: str) -> Optional[Dict]:
        if not self.key or not self.account:
            raise RuntimeError("broker credentials unavailable")
        with self._client() as client:
            return read_trade_details(client, self.account, trade_id)

    def connectivity_check(self) -> dict:
        """Log a quick read-only call to prove creds (demo or live)."""
        if not (self.key and self.account):
            print("[OANDA] No credentials set; skipping connectivity check.")
            return {"ok": False, "reason": "no-creds"}
        try:
            with self._client() as client:
                resp = client.get(f"/v3/accounts/{self.account}/summary")
                if resp.status_code == 200:
                    data = resp.json().get("account", {})
                    balance = data.get("balance")
                    currency = data.get("currency")
                    print(
                        f"[OANDA] Connected ok. Balance={balance} {currency} (mode={self.mode})",
                        flush=True,
                    )
                    return {"ok": True, "balance": balance, "currency": currency}
                print(
                    f"[OANDA] Connectivity error {resp.status_code}: {resp.text}",
                    flush=True,
                )
                return {"ok": False, "status": resp.status_code, "text": resp.text}
        except Exception as exc:
            print(f"[OANDA] Connectivity exception: {exc}", flush=True)
            return {"ok": False, "error": str(exc)}

    def place_order(
        self,
        instrument: str,
        signal: str,
        units: float,
        *,
        sl_distance: float | None = None,
        tp_distance: float | None = None,
        entry_price: float | None = None,
    ) -> dict:
        side = signal.upper()
        if side not in ("BUY", "SELL"):
            print(f"[BROKER] Ignoring unknown signal: {signal}", flush=True)
            return {"status": "IGNORED", "reason": "unknown-signal"}

        if self.mode == "simulation":
            print(
                f"[BROKER] {self.mode.upper()} SIMULATED {side} order for {instrument} size={units}",
                flush=True,
            )
            return {"status": "SIMULATED"}

        if not (self.key and self.account):
            print(
                f"[BROKER] {self.mode.upper()} order failed: missing credentials.",
                flush=True,
            )
            return {"status": "ERROR", "reason": "missing-creds"}

        trade_units = int(units if side == "BUY" else -units)
        order_payload = {
            "type": "MARKET",
            "instrument": instrument,
            "units": str(trade_units),
        }

        if sl_distance is not None and sl_distance > 0:
            try:
                normalized_sl_distance = normalize_distance(instrument, sl_distance)
            except ValueError:
                normalized_sl_distance = None
            if normalized_sl_distance is not None:
                order_payload["stopLossOnFill"] = {
                    "timeInForce": "GTC",
                    "distance": normalized_sl_distance,
                }
        if (
            entry_price is not None
            and tp_distance is not None
            and tp_distance > 0
        ):
            try:
                entry_val = Decimal(str(entry_price))
                tp_val = Decimal(str(tp_distance))
            except (TypeError, ValueError, InvalidOperation):
                entry_val = None
                tp_val = None
            if entry_val is not None and tp_val is not None:
                if side == "BUY":
                    tp_price = entry_val + tp_val
                else:
                    tp_price = entry_val - tp_val
                rounded_tp = normalize_price(instrument, tp_price)
                print(
                    f"[ORDER_FMT] instrument={instrument} raw_tp={tp_price} rounded_tp={rounded_tp}",
                    flush=True,
                )
                order_payload["takeProfitOnFill"] = {
                    "timeInForce": "GTC",
                    "price": rounded_tp,
                }

        payload = {"order": order_payload}

        try:
            with self._client() as client:
                resp = client.post(f"/v3/accounts/{self.account}/orders", json=payload)
                if resp.status_code in (200, 201):
                    data = resp.json()
                    if not isinstance(data, dict):
                        return {"status": "UNKNOWN", "reason": "invalid-order-response"}
                    if data.get("orderCancelTransaction"):
                        return {"status": "CANCELLED", "response": data}
                    if data.get("orderRejectTransaction"):
                        return {"status": "REJECTED", "response": data}
                    filled = opened_trade_fill(data)
                    if filled is None:
                        return {"status": "UNKNOWN", "reason": "no-confirmed-trade-opening", "response": data}
                    if (data["orderFillTransaction"].get("instrument") != instrument
                            or filled["units"] * trade_units <= 0):
                        return {"status": "UNKNOWN", "reason": "fill-does-not-match-request", "response": data}
                    print(f"[OANDA] DEMO TRADE OPENED trade_id={filled['trade_id']} "
                          f"price={filled['price']} units={filled['units']}", flush=True)
                    return {"status": "SENT", "response": data}
                if self.mode == "demo":
                    print(f"[OANDA] DEMO ORDER FAILED {resp.text}", flush=True)
                else:
                    print(
                        f"[BROKER] LIVE order error {resp.status_code}: {resp.text}",
                        flush=True,
                    )
                return {"status": "UNKNOWN" if resp.status_code >= 500 else "ERROR",
                        "code": resp.status_code, "text": resp.text}
        except Exception as exc:
            if self.mode == "demo":
                print(f"[OANDA] DEMO ORDER FAILED {exc}", flush=True)
            else:
                print(f"[BROKER] LIVE order exception: {exc}", flush=True)
            return {"status": "UNKNOWN", "error": str(exc)}

    def list_open_trades(self) -> Optional[list]:
        """Return currently open trades, or ``None`` when the broker cannot be read.

        An empty list is a valid broker answer.  It must not also be used as the
        error value because callers use this result to prevent duplicate exposure.
        """
        if self.mode == "simulation":
            return []
        if not (self.key and self.account):
            return None
        try:
            with self._client() as client:
                resp = client.get(f"/v3/accounts/{self.account}/openTrades")
                if resp.status_code == 200:
                    data = resp.json()
                    trades = data.get("trades")
                    if not isinstance(trades, list) or any(
                        not isinstance(trade, dict) or not trade.get("instrument")
                        for trade in trades
                    ):
                        return None
                    return trades
                print(
                    f"[OANDA] Failed to read open trades status={resp.status_code} body={resp.text}",
                    flush=True,
                )
        except Exception as exc:
            print(f"[OANDA] Exception fetching open trades: {exc}", flush=True)
        return None

    def get_unrealized_profit(self, instrument: str) -> Optional[float]:
        """Return the unrealized P/L for the given instrument in account currency."""
        if not instrument:
            return None
        if self.mode == "simulation":
            return 0.0
        if not (self.key and self.account):
            return None
        try:
            with self._client() as client:
                resp = client.get(
                    f"/v3/accounts/{self.account}/positions/{instrument}"
                )
                if resp.status_code != 200:
                    return None
                position = resp.json().get("position", {}) or {}
                unrealized = position.get("unrealizedPL")
                if unrealized is not None:
                    try:
                        value = float(unrealized)
                        return value if math.isfinite(value) else None
                    except (TypeError, ValueError):
                        return None
                total = 0.0
                found = False
                for side in ("long", "short"):
                    side_pl = (position.get(side) or {}).get("unrealizedPL")
                    try:
                        total += float(side_pl)
                        found = True
                    except (TypeError, ValueError):
                        continue
                return total if found and math.isfinite(total) else None
        except Exception as exc:
            print(
                f"[OANDA] Exception fetching unrealized P/L for {instrument}: {exc}",
                flush=True,
            )
            return None

    def position_snapshot(self, instrument: str) -> Optional[Dict]:
        """Return the broker position payload for the instrument, or None on error."""

        if not instrument:
            return None
        if self.mode == "simulation":
            return {
                "instrument": instrument,
                "long": {"units": "0"},
                "short": {"units": "0"},
                "longUnits": "0",
                "shortUnits": "0",
            }
        if not (self.key and self.account):
            return None
        try:
            with self._client() as client:
                resp = client.get(f"/v3/accounts/{self.account}/positions/{instrument}")
                if resp.status_code != 200:
                    return None
                position = resp.json().get("position")
                if not isinstance(position, dict):
                    return None
                for side in ("long", "short"):
                    if not math.isfinite(float(position[side]["units"])):
                        return None
                return position
        except Exception:
            return None

    def close_position_side(self, instrument: str, long_units: float, short_units: float) -> Dict:
        """Close a position using side-specific payloads for the OANDA positions close endpoint."""

        if not instrument:
            return {"status": "ERROR", "reason": "invalid-instrument"}

        if long_units > 0 and short_units == 0:
            payload: Dict[str, str] = {"longUnits": "ALL"}
        elif short_units < 0 and long_units == 0:
            payload = {"shortUnits": "ALL"}
        elif long_units != 0 or short_units != 0:
            payload = {"longUnits": "ALL", "shortUnits": "ALL"}
        else:
            payload = {"longUnits": "0", "shortUnits": "0"}

        if self.mode == "simulation":
            print(f"[BROKER] SIMULATION close position {instrument} payload={payload}", flush=True)
            return {"status": "SIMULATED", "payload": payload}
        if not (self.key and self.account):
            print(
                f"[BROKER] {self.mode.upper()} close failed: missing credentials.",
                flush=True,
            )
            return {"status": "ERROR", "reason": "missing-creds"}

        try:
            with self._client() as client:
                resp = client.put(
                    f"/v3/accounts/{self.account}/positions/{instrument}/close",
                    json=payload,
                )
                if resp.status_code in (200, 201):
                    print(f"[OANDA] Closed position {instrument} payload={payload}", flush=True)
                    return {"status": "CLOSED", "response": resp.json()}
                print(
                    f"[OANDA] Failed to close {instrument} status={resp.status_code} body={resp.text}",
                    flush=True,
                )
                return {
                    "status": "ERROR",
                    "code": resp.status_code,
                    "text": resp.text,
                }
        except Exception as exc:
            print(
                f"[OANDA] Exception closing position {instrument}: {exc}", flush=True
            )
            return {"status": "ERROR", "error": str(exc)}

    # Backwards-compatible wrapper.
    def close_position(
        self,
        instrument: str,
        *,
        long_units: str | None = "ALL",
        short_units: str | None = "ALL",
        trade_id: str | None = None,
    ) -> Dict:
        try:
            long_val = 0.0 if long_units is None else float(long_units)
        except (TypeError, ValueError):
            long_val = 0.0
        try:
            short_val = 0.0 if short_units is None else float(short_units)
        except (TypeError, ValueError):
            short_val = 0.0
        return self.close_position_side(instrument, long_val, short_val)

    def account_equity(self) -> Optional[float]:
        if not (self.key and self.account):
            return None
        try:
            with self._client() as client:
                resp = client.get(f"/v3/accounts/{self.account}/summary")
                if resp.status_code == 200:
                    data = resp.json().get("account", {})
                    nav = data.get("NAV")
                    try:
                        equity = float(nav)
                        return equity if math.isfinite(equity) and equity > 0 else None
                    except (TypeError, ValueError):
                        return None
        except Exception as exc:
            print(f"[OANDA] Exception fetching equity: {exc}", flush=True)
        return None

    def current_spread(self, instrument: str) -> Optional[float]:
        if not (self.key and self.account):
            return None
        try:
            with self._client() as client:
                resp = client.get(
                    f"/v3/accounts/{self.account}/pricing",
                    params={"instruments": instrument},
                )
                if resp.status_code != 200:
                    return None
                data = resp.json().get("prices", [])
                if not data:
                    return None
                price = data[0]
                bids = price.get("bids") or []
                asks = price.get("asks") or []
                if not bids or not asks:
                    return None
                try:
                    bid = float(bids[0]["price"])
                    ask = float(asks[0]["price"])
                except (KeyError, TypeError, ValueError):
                    return None
                spread = ask - bid
                if not (math.isfinite(bid) and math.isfinite(ask)) or bid <= 0 or ask < bid:
                    return None
                pip_size = self._pip_size(instrument)
                if pip_size <= 0:
                    return None
                return spread / pip_size
        except Exception as exc:
            print(f"[OANDA] Exception fetching spread for {instrument}: {exc}", flush=True)
            return None

    def mid_price(self, instrument: str) -> float | None:
        if not (self.key and self.account):
            return None
        try:
            with self._client() as client:
                resp = client.get(
                    f"/v3/accounts/{self.account}/pricing",
                    params={"instruments": instrument},
                )
                if resp.status_code != 200:
                    return None
                data = resp.json().get("prices", [])
                if not data:
                    return None
                price = data[0]
                bids = price.get("bids") or []
                asks = price.get("asks") or []
                if not bids or not asks:
                    return None
                bid = float(bids[0]["price"])
                ask = float(asks[0]["price"])
                return (bid + ask) / 2.0
        except Exception:
            return None

    def conversion_rate(self, from_ccy: str, to_ccy: str) -> float | None:
        source = (from_ccy or "").upper()
        target = (to_ccy or "").upper()
        if not source or not target:
            return None
        if source == target:
            return 1.0

        direct = f"{source}_{target}"
        direct_mid = self.mid_price(direct)
        if direct_mid and direct_mid > 0:
            return direct_mid

        inverse = f"{target}_{source}"
        inverse_mid = self.mid_price(inverse)
        if inverse_mid and inverse_mid > 0:
            return 1.0 / inverse_mid

        return None

    def close_all_positions(self) -> None:
        if not (self.key and self.account):
            return
        try:
            with self._client() as client:
                resp = client.get(f"/v3/accounts/{self.account}/openPositions")
                if resp.status_code != 200:
                    return
                for position in resp.json().get("positions", []):
                    instrument = position.get("instrument")
                    if not instrument:
                        continue
                    payload: Dict[str, str] = {}
                    long_units = position.get("long", {}).get("units")
                    short_units = position.get("short", {}).get("units")
                    if long_units and float(long_units) != 0:
                        payload["longUnits"] = "ALL"
                    if short_units and float(short_units) != 0:
                        payload["shortUnits"] = "ALL"
                    if not payload:
                        continue
                    client.put(
                        f"/v3/accounts/{self.account}/positions/{instrument}/close",
                        json=payload,
                    )
        except Exception as exc:
            print(f"[OANDA] Exception closing positions: {exc}", flush=True)

    @staticmethod
    def _pip_size(instrument: str) -> float:
        if instrument.endswith("JPY"):
            return 0.01
        if instrument.startswith("XAU"):
            return 0.1
        if instrument.startswith("XAG"):
            return 0.01
        return 0.0001
