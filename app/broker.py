from __future__ import annotations

from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Dict, Optional

import httpx

from app.config import settings

PRACTICE = "https://api-fxpractice.oanda.com"
LIVE = "https://api-fxtrade.oanda.com"


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
                    if self.mode == "demo":
                        order_id = (
                            data.get("orderCreateTransaction", {}).get("id")
                            or data.get("orderFillTransaction", {}).get("id")
                            or data.get("lastTransactionID")
                        )
                        print(f"[OANDA] DEMO ORDER SENT id={order_id}", flush=True)
                    else:
                        print(
                            f"[BROKER] LIVE {side} sent order for {instrument} size={units} resp={resp.status_code}",
                            flush=True,
                        )
                    return {"status": "SENT", "response": data}
                if self.mode == "demo":
                    print(f"[OANDA] DEMO ORDER FAILED {resp.text}", flush=True)
                else:
                    print(
                        f"[BROKER] LIVE order error {resp.status_code}: {resp.text}",
                        flush=True,
                    )
                return {"status": "ERROR", "code": resp.status_code, "text": resp.text}
        except Exception as exc:
            if self.mode == "demo":
                print(f"[OANDA] DEMO ORDER FAILED {exc}", flush=True)
            else:
                print(f"[BROKER] LIVE order exception: {exc}", flush=True)
            return {"status": "ERROR", "error": str(exc)}

    def list_open_trades(self) -> list:
        """Return currently open trades for the configured account."""
        if self.mode == "simulation" or not (self.key and self.account):
            return []
        try:
            with self._client() as client:
                resp = client.get(f"/v3/accounts/{self.account}/openTrades")
                if resp.status_code == 200:
                    data = resp.json()
                    return data.get("trades", [])
                print(
                    f"[OANDA] Failed to read open trades status={resp.status_code} body={resp.text}",
                    flush=True,
                )
        except Exception as exc:
            print(f"[OANDA] Exception fetching open trades: {exc}", flush=True)
        return []

    def get_unrealized_profit(self, instrument: str) -> Optional[float]:
        """Return the unrealized P/L for the given instrument in account currency."""
        if not instrument:
            return None
        if self.mode == "simulation" or not (self.key and self.account):
            return 0.0
        try:
            with self._client() as client:
                resp = client.get(
                    f"/v3/accounts/{self.account}/positions/{instrument}"
                )
                if resp.status_code != 200:
                    return 0.0
                position = resp.json().get("position", {}) or {}
                unrealized = position.get("unrealizedPL")
                if unrealized is not None:
                    try:
                        return float(unrealized)
                    except (TypeError, ValueError):
                        return 0.0
                total = 0.0
                found = False
                for side in ("long", "short"):
                    side_pl = (position.get(side) or {}).get("unrealizedPL")
                    try:
                        total += float(side_pl)
                        found = True
                    except (TypeError, ValueError):
                        continue
                return total if found else 0.0
        except Exception as exc:
            print(
                f"[OANDA] Exception fetching unrealized P/L for {instrument}: {exc}",
                flush=True,
            )
            return 0.0

    def position_snapshot(self, instrument: str) -> Optional[Dict]:
        """Return the broker position payload for the instrument, or None on error."""

        if not instrument:
            return None
        if self.mode == "simulation" or not (self.key and self.account):
            return {
                "instrument": instrument,
                "long": {"units": "0"},
                "short": {"units": "0"},
                "longUnits": "0",
                "shortUnits": "0",
            }
        try:
            with self._client() as client:
                resp = client.get(f"/v3/accounts/{self.account}/positions/{instrument}")
                if resp.status_code != 200:
                    return None
                return resp.json().get("position", {}) or {}
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

    def account_equity(self) -> float:
        if not (self.key and self.account):
            return 0.0
        try:
            with self._client() as client:
                resp = client.get(f"/v3/accounts/{self.account}/summary")
                if resp.status_code == 200:
                    data = resp.json().get("account", {})
                    nav = data.get("NAV") or data.get("balance")
                    try:
                        return float(nav)
                    except (TypeError, ValueError):
                        return 0.0
        except Exception as exc:
            print(f"[OANDA] Exception fetching equity: {exc}", flush=True)
        return 0.0

    def current_spread(self, instrument: str) -> float:
        if not (self.key and self.account):
            return 0.0
        try:
            with self._client() as client:
                resp = client.get(
                    f"/v3/accounts/{self.account}/pricing",
                    params={"instruments": instrument},
                )
                if resp.status_code != 200:
                    return 0.0
                data = resp.json().get("prices", [])
                if not data:
                    return 0.0
                price = data[0]
                bids = price.get("bids") or []
                asks = price.get("asks") or []
                if not bids or not asks:
                    return 0.0
                try:
                    bid = float(bids[0]["price"])
                    ask = float(asks[0]["price"])
                except (KeyError, TypeError, ValueError):
                    return 0.0
                spread = ask - bid
                pip_size = self._pip_size(instrument)
                if pip_size <= 0:
                    return 0.0
                return spread / pip_size
        except Exception as exc:
            print(f"[OANDA] Exception fetching spread for {instrument}: {exc}", flush=True)
            return 0.0

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
