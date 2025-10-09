from __future__ import annotations

import httpx

from app.config import settings

PRACTICE = "https://api-fxpractice.oanda.com"
LIVE = "https://api-fxtrade.oanda.com"

class Broker:
    def __init__(self):
        self.mode = (settings.MODE or "demo").lower()
        self.account = settings.OANDA_ACCOUNT_ID
        self.key = settings.OANDA_API_KEY
        if self.mode == "demo":
            self.base_url = PRACTICE
        elif self.mode == "live":
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

    def place_order(self, instrument: str, signal: str, units: float) -> dict:
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
        payload = {
            "order": {
                "type": "MARKET",
                "instrument": instrument,
                "units": str(trade_units),
            }
        }

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

            
