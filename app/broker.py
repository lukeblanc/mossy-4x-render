from __future__ import annotations
import json
import httpx
from app.config import settings

PRACTICE = "https://api-fxpractice.oanda.com"
LIVE = "https://api-fxtrade.oanda.com"

class Broker:
    def __init__(self):
        self.mode = (settings.MODE or "demo").lower()
        self.account = settings.OANDA_ACCOUNT_ID
        self.key = settings.OANDA_API_KEY
        self.base_url = PRACTICE if self.mode == "demo" else LIVE
        self._headers = {"Authorization": f"Bearer {self.key}"} if self.key else {}

    def _client(self) -> httpx.Client:
        return httpx.Client(base_url=self.base_url, headers=self._headers, timeout=15.0)

    def connectivity_check(self) -> dict:
        """Log a quick read-only call to prove creds (demo or live)."""
        if not (self.key and self.account):
            print("[OANDA] No credentials set; skipping connectivity check.")
            return {"ok": False, "reason": "no-creds"}
        try:
            with self._client() as c:
                r = c.get(f"/v3/accounts/{self.account}/summary")
                if r.status_code == 200:
                    data = r.json().get("account", {})
                    bal = data.get("balance")
                    curr = data.get("currency")
                    print(f"[OANDA] Connected ok. Balance={bal} {curr} (mode={self.mode})", flush=True)
                    return {"ok": True, "balance": bal, "currency": curr}
                else:
                    print(f"[OANDA] Connectivity error {r.status_code}: {r.text}", flush=True)
                    return {"ok": False, "status": r.status_code}
        except Exception as e:
            print(f"[OANDA] Connectivity exception: {e}", flush=True)
            return {"ok": False, "error": str(e)}

def place_order(self, instrument: str, signal: str, units: float):
    side = signal.upper()
    if side not in ("BUY", "SELL"):
        print(f"[BROKER] Ignoring unknown signal: {signal}", flush=True)
        return {"status": "IGNORED"}

    if self.mode.lower() != "live" or not (self.key and self.account):
        print(f"[BROKER] {self.mode.upper()} SIMULATED {side} order for {instrument} size={units}", flush=True)
        return {"status": "SIMULATED"}

    trade_units = str(int(units if side == "BUY" else -units))
    payload = {
        "order": {
            "type": "MARKET",
            "instrument": instrument,
            "units": trade_units,
        }
    }

    try:
        with self._client() as c:
            r = c.post(f"/v3/accounts/{self.account}/orders", json=payload)
            if r.status_code in (200, 201):
                print(f"[BROKER] LIVE {side} sent order for {instrument} size={units} resp={r.status_code}", flush=True)
                return {"status": "SENT", "resp": r.json()}
            else:
                print(f"[BROKER] LIVE order error {r.status_code}: {r.text}", flush=True)
                return {"status": "ERROR", "code": r.status_code, "text": r.text}
        except Exception as e:
            print(f"[BROKER] LIVE order exception: {e}", flush=True)
                  return {"status": "ERROR", "error": str(e)} 

            
