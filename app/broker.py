from app.config import settings

class Broker:
    def __init__(self):
        self.mode = settings.MODE
        self.account = settings.OANDA_ACCOUNT_ID
        self.key = settings.OANDA_API_KEY

    def place_order(self, side: str, size: float = 1.0):
        if not self.key or not self.account or self.mode != "live":
            print(f"[BROKER] ({self.mode}) Simulated {side} order size={size}")
            return {"status":"SIMULATED"}
        # TODO: implement real OANDA API order
        print(f"[BROKER] LIVE {side} order size={size} (placeholder)")
        return {"status":"SENT"}
