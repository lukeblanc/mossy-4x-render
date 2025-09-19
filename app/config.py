import os
from pydantic import BaseModel

class Settings(BaseModel):
    MODE: str = os.getenv("MODE", "demo")  # demo | live
    TZ: str = os.getenv("TZ", "Australia/Perth")
    HEARTBEAT_SECONDS: int = int(os.getenv("HEARTBEAT_SECONDS", "30"))
    DECISION_SECONDS: int = int(os.getenv("DECISION_SECONDS", "60"))
    OANDA_API_KEY: str | None = os.getenv("OANDA_API_KEY")
    OANDA_ACCOUNT_ID: str | None = os.getenv("OANDA_ACCOUNT_ID")

settings = Settings()
