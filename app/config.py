import os
from pydantic import BaseModel
from typing import Optional

class Settings(BaseModel):
    # Core
    MODE: str = os.getenv("MODE", "demo")
    TZ: str = os.getenv("TZ", "Australia/Perth")
    HEARTBEAT_SECONDS: int = int(os.getenv("HEARTBEAT_SECONDS", "30"))
    DECISION_SECONDS: int = int(os.getenv("DECISION_SECONDS", "60"))

    # Health / alerts
    MAX_SILENCE_SECONDS: int = int(os.getenv("MAX_SILENCE_SECONDS", "180"))
    ERROR_BURST_THRESHOLD: int = int(os.getenv("ERROR_BURST_THRESHOLD", "3"))
    ALERT_EMAIL: Optional[str] = os.getenv("ALERT_EMAIL")
    SMTP_HOST: Optional[str] = os.getenv("SMTP_HOST")
    SMTP_PORT: int = int(os.getenv("SMTP_PORT", "587"))
    SMTP_USER: Optional[str] = os.getenv("SMTP_USER")
    SMTP_PASS: Optional[str] = os.getenv("SMTP_PASS")

    # Trading
    OANDA_API_KEY: Optional[str] = os.getenv("OANDA_API_KEY")
    OANDA_ACCOUNT_ID: Optional[str] = os.getenv("OANDA_ACCOUNT_ID")
    INSTRUMENT: str = os.getenv("INSTRUMENT", "EUR_USD")
    ORDER_SIZE: float = float(os.getenv("ORDER_SIZE", "1000"))

    ...


