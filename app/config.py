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

    # Strategy parameters
    STRAT_EMA_FAST: int = int(os.getenv("STRAT_EMA_FAST", "10"))
    STRAT_EMA_SLOW: int = int(os.getenv("STRAT_EMA_SLOW", "20"))
    STRAT_RSI_LEN: int = int(os.getenv("STRAT_RSI_LEN", "14"))
    STRAT_RSI_BUY: int = int(os.getenv("STRAT_RSI_BUY", "55"))
    STRAT_RSI_SELL: int = int(os.getenv("STRAT_RSI_SELL", "45"))
    STRAT_TIMEFRAME: str = os.getenv("STRAT_TIMEFRAME", "M5")
    STRAT_COOLDOWN_BARS: int = int(os.getenv("STRAT_COOLDOWN_BARS", "2"))
    ATR_LEN: int = int(os.getenv("ATR_LEN", "14"))
            # Additional rule variables
        MAX_RISK_PER_TRADE: float = float(os.getenv("MAX_RISK_PER_TRADE", "0.02"))
        DAILY_LOSS_CAP: float = float(os.getenv("DAILY_LOSS_CAP", "0.05"))
        DRAWDOWN_CAP: float = float(os.getenv("DRAWDOWN_CAP", "0.10"))
        NEWS_GUARD_MINUTES: int = int(os.getenv("NEWS_GUARD_MINUTES", "60"))
        EXIT_LOGIC: str = os.getenv("EXIT_LOGIC", "chandelier")
        MAX_OPEN_TRADES: int = int(os.getenv("MAX_OPEN_TRADES", "3"))
        ADX_FILTER: int = int(os.getenv("ADX_FILTER", "25"))
        MTF_ALIGN: bool = os.getenv("MTF_ALIGN", "True").lower() == "true"
        RULES_VERSION: str = os.getenv("RULES_VERSION", "V1.6")

    MIN_ATR: float = float(os.getenv("MIN_ATR", "0.00005"))

    # Risk
    RISK_PCT: float = float(os.getenv("RISK_PCT", "0.01"))
    SL_R: float = float(os.getenv("SL_R", "1.0"))
    TP_R: float = float(os.getenv("TP_R", "1.5"))
    MAX_DD_DAY: float = float(os.getenv("MAX_DD_DAY", "0.05"))

settings = Settings()



