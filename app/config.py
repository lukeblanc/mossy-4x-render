import os
from typing import Optional

from pydantic import BaseModel


def _env_str(key: str, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(key)
    return value if value is not None else default


def _env_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


class Settings(BaseModel):
    MODE: str
    TZ: str
    HEARTBEAT_SECONDS: int
    DECISION_SECONDS: int

    MAX_SILENCE_SECONDS: int
    ERROR_BURST_THRESHOLD: int
    ALERT_EMAIL: Optional[str]
    SMTP_HOST: Optional[str]
    SMTP_PORT: int
    SMTP_USER: Optional[str]
    SMTP_PASS: Optional[str]

    OANDA_API_KEY: Optional[str]
    OANDA_ACCOUNT_ID: Optional[str]
    INSTRUMENT: str
    ORDER_SIZE: float

    STRAT_EMA_FAST: int
    STRAT_EMA_SLOW: int
    STRAT_RSI_LEN: int
    STRAT_RSI_BUY: int
    STRAT_RSI_SELL: int
    STRAT_TIMEFRAME: str
    STRAT_COOLDOWN_BARS: int
    ATR_LEN: int
    MIN_ATR: float

    RISK_PCT: float
    SL_R: float
    TP_R: float
    MAX_DD_DAY: float
    MAX_RISK_PER_TRADE: float

    @classmethod
    def load(cls) -> "Settings":
        return cls(
            MODE=_env_str("MODE", "demo"),
            TZ=_env_str("TZ", "Australia/Perth"),
            HEARTBEAT_SECONDS=_env_int("HEARTBEAT_SECONDS", 30),
            DECISION_SECONDS=_env_int("DECISION_SECONDS", 60),
            MAX_SILENCE_SECONDS=_env_int("MAX_SILENCE_SECONDS", 180),
            ERROR_BURST_THRESHOLD=_env_int("ERROR_BURST_THRESHOLD", 3),
            ALERT_EMAIL=_env_str("ALERT_EMAIL"),
            SMTP_HOST=_env_str("SMTP_HOST"),
            SMTP_PORT=_env_int("SMTP_PORT", 587),
            SMTP_USER=_env_str("SMTP_USER"),
            SMTP_PASS=_env_str("SMTP_PASS"),
            OANDA_API_KEY=_env_str("OANDA_API_KEY"),
            OANDA_ACCOUNT_ID=_env_str("OANDA_ACCOUNT_ID"),
            INSTRUMENT=_env_str("INSTRUMENT", "EUR_USD"),
            ORDER_SIZE=_env_float("ORDER_SIZE", 1000.0),
            STRAT_EMA_FAST=_env_int("STRAT_EMA_FAST", 10),
            STRAT_EMA_SLOW=_env_int("STRAT_EMA_SLOW", 20),
            STRAT_RSI_LEN=_env_int("STRAT_RSI_LEN", 14),
            STRAT_RSI_BUY=_env_int("STRAT_RSI_BUY", 55),
            STRAT_RSI_SELL=_env_int("STRAT_RSI_SELL", 45),
            STRAT_TIMEFRAME=_env_str("STRAT_TIMEFRAME", "M5"),
            STRAT_COOLDOWN_BARS=_env_int("STRAT_COOLDOWN_BARS", 2),
            ATR_LEN=_env_int("ATR_LEN", 14),
            MIN_ATR=_env_float("MIN_ATR", 0.00005),
            RISK_PCT=_env_float("RISK_PCT", 0.01),
            SL_R=_env_float("SL_R", 1.0),
            TP_R=_env_float("TP_R", 1.5),
            MAX_DD_DAY=_env_float("MAX_DD_DAY", 0.05),
            MAX_RISK_PER_TRADE=_env_float("MAX_RISK_PER_TRADE", 0.02),
        )


settings = Settings.load()



