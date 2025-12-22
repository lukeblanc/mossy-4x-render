"""Application configuration for the Render worker."""

from __future__ import annotations

import os

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables."""

    # ------------------------------------------------------------------
    # OANDA connectivity
    # ------------------------------------------------------------------
    OANDA_API_KEY: str = Field(
        "",
        description="OANDA API key used for authenticated requests.",
        validation_alias=AliasChoices("OANDA_API_KEY", "OANDA_API_TOKEN"),
    )
    OANDA_ACCOUNT_ID: str = Field(
        "",
        description="OANDA account identifier.",
        validation_alias=AliasChoices("OANDA_ACCOUNT_ID", "ACCOUNT_ID"),
    )
    OANDA_ENV: str = Field(
        "practice",
        description="Target OANDA environment: practice or live.",
    )
    BASE_URL: str = Field(
        "https://api-fxpractice.oanda.com/v3",
        description="Base REST API URL for OANDA requests.",
    )
    MODE: str = Field(
        "demo",
        description="Run mode for the bot: demo, live, or simulation.",
    )

    # ------------------------------------------------------------------
    # Logging & scheduling
    # ------------------------------------------------------------------
    TZ: str = Field(
        "UTC",
        description="Timezone label shown in heartbeat logs.",
    )
    HEARTBEAT_SECONDS: int = Field(
        30,
        description="Seconds between heartbeat log entries.",
    )
    DECISION_SECONDS: int = Field(
        60,
        description="Seconds between strategy evaluation cycles.",
    )
    MAX_SILENCE_SECONDS: int = Field(
        180,
        description="Maximum silence in seconds before watchdog alerts.",
    )
    ERROR_BURST_THRESHOLD: int = Field(
        3,
        description="Errors within the rolling window that trigger alerts.",
    )

    # ------------------------------------------------------------------
    # Strategy inputs
    # ------------------------------------------------------------------
    INSTRUMENT: str = Field(
        "EUR_USD",
        description="Primary instrument traded by the worker.",
    )
    ORDER_SIZE: int = Field(
        1000,
        description="Default order size used for demo trades.",
    )
    STRAT_TIMEFRAME: str = Field(
        "M5",
        description="Granularity for fetched candles (OANDA notation).",
    )
    STRAT_EMA_FAST: int = Field(
        12,
        description="Fast EMA lookback length for the crossover strategy.",
    )
    STRAT_EMA_SLOW: int = Field(
        26,
        description="Slow EMA lookback length for the crossover strategy.",
    )
    STRAT_RSI_LEN: int = Field(
        14,
        description="RSI lookback length.",
    )
    STRAT_RSI_BUY: float = Field(
        52.0,
        description="RSI threshold required to issue a BUY signal.",
    )
    STRAT_RSI_SELL: float = Field(
        48.0,
        description="RSI threshold required to issue a SELL signal.",
    )
    ADX_FILTER: float = Field(
        20.0,
        description="Minimum ADX value required before enabling trade signals.",
    )
    ATR_LEN: int = Field(
        14,
        description="ATR lookback length.",
    )
    MIN_ATR: float = Field(
        0.00005,
        description="Minimum ATR required before issuing a trade signal.",
    )
    STRAT_COOLDOWN_BARS: int = Field(
        9,
        description="Bars to wait after a trade before considering a new one.",
    )

    # ------------------------------------------------------------------
    # Alerting
    # ------------------------------------------------------------------
    ALERT_EMAIL: str = Field(
        "",
        description="Email address to receive watchdog alerts.",
    )
    SMTP_HOST: str = Field(
        "",
        description="SMTP host used for watchdog alert emails.",
    )
    SMTP_PORT: int = Field(
        587,
        description="SMTP port used for watchdog alert emails.",
    )
    SMTP_USER: str = Field(
        "",
        description="SMTP username for authentication.",
    )
    SMTP_PASS: str = Field(
        "",
        description="SMTP password for authentication.",
    )

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Trading parameters
    MAX_RISK_PER_TRADE: float = float(os.getenv("MAX_RISK_PER_TRADE", "0.02"))

settings = Settings()
