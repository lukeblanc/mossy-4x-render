from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration for the Mossy 4X trading bot."""

    # Core runtime configuration
    MODE: str = Field("demo", description="Run mode for the bot: demo or live")
    TZ: str = Field("Australia/Perth", description="Local timezone for logging")
    HEARTBEAT_SECONDS: int = Field(30, description="Heartbeat interval in seconds")
    DECISION_SECONDS: int = Field(60, description="Decision engine interval in seconds")

    # Health / alerts
    MAX_SILENCE_SECONDS: int = Field(180, description="Alert threshold for missing heartbeats")
    ERROR_BURST_THRESHOLD: int = Field(3, description="Number of consecutive errors before alerting")
    ALERT_EMAIL: Optional[str] = Field(default=None, description="Destination email for alerts")
    SMTP_HOST: Optional[str] = Field(default=None, description="SMTP host for alert emails")
    SMTP_PORT: int = Field(587, description="SMTP port for alert emails")
    SMTP_USER: Optional[str] = Field(default=None, description="SMTP username")
    SMTP_PASS: Optional[str] = Field(default=None, description="SMTP password")

    # Trading configuration
    OANDA_API_KEY: Optional[str] = Field(default=None, description="OANDA API key")
    OANDA_ACCOUNT_ID: Optional[str] = Field(default=None, description="OANDA account identifier")
    INSTRUMENT: str = Field("EUR_USD", description="Default trading instrument")
    ORDER_SIZE: float = Field(1000.0, description="Default trade size")

    # Strategy parameters
    STRAT_EMA_FAST: int = Field(10, description="Fast EMA length")
    STRAT_EMA_SLOW: int = Field(20, description="Slow EMA length")
    STRAT_RSI_LEN: int = Field(14, description="RSI length")
    STRAT_RSI_BUY: int = Field(55, description="RSI threshold for buy signals")
    STRAT_RSI_SELL: int = Field(45, description="RSI threshold for sell signals")
    STRAT_TIMEFRAME: str = Field("M5", description="Primary strategy timeframe")
    STRAT_COOLDOWN_BARS: int = Field(2, description="Bars to wait before emitting a new signal")
    ATR_LEN: int = Field(14, description="ATR lookback for volatility checks")

    # Additional rule variables
    MAX_RISK_PER_TRADE: float = Field(0.02, description="Maximum risk per trade as a fraction of equity")
    DAILY_LOSS_CAP: float = Field(0.05, description="Daily loss cap as a fraction of equity")
    DRAWDOWN_CAP: float = Field(0.10, description="Maximum drawdown before pausing trading")
    NEWS_GUARD_MINUTES: int = Field(60, description="Minutes to avoid trading around news events")
    EXIT_LOGIC: str = Field("chandelier", description="Exit logic identifier")
    MAX_OPEN_TRADES: int = Field(3, description="Maximum number of open trades")
    ADX_FILTER: int = Field(25, description="Minimum ADX value to enable trades")
    MTF_ALIGN: bool = Field(True, description="Require multi-timeframe alignment")
    RULES_VERSION: str = Field("V1.6", description="Strategy rules version identifier")

    MIN_ATR: float = Field(0.00005, description="Minimum ATR required for trades")

    # Risk management
    RISK_PCT: float = Field(0.01, description="Fraction of capital risked per trade")
    SL_R: float = Field(1.0, description="Stop-loss multiple of risk")
    TP_R: float = Field(1.5, description="Take-profit multiple of risk")
    MAX_DD_DAY: float = Field(0.05, description="Maximum daily drawdown")

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )


settings = Settings()
