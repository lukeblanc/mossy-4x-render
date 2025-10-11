import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Global configuration settings for the Mossy 4X bot.
    Reads environment variables, provides sane defaults.
    """

    # OANDA API configuration
    OANDA_API_KEY: str = os.getenv("OANDA_API_KEY", "")
    ACCOUNT_ID: str = os.getenv("ACCOUNT_ID", "")
    BASE_URL: str = os.getenv(
        "BASE_URL", "https://api-fxpractice.oanda.com/v3"
    )

    # Trading parameters
    MAX_RISK_PER_TRADE: float = float(os.getenv("MAX_RISK_PER_TRADE", "0.02"))

    # Run mode: "demo" or "live"
    MODE: str = os.getenv("MODE", "demo")

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
