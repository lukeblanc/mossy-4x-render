import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    OANDA_API_KEY: str = os.getenv("OANDA_API_KEY", "")
    ACCOUNT_ID: str = os.getenv("ACCOUNT_ID", "")
    BASE_URL: str = os.getenv(
        "BASE_URL", "https://api-fxpractice.oanda.com/v3"
    )

    MAX_RISK_PER_TRADE: float = float(
        os.getenv("MAX_RISK_PER_TRADE", "0.02")
    )


settings = Settings()
