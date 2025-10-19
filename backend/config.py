from functools import lru_cache
from typing import List, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class BaseConfig(BaseSettings):
    # Manage settings via a .env file

    # Configuration variables
    ENV_STATE: Optional[str] = None

    # Load environment variables from a .env file
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class GlobalConfig(BaseConfig):
    # Application settings
    APP_NAME: str = "Digit Recognizer API"
    APP_VERSION: str = "1.0.0"
    APP_DESCRIPTION: str = (
        "A simple API to predict handwritten digits using a CNN model."
    )

    # CORS settings
    CORS_ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:5500",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://127.0.0.1:5500",  # LiveServer alternative
    ]

    # API settings
    API_V1_PREFIX: str = "/api/v1"


class DevConfig(GlobalConfig):
    model_config = SettingsConfigDict(env_prefix="DEV_")


class ProdConfig(GlobalConfig):
    model_config = SettingsConfigDict(env_prefix="PROD_")


class TestConfig(GlobalConfig):
    model_config = SettingsConfigDict(env_prefix="TEST_")


# Configuration selector based on ENV_STATE
@lru_cache()
def get_config(env_state: str):
    configs = {
        "dev": DevConfig,
        "prod": ProdConfig,
        "test": TestConfig,
    }
    return configs[env_state]()


config = get_config(BaseConfig().ENV_STATE)
