from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional
import os


class ServiceConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    MODEL_SERVICE_HOST: str = "localhost"
    MODEL_SERVICE_PORT: int = 8000

    CORE_API_HOST: str = "localhost"
    CORE_API_PORT: int = 8001

    MULTIMEDIA_SERVICE_HOST: str = "localhost"
    MULTIMEDIA_SERVICE_PORT: int = 8002

    SEARCH_SERVICE_HOST: str = "localhost"
    SEARCH_SERVICE_PORT: int = 8003

    API_GATEWAY_HOST: str = "0.0.0.0"
    API_GATEWAY_PORT: int = 8080

    USE_MODEL_SERVICE: bool = True
    USE_API_GATEWAY: bool = True

    ENABLE_NSFW_DETECTION: bool = True

    CACHE_ENABLED: bool = True
    CACHE_TTL: int = 3600

    USE_REDIS: bool = False
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    CACHE_STRATEGY: str = "local_first"

    LOG_LEVEL: str = "INFO"

    DATABASE_MODE: str = "sqlite"
    SQLITE_URL: Optional[str] = None
    MYSQL_HOST: Optional[str] = None
    MYSQL_PORT: int = 3306
    MYSQL_USER: Optional[str] = None
    MYSQL_PASSWORD: Optional[str] = None
    MYSQL_DB: Optional[str] = None

    @property
    def MODEL_SERVICE_URL(self) -> str:
        return f"http://{self.MODEL_SERVICE_HOST}:{self.MODEL_SERVICE_PORT}"

    @property
    def CORE_API_URL(self) -> str:
        return f"http://{self.CORE_API_HOST}:{self.CORE_API_PORT}"

    @property
    def MULTIMEDIA_SERVICE_URL(self) -> str:
        return f"http://{self.MULTIMEDIA_SERVICE_HOST}:{self.MULTIMEDIA_SERVICE_PORT}"

    @property
    def SEARCH_SERVICE_URL(self) -> str:
        return f"http://{self.SEARCH_SERVICE_HOST}:{self.SEARCH_SERVICE_PORT}"

    @property
    def API_GATEWAY_URL(self) -> str:
        return f"http://{self.API_GATEWAY_HOST}:{self.API_GATEWAY_PORT}"

    def is_production(self) -> bool:
        return os.environ.get("ENVIRONMENT", "development").lower() == "production"


def get_service_config() -> ServiceConfig:
    return ServiceConfig()
