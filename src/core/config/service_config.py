from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator
from typing import Optional
import os

from .ports import coerce_port


class ServiceConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8001

    MODEL_SERVICE_HOST: str = "localhost"
    MODEL_SERVICE_PORT: int = 8000

    CORE_API_HOST: str = "localhost"
    CORE_API_PORT: int = 8001

    MULTIMEDIA_SERVICE_HOST: str = "localhost"
    MULTIMEDIA_SERVICE_PORT: int = 8002

    SEARCH_SERVICE_HOST: str = "localhost"
    SEARCH_SERVICE_PORT: int = 8003

    T2I_SERVICE_HOST: str = "localhost"
    T2I_SERVICE_PORT: int = 8100

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

    HF_CACHE_DIR: str = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "cache", "huggingface")
    KERAS_CACHE_DIR: str = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "cache", "keras")

    DATABASE_MODE: str = "sqlite"
    SQLITE_URL: Optional[str] = None
    MYSQL_HOST: Optional[str] = None
    MYSQL_PORT: int = 3306
    MYSQL_USER: Optional[str] = None
    MYSQL_PASSWORD: Optional[str] = None
    MYSQL_DB: Optional[str] = None

    # ==================== 性能优化配置 ====================
    # 推理批处理大小（默认 8，可根据 GPU 显存调整）
    INFERENCE_BATCH_SIZE: int = int(os.getenv("INFERENCE_BATCH_SIZE", "8"))
    # 强制设备: None=自动检测, "cuda"/"mps"/"cpu"=强制指定
    FORCE_DEVICE: Optional[str] = os.getenv("FORCE_DEVICE", None)
    # 关键点检测 worker 进程数
    KEYPOINT_WORKER_COUNT: int = int(os.getenv("KEYPOINT_WORKER_COUNT", "2"))
    # Uvicorn 并发连接数限制
    UVICORN_LIMIT_CONCURRENCY: int = int(os.getenv("UVICORN_LIMIT_CONCURRENCY", "64"))
    # 数据库连接池大小
    DB_POOL_SIZE: int = int(os.getenv("DB_POOL_SIZE", "5"))
    # 数据库连接池最大溢出
    DB_MAX_OVERFLOW: int = int(os.getenv("DB_MAX_OVERFLOW", "10"))

    # ==================== 基础设施降级 & 模型镜像配置 ====================
    # HuggingFace 镜像端点（国内加速）
    HF_ENDPOINT: str = os.getenv("HF_ENDPOINT", "https://hf-mirror.com")
    # OCR 超时时间（秒）
    OCR_TIMEOUT: int = int(os.getenv("OCR_TIMEOUT", "10"))
    # Redis 重连间隔（秒）
    REDIS_RECONNECT_INTERVAL: int = int(os.getenv("REDIS_RECONNECT_INTERVAL", "30"))

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
    def T2I_SERVICE_URL(self) -> str:
        return f"http://{self.T2I_SERVICE_HOST}:{self.T2I_SERVICE_PORT}"

    @property
    def API_GATEWAY_URL(self) -> str:
        return f"http://{self.API_GATEWAY_HOST}:{self.API_GATEWAY_PORT}"

    @field_validator(
        "API_PORT",
        "MODEL_SERVICE_PORT",
        "CORE_API_PORT",
        "MULTIMEDIA_SERVICE_PORT",
        "SEARCH_SERVICE_PORT",
        "T2I_SERVICE_PORT",
        "API_GATEWAY_PORT",
        "REDIS_PORT",
        "MYSQL_PORT",
        mode="before",
    )
    @classmethod
    def _coerce_port_fields(cls, v):
        # Kubernetes injects <SERVICE>_PORT as "tcp://<ip>:<port>"; tolerate it.
        if v is None:
            return v
        return coerce_port(v, 0)

    def is_production(self) -> bool:
        return os.environ.get("ENVIRONMENT", "development").lower() == "production"


def get_service_config() -> ServiceConfig:
    return ServiceConfig()
