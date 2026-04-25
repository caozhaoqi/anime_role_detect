import os
from typing import Optional

class ServiceConfig:
    """服务配置中心 - 统一管理所有服务的配置"""

    _instance: Optional['ServiceConfig'] = None
    _initialized: bool = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True

        self.API_HOST = os.environ.get('API_HOST', '0.0.0.0')
        self.API_PORT = int(os.environ.get('API_PORT', '8000'))

        self.MODEL_SERVICE_HOST = os.environ.get('MODEL_SERVICE_HOST', 'localhost')
        self.MODEL_SERVICE_PORT = int(os.environ.get('MODEL_SERVICE_PORT', '8888'))
        self.MODEL_SERVICE_URL = f"http://{self.MODEL_SERVICE_HOST}:{self.MODEL_SERVICE_PORT}"

        self.API_GATEWAY_HOST = os.environ.get('API_GATEWAY_HOST', '0.0.0.0')
        self.API_GATEWAY_PORT = int(os.environ.get('API_GATEWAY_PORT', '8000'))
        self.API_GATEWAY_URL = f"http://{self.API_GATEWAY_HOST}:{self.API_GATEWAY_PORT}"

        self.USE_MODEL_SERVICE = os.environ.get('USE_MODEL_SERVICE', 'true').lower() == 'true'
        self.USE_API_GATEWAY = os.environ.get('USE_API_GATEWAY', 'true').lower() == 'true'

        self.ENABLE_NSFW_DETECTION = os.environ.get('ENABLE_NSFW_DETECTION', 'true').lower() == 'true'

        self.CACHE_ENABLED = os.environ.get('CACHE_ENABLED', 'true').lower() == 'true'
        self.CACHE_TTL = int(os.environ.get('CACHE_TTL', '3600'))

        self.USE_REDIS = os.environ.get('USE_REDIS', 'false').lower() == 'true'
        self.REDIS_HOST = os.environ.get('REDIS_HOST', 'localhost')
        self.REDIS_PORT = int(os.environ.get('REDIS_PORT', '6379'))
        self.REDIS_DB = int(os.environ.get('REDIS_DB', '0'))
        self.CACHE_STRATEGY = os.environ.get('CACHE_STRATEGY', 'local_first')

        self.MODEL_CACHE_DIR = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            'models'
        )

        self.HF_CACHE_DIR = os.environ.get(
            'HF_HOME',
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'huggingface_cache')
        )
        self.KERAS_CACHE_DIR = os.environ.get(
            'KERAS_HOME',
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'keras_cache')
        )

        self.LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
        self.LOG_DIR = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
            'logs'
        )

    def get_model_service_url(self, endpoint: str = "") -> str:
        """获取模型服务的完整URL"""
        if endpoint:
            return f"{self.MODEL_SERVICE_URL}{endpoint}"
        return self.MODEL_SERVICE_URL

    def is_production(self) -> bool:
        """检查是否为生产环境"""
        return os.environ.get('ENVIRONMENT', 'development').lower() == 'production'

    def reload(self):
        """重新加载配置"""
        self._initialized = False
        self.__init__()


_service_config_instance: Optional[ServiceConfig] = None


def get_service_config() -> ServiceConfig:
    """获取服务配置单例"""
    global _service_config_instance
    if _service_config_instance is None:
        _service_config_instance = ServiceConfig()
    return _service_config_instance