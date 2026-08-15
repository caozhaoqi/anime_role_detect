#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一配置模块 — 项目唯一的配置入口

合并了原分散在 5 个模块中的配置：
  - src/config/config.py          → 项目路径 + ML 参数
  - src/utils/config_manager.py   → 数据采集 / 网络配置
  - src/utils/config_utils.py     → 便利访问函数
  - src/data_pipeline/utils/device_config.py → 设备检测
  - src/run/services_config.py    → 服务端口定义

用法:
    from src.config import config           # 旧式 Config 兼容
    from src.config import project_config   # 推荐: 统一配置实例
    from src.config import SERVICES         # 服务注册表
    from src.config import get_device       # 设备检测
"""

import json
import os
import platform
from pathlib import Path
from typing import Any, Dict, List, Optional

# ============================================================================
# 项目根目录
# ============================================================================
PROJECT_ROOT = Path(__file__).parent.parent.parent


# ============================================================================
# 目录路径
# ============================================================================
class ProjectPaths:
    """项目目录路径 — 统一管理所有文件系统路径"""

    BASE_DIR: Path = PROJECT_ROOT
    DATA_DIR: Path = BASE_DIR / "data"
    MODEL_DIR: Path = BASE_DIR / "models"
    LOG_DIR: Path = BASE_DIR / "logs"
    DOCS_DIR: Path = BASE_DIR / "docs"
    SCRIPTS_DIR: Path = BASE_DIR / "scripts"
    SRC_DIR: Path = BASE_DIR / "src"
    CONFIG_DIR: Path = BASE_DIR / "config"
    TESTS_DIR: Path = BASE_DIR / "tests"
    CACHE_DIR: Path = BASE_DIR / "cache"

    # 数据子目录
    TRAIN_DIR: Path = DATA_DIR / "training_dataset"
    VAL_DIR: Path = DATA_DIR / "final_dataset"
    TEST_DIR: Path = DATA_DIR / "test"

    # 模型子目录
    CHECKPOINT_DIR: Path = MODEL_DIR / "checkpoints"
    ONNX_DIR: Path = MODEL_DIR / "onnx"

    # 角色列表（legacy）
    CHARACTERS_DIR: Path = BASE_DIR / "auto_spider_img" / "characters"
    ANIME_SET_FILE: Path = BASE_DIR / "auto_spider_img" / "anime_set.txt"

    @classmethod
    def ensure_dirs(cls) -> None:
        """创建所有必要的目录"""
        for attr_name in dir(cls):
            if attr_name.endswith("_DIR"):
                path = getattr(cls, attr_name)
                if isinstance(path, Path):
                    path.mkdir(exist_ok=True, parents=True)

    @classmethod
    def get(cls, path_attr: str) -> str:
        """获取路径字符串"""
        return str(getattr(cls, path_attr))


# ============================================================================
# 图像与数据采集配置
# ============================================================================
class ImageConfig:
    """图像处理与数据采集参数"""

    # 图像质量阈值
    MIN_IMAGE_SIZE: int = int(os.getenv("ARD_MIN_IMAGE_SIZE", "200"))
    MAX_IMAGE_SIZE: int = int(os.getenv("ARD_MAX_IMAGE_SIZE", "2048"))
    MIN_ASPECT_RATIO: float = 0.3
    MAX_ASPECT_RATIO: float = 3.0

    # 数据采集
    MAX_IMAGES_PER_CHARACTER: int = 100
    MIN_IMAGES_PER_CHARACTER: int = 50


# ============================================================================
# 模型训练配置
# ============================================================================
class TrainingConfig:
    """模型训练超参数"""

    BATCH_SIZE: int = int(os.getenv("ARD_BATCH_SIZE", "32"))
    EPOCHS: int = int(os.getenv("ARD_EPOCHS", "100"))
    LEARNING_RATE: float = float(os.getenv("ARD_LEARNING_RATE", "1e-4"))
    VALIDATION_SPLIT: float = 0.2


# ============================================================================
# 推理配置
# ============================================================================
class InferenceConfig:
    """推理参数"""

    CONFIDENCE_THRESHOLD: float = float(os.getenv("ARD_CONFIDENCE_THRESHOLD", "0.7"))
    MAX_BATCH_SIZE: int = int(os.getenv("ARD_MAX_BATCH_SIZE", "32"))
    DEFAULT_INDEX_PATH: str = str(ProjectPaths.DATA_DIR / "index")


# ============================================================================
# 网络与采集配置（原 utils/config_manager.py）
# ============================================================================
class AppFeatureConfig:
    """应用特性开关与 UI 服务地址 — 统一由环境变量控制，供前后端共享语义"""

    # 监控面板 base URL（monitor_dashboard 服务地址；前端 Header 监控按钮跳转用）
    MONITOR_BASE_URL: str = os.getenv("ARD_MONITOR_BASE_URL", "http://localhost:9000")

    # 实时视频识别端点开关（默认关闭：该端点会阻塞 worker 且无消费方）
    ENABLE_REALTIME_VIDEO: bool = os.getenv("ARD_ENABLE_REALTIME_VIDEO", "false").lower() in (
        "1", "true", "yes", "on"
    )


class NetworkConfig:
    """网络请求与采集参数"""

    MAX_RETRIES: int = 3
    BACKOFF_FACTOR: float = 0.5
    TIMEOUT: int = 15
    DOWNLOAD_TIMEOUT: int = 30
    POOL_CONNECTIONS: int = 10
    POOL_MAXSIZE: int = 10


class CollectionConfig:
    """数据采集配置"""

    MAX_WORKERS: int = 5
    MAX_IMAGES_PER_CHARACTER: int = 500
    MIN_IMAGE_SIZE: int = 300
    RANKING_MODES: List[str] = ["daily", "weekly", "monthly"]
    SEARCH_URLS: List[str] = [
        "https://sd.vv50.de/search.php?word={}",
        "https://sd.vv50.de/illustration?word={}",
        "https://sd.vv50.de/ranking.php?word={}",
        "https://sd.vv50.de/bookmark_new_illust.php?word={}",
    ]


class StorageConfig:
    """存储配置"""

    OUTPUT_DIR: str = "data/sdv50_train"
    TEST_DIR: str = "data/test_sdv50"
    TEMP_DIR: str = "data/temp"
    FILE_EXTENSION: str = "jpg"
    QUALITY: int = 95


class ConcurrencyConfig:
    """并发控制"""

    MAX_WORKERS: int = 5
    DYNAMIC_ADJUSTMENT: bool = True
    MIN_WORKERS: int = 2
    MAX_WORKERS_LIMIT: int = 10


class DataSourceConfig:
    """数据源配置"""

    SDV50_BASE_URL: str = "https://sd.vv50.de"
    SDV50_RANKING_URL: str = "https://sd.vv50.de/ranking.php?mode={}&content={}"
    SDV50_ENABLED: bool = True
    BING_BASE_URL: str = "https://www.bing.com"
    BING_SEARCH_URL: str = "https://www.bing.com/images/search?q={}&count=50"
    BING_ENABLED: bool = True


# ============================================================================
# 服务注册表（原 src/run/services_config.py）
# ============================================================================
SERVICES: Dict[str, Dict[str, Any]] = {
    "model_service": {
        "name": "模型服务",
        "script": "services/model_service/app.py",
        "port": 8000,
        "health_path": "/api/health",
        "api_base": "/api/",
        "has_swagger": True,
        "description": "AI模型推理服务，支持角色分类和标签识别，被所有核心服务依赖",
        "enabled": True,
        "is_core": True,
    },
    "api_service": {
        "name": "主API服务",
        "script": "api/run_api.py",
        "port": 8001,
        "health_path": "/api/health",
        "api_base": "/api/",
        "has_swagger": False,
        "description": "主API服务，提供角色识别和综合接口",
        "enabled": True,
        "is_core": True,
    },
    "multimedia_service": {
        "name": "多媒体服务",
        "script": "services/multimedia_service/multimedia_service_app.py",
        "port": 8002,
        "health_path": "/api/health",
        "api_base": "/api/",
        "has_swagger": True,
        "description": "整合图像搜索和视频识别功能",
        "enabled": True,
        "is_core": True,
    },
    "search_service": {
        "name": "搜索服务",
        "script": "services/search_service/app_queue.py",
        "port": 8003,
        "health_path": "/api/health",
        "api_base": "/api/",
        "has_swagger": True,
        "description": "图像搜索队列服务，异步处理以图搜图请求",
        "enabled": True,
        "is_core": False,
    },
    "api_gateway": {
        "name": "API网关",
        "script": "services/api_gateway/app.py",
        "port": 8080,
        "health_path": "/api/health",
        "api_base": "/",
        "has_swagger": True,
        "description": "统一API网关，聚合所有后端服务，前端唯一入口",
        "enabled": True,
        "is_core": True,
    },
    "frontend": {
        "name": "前端页面",
        "script": "frontend/package.json",
        "port": 3000,
        "health_path": "/",
        "api_base": "/",
        "has_swagger": False,
        "description": "前端应用，用户交互界面",
        "enabled": True,
        "is_core": False,
    },
    "monitor_dashboard": {
        "name": "监控面板",
        "script": "run/monitor/monitor_dashboard.py",
        "port": 9000,
        "health_path": "/api/health",
        "api_base": "/",
        "has_swagger": False,
        "description": "服务状态监控、微服务拓扑图、API链路追踪仪表板",
        "enabled": True,
        "is_core": False,
    },
}

SERVICE_GROUPS: Dict[str, List[str]] = {
    "core": [k for k, v in SERVICES.items() if v.get("is_core")],
    "ai": ["model_service"],
    "search": ["search_service"],
    "multimedia": ["multimedia_service"],
    "gateway": ["api_gateway"],
    "frontend": ["frontend"],
    "monitoring": ["monitor_dashboard"],
    "all": list(SERVICES.keys()),
}


def get_service_by_name(name: str) -> Optional[Dict[str, Any]]:
    """根据名称获取服务配置"""
    return SERVICES.get(name)


def get_services_by_group(group_name: str) -> List[str]:
    """根据分组获取服务列表"""
    return SERVICE_GROUPS.get(group_name, [])


def list_all_services() -> List[str]:
    """列出所有服务"""
    return list(SERVICES.keys())


def get_service_port(name: str) -> Optional[int]:
    """获取服务端口"""
    service = SERVICES.get(name)
    return service["port"] if service else None


# ============================================================================
# 设备检测（原 src/data_pipeline/utils/device_config.py）
# ============================================================================
def configure_device() -> Optional[str]:
    """
    配置设备环境变量，必须在导入 PyTorch 之前调用。
    Mac 上禁用 CUDA 避免 mutex 错误。
    """
    if platform.system() == "Darwin":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        os.environ["FORCE_CPU"] = "1"
        return "cpu"

    if os.environ.get("CUDA_VISIBLE_DEVICES", "") == "":
        return "cpu"

    return None


def get_device(device: Optional[str] = None) -> str:
    """
    获取设备类型。

    Returns:
        'cpu', 'cuda', 或 'mps'
    """
    if device is not None:
        return device

    configured = configure_device()
    if configured:
        return configured

    return "cuda"


# 模块加载时立即配置（保持副作用兼容）
configure_device()


# ============================================================================
# 统一配置类（兼容旧式 Config 接口）
# ============================================================================
class _UnifiedConfig:
    """
    向后兼容的配置类 — 聚合所有配置域的属性。

    用法（兼容旧代码）:
        from src.config import config
        print(config.DATA_DIR)
        print(config.BATCH_SIZE)

    用法（推荐）:
        from src.config import project_config
        print(project_config.paths.DATA_DIR)
        print(project_config.training.BATCH_SIZE)
    """

    def __init__(self):
        self.paths = ProjectPaths
        self.image = ImageConfig
        self.training = TrainingConfig
        self.inference = InferenceConfig
        self.network = NetworkConfig
        self.collection = CollectionConfig
        self.storage = StorageConfig
        self.concurrency = ConcurrencyConfig
        self.datasource = DataSourceConfig
        self.app_features = AppFeatureConfig

    # --- 旧式属性代理（兼容 from src.config.config import config） ---

    @property
    def BASE_DIR(self) -> Path:
        return ProjectPaths.BASE_DIR

    @property
    def DATA_DIR(self) -> Path:
        return ProjectPaths.DATA_DIR

    @property
    def TRAIN_DIR(self) -> Path:
        return ProjectPaths.TRAIN_DIR

    @property
    def VAL_DIR(self) -> Path:
        return ProjectPaths.VAL_DIR

    @property
    def TEST_DIR(self) -> Path:
        return ProjectPaths.TEST_DIR

    @property
    def MODEL_DIR(self) -> Path:
        return ProjectPaths.MODEL_DIR

    @property
    def CHECKPOINT_DIR(self) -> Path:
        return ProjectPaths.CHECKPOINT_DIR

    @property
    def ONNX_DIR(self) -> Path:
        return ProjectPaths.ONNX_DIR

    @property
    def LOG_DIR(self) -> Path:
        return ProjectPaths.LOG_DIR

    @property
    def DOCS_DIR(self) -> Path:
        return ProjectPaths.DOCS_DIR

    @property
    def SCRIPTS_DIR(self) -> Path:
        return ProjectPaths.SCRIPTS_DIR

    @property
    def SRC_DIR(self) -> Path:
        return ProjectPaths.SRC_DIR

    @property
    def CONFIG_DIR(self) -> Path:
        return ProjectPaths.CONFIG_DIR

    @property
    def TESTS_DIR(self) -> Path:
        return ProjectPaths.TESTS_DIR

    @property
    def CHARACTERS_DIR(self) -> Path:
        return ProjectPaths.CHARACTERS_DIR

    @property
    def ANIME_SET_FILE(self) -> Path:
        return ProjectPaths.ANIME_SET_FILE

    @property
    def MIN_IMAGE_SIZE(self) -> int:
        return ImageConfig.MIN_IMAGE_SIZE

    @property
    def MAX_IMAGE_SIZE(self) -> int:
        return ImageConfig.MAX_IMAGE_SIZE

    @property
    def MIN_ASPECT_RATIO(self) -> float:
        return ImageConfig.MIN_ASPECT_RATIO

    @property
    def MAX_ASPECT_RATIO(self) -> float:
        return ImageConfig.MAX_ASPECT_RATIO

    @property
    def MAX_IMAGES_PER_CHARACTER(self) -> int:
        return ImageConfig.MAX_IMAGES_PER_CHARACTER

    @property
    def MIN_IMAGES_PER_CHARACTER(self) -> int:
        return ImageConfig.MIN_IMAGES_PER_CHARACTER

    @property
    def BATCH_SIZE(self) -> int:
        return TrainingConfig.BATCH_SIZE

    @property
    def EPOCHS(self) -> int:
        return TrainingConfig.EPOCHS

    @property
    def LEARNING_RATE(self) -> float:
        return TrainingConfig.LEARNING_RATE

    @property
    def CONFIDENCE_THRESHOLD(self) -> float:
        return InferenceConfig.CONFIDENCE_THRESHOLD

    @property
    def DEFAULT_INDEX_PATH(self) -> str:
        return InferenceConfig.DEFAULT_INDEX_PATH

    # 兼容旧方法
    def get_path(self, path: Path) -> str:
        return str(path)

    def get_character_file(self, series_name: str) -> str:
        return str(ProjectPaths.CHARACTERS_DIR / f"{series_name}.txt")

    def _create_directories(self) -> None:
        ProjectPaths.ensure_dirs()


# ============================================================================
# 全局单例
# ============================================================================
# 推荐使用
project_config = _UnifiedConfig()

# 旧式兼容 — from src.config.config import config
config = project_config

# 旧式兼容 — from src.utils.config_manager import config_manager
class _LegacyConfigManager:
    """兼容旧的 utils/config_manager.py ConfigManager 接口"""

    def __init__(self):
        self.config = {
            "network": {
                "max_retries": NetworkConfig.MAX_RETRIES,
                "backoff_factor": NetworkConfig.BACKOFF_FACTOR,
                "timeout": NetworkConfig.TIMEOUT,
                "download_timeout": NetworkConfig.DOWNLOAD_TIMEOUT,
                "pool_connections": NetworkConfig.POOL_CONNECTIONS,
                "pool_maxsize": NetworkConfig.POOL_MAXSIZE,
            },
            "storage": {
                "output_dir": StorageConfig.OUTPUT_DIR,
                "test_dir": StorageConfig.TEST_DIR,
                "temp_dir": StorageConfig.TEMP_DIR,
                "file_extension": StorageConfig.FILE_EXTENSION,
                "quality": StorageConfig.QUALITY,
            },
            "collection": {
                "max_workers": CollectionConfig.MAX_WORKERS,
                "max_images_per_character": CollectionConfig.MAX_IMAGES_PER_CHARACTER,
                "min_image_size": CollectionConfig.MIN_IMAGE_SIZE,
                "ranking_modes": CollectionConfig.RANKING_MODES,
                "search_urls": CollectionConfig.SEARCH_URLS,
            },
            "data_sources": {
                "sdv50": {
                    "base_url": DataSourceConfig.SDV50_BASE_URL,
                    "ranking_url": DataSourceConfig.SDV50_RANKING_URL,
                    "enabled": DataSourceConfig.SDV50_ENABLED,
                },
                "bing": {
                    "base_url": DataSourceConfig.BING_BASE_URL,
                    "search_url": DataSourceConfig.BING_SEARCH_URL,
                    "enabled": DataSourceConfig.BING_ENABLED,
                },
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "log_dir": "logs",
            },
            "concurrency": {
                "max_workers": ConcurrencyConfig.MAX_WORKERS,
                "dynamic_adjustment": ConcurrencyConfig.DYNAMIC_ADJUSTMENT,
                "min_workers": ConcurrencyConfig.MIN_WORKERS,
                "max_workers_limit": ConcurrencyConfig.MAX_WORKERS_LIMIT,
            },
        }

    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split(".")
        value = self.config
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def set(self, key: str, value: Any) -> None:
        keys = key.split(".")
        config = self.config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value

    def get_all(self) -> Dict:
        return self.config


config_manager = _LegacyConfigManager()

# 确保目录存在
ProjectPaths.ensure_dirs()
