import os
import json
import random
from typing import Optional, Dict, List, Tuple

from src.core.logging.global_logger import get_logger

logger = get_logger("model_version_service")


class ModelVersionService:
    """模型版本管理服务"""

    _instance: Optional["ModelVersionService"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "initialized") and self.initialized:
            return

        self.initialized = True

        # 模型版本配置文件
        self.config_file = os.path.join(
            os.path.dirname(__file__), "..", "..", "config", "model_versions.json"
        )

        # 确保配置目录存在
        config_dir = os.path.dirname(self.config_file)
        os.makedirs(config_dir, exist_ok=True)

        # 加载模型版本配置
        self.model_versions = self._load_config()

        # A/B测试配置
        self.ab_test_config = {
            "enabled": False,
            "test_models": [],
            "weights": [],
            "control_model": "default",
        }

        logger.info("模型版本管理服务初始化完成")

    def _load_config(self) -> Dict[str, Dict[str, any]]:
        """加载模型版本配置"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"加载模型版本配置失败: {e}")
                return {}
        return {}

    def _save_config(self):
        """保存模型版本配置"""
        try:
            with open(self.config_file, "w", encoding="utf-8") as f:
                json.dump(self.model_versions, f, ensure_ascii=False, indent=2)
            logger.info("模型版本配置保存成功")
        except Exception as e:
            logger.error(f"保存模型版本配置失败: {e}")

    def register_model(
        self, model_name: str, version: str, path: str, description: str = ""
    ) -> bool:
        """注册模型版本"""
        try:
            if model_name not in self.model_versions:
                self.model_versions[model_name] = {}

            self.model_versions[model_name][version] = {
                "path": path,
                "description": description,
                "created_at": os.path.getctime(path) if os.path.exists(path) else None,
            }

            self._save_config()
            logger.info(f"模型版本注册成功: {model_name} v{version}")
            return True
        except Exception as e:
            logger.error(f"注册模型版本失败: {e}")
            return False

    def get_model_versions(self, model_name: Optional[str] = None) -> Dict[str, Dict[str, any]]:
        """获取模型版本列表"""
        if model_name:
            return self.model_versions.get(model_name, {})
        return self.model_versions

    def get_model_path(self, model_name: str, version: str = "latest") -> Optional[str]:
        """获取模型路径"""
        if model_name not in self.model_versions:
            return None

        versions = self.model_versions[model_name]
        if not versions:
            return None

        if version == "latest":
            # 按创建时间排序，返回最新版本
            latest_version = sorted(
                versions.items(), key=lambda x: x[1].get("created_at", 0), reverse=True
            )[0][0]
            return versions[latest_version].get("path")

        return versions.get(version, {}).get("path")

    def enable_ab_test(
        self, test_models: List[Tuple[str, str]], weights: List[float], control_model: str
    ):
        """启用A/B测试"""
        self.ab_test_config = {
            "enabled": True,
            "test_models": test_models,  # 格式: [(model_name, version), ...]
            "weights": weights,  # 格式: [0.3, 0.7]，总和应为1.0
            "control_model": control_model,
        }
        logger.info(f"A/B测试已启用: {self.ab_test_config}")

    def disable_ab_test(self):
        """禁用A/B测试"""
        self.ab_test_config["enabled"] = False
        logger.info("A/B测试已禁用")

    def get_ab_test_config(self) -> Dict[str, any]:
        """获取A/B测试配置"""
        return self.ab_test_config

    def select_model_for_ab_test(self) -> Tuple[str, str]:
        """为A/B测试选择模型"""
        if not self.ab_test_config["enabled"] or not self.ab_test_config["test_models"]:
            # 如果A/B测试未启用，返回默认模型
            return self.ab_test_config["control_model"], "latest"

        # 根据权重随机选择模型
        test_models = self.ab_test_config["test_models"]
        weights = self.ab_test_config["weights"]

        # 归一化权重
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]

        # 随机选择
        selected = random.choices(test_models, weights=normalized_weights, k=1)[0]
        logger.info(f"A/B测试选择模型: {selected}")
        return selected

    def update_model_description(self, model_name: str, version: str, description: str) -> bool:
        """更新模型描述"""
        try:
            if model_name in self.model_versions and version in self.model_versions[model_name]:
                self.model_versions[model_name][version]["description"] = description
                self._save_config()
                logger.info(f"模型描述更新成功: {model_name} v{version}")
                return True
            return False
        except Exception as e:
            logger.error(f"更新模型描述失败: {e}")
            return False

    def delete_model_version(self, model_name: str, version: str) -> bool:
        """删除模型版本"""
        try:
            if model_name in self.model_versions and version in self.model_versions[model_name]:
                del self.model_versions[model_name][version]
                # 如果模型没有版本了，删除模型
                if not self.model_versions[model_name]:
                    del self.model_versions[model_name]
                self._save_config()
                logger.info(f"模型版本删除成功: {model_name} v{version}")
                return True
            return False
        except Exception as e:
            logger.error(f"删除模型版本失败: {e}")
            return False


# 全局模型版本服务实例
_model_version_service: Optional[ModelVersionService] = None


def get_model_version_service() -> ModelVersionService:
    """获取模型版本服务实例"""
    global _model_version_service
    if _model_version_service is None:
        _model_version_service = ModelVersionService()
    return _model_version_service


def init_model_version_service():
    """初始化模型版本服务"""
    global _model_version_service
    if _model_version_service is None:
        _model_version_service = ModelVersionService()
        logger.info("模型版本服务初始化完成")
    return _model_version_service


def register_model(model_name: str, version: str, path: str, description: str = "") -> bool:
    """注册模型版本"""
    return get_model_version_service().register_model(model_name, version, path, description)


def get_model_versions(model_name: Optional[str] = None) -> Dict[str, Dict[str, any]]:
    """获取模型版本列表"""
    return get_model_version_service().get_model_versions(model_name)


def get_model_path(model_name: str, version: str = "latest") -> Optional[str]:
    """获取模型路径"""
    return get_model_version_service().get_model_path(model_name, version)


def enable_ab_test(test_models: List[Tuple[str, str]], weights: List[float], control_model: str):
    """启用A/B测试"""
    return get_model_version_service().enable_ab_test(test_models, weights, control_model)


def disable_ab_test():
    """禁用A/B测试"""
    return get_model_version_service().disable_ab_test()


def get_ab_test_config() -> Dict[str, any]:
    """获取A/B测试配置"""
    return get_model_version_service().get_ab_test_config()


def select_model_for_ab_test() -> Tuple[str, str]:
    """为A/B测试选择模型"""
    return get_model_version_service().select_model_for_ab_test()


def update_model_description(model_name: str, version: str, description: str) -> bool:
    """更新模型描述"""
    return get_model_version_service().update_model_description(model_name, version, description)


def delete_model_version(model_name: str, version: str) -> bool:
    """删除模型版本"""
    return get_model_version_service().delete_model_version(model_name, version)
