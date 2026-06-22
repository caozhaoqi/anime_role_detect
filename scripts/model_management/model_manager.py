#!/usr/bin/env python3
"""模型管理脚本 - 参考MLflow/Hugging Face的成熟模型管理功能"""
import os
import json
import torch
import torch.nn as nn
from torchvision import models
from pathlib import Path
from typing import Dict, Optional, List, Any, Union
from datetime import datetime
from dataclasses import dataclass, asdict, field
from enum import Enum
import hashlib
import shutil
import logging
from contextlib import contextmanager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelStatus(Enum):
    """模型状态"""
    TRAINING = "training"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"
    DEPLOYED = "deployed"
    ARCHIVED = "archived"


class ModelFormat(Enum):
    """模型格式"""
    PYTORCH = "pytorch"
    ONNX = "onnx"
    TENSORRT = "tensorrt"
    TFLITE = "tflite"


@dataclass
class ModelMetadata:
    """模型元数据"""
    name: str
    version: str
    architecture: str
    num_classes: int
    input_size: int
    created_at: str
    updated_at: str
    status: str
    metrics: Dict[str, float] = field(default_factory=dict)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    dataset_info: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    description: str = ""
    author: str = ""
    file_size_mb: float = 0.0
    hash: str = ""


@dataclass
class ModelConfig:
    """模型配置"""
    name: str
    architecture: str
    num_classes: int
    input_size: int = 224
    pretrained: bool = False
    version: str = "1.0.0"


class ModelRegistry:
    """模型注册表"""

    _models: Dict[str, ModelConfig] = {}

    @classmethod
    def register(cls, config: ModelConfig) -> None:
        cls._models[config.name] = config
        logger.info(f"注册模型: {config.name} v{config.version}")

    @classmethod
    def get(cls, name: str) -> Optional[ModelConfig]:
        return cls._models.get(name)

    @classmethod
    def list_models(cls) -> List[str]:
        return list(cls._models.keys())

    @classmethod
    def unregister(cls, name: str) -> bool:
        if name in cls._models:
            del cls._models[name]
            logger.info(f"注销模型: {name}")
            return True
        return False


class ModelFactory:
    """模型工厂"""

    @staticmethod
    def create_model(config: ModelConfig) -> nn.Module:
        """创建模型"""
        if config.architecture == "mobilenet_v2":
            model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT if config.pretrained else None)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, config.num_classes)
        elif config.architecture == "efficientnet_b3":
            model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT if config.pretrained else None)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, config.num_classes)
        elif config.architecture == "resnet50":
            model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if config.pretrained else None)
            model.fc = nn.Linear(model.fc.in_features, config.num_classes)
        elif config.architecture == "vit_b_16":
            model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT if config.pretrained else None)
            model.heads.head = nn.Linear(model.heads.head.in_features, config.num_classes)
        elif config.architecture == "swin_v2_b":
            model = models.swin_v2_b(weights=models.Swin_V2_B_Weights.DEFAULT if config.pretrained else None)
            model.head = nn.Linear(model.head.in_features, config.num_classes)
        else:
            raise ValueError(f"不支持的模型架构: {config.architecture}")

        return model


class ModelManager:
    """模型管理器 - 参考MLflow/Hugging Face的成熟功能"""

    def __init__(self, model_dir: Path):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir = self.model_dir / "metadata"
        self.metadata_dir.mkdir(exist_ok=True)
        self.checkpoints_dir = self.model_dir / "checkpoints"
        self.checkpoints_dir.mkdir(exist_ok=True)
        self.deployed_dir = self.model_dir / "deployed"
        self.deployed_dir.mkdir(exist_ok=True)

    def _calculate_hash(self, file_path: Path) -> str:
        """计算文件哈希值"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def _generate_metadata(
        self,
        model_name: str,
        config: ModelConfig,
        metrics: Dict[str, float],
        hyperparameters: Dict[str, Any],
        dataset_info: Dict[str, Any],
        tags: List[str],
        description: str,
        author: str,
    ) -> ModelMetadata:
        """生成模型元数据"""
        now = datetime.now().isoformat()
        model_path = self.checkpoints_dir / f"{model_name}.pth"

        file_size_mb = 0.0
        file_hash = ""
        if model_path.exists():
            file_size_mb = model_path.stat().st_size / (1024 * 1024)
            file_hash = self._calculate_hash(model_path)

        return ModelMetadata(
            name=model_name,
            version=config.version,
            architecture=config.architecture,
            num_classes=config.num_classes,
            input_size=config.input_size,
            created_at=now,
            updated_at=now,
            status=ModelStatus.COMPLETED.value,
            metrics=metrics,
            hyperparameters=hyperparameters,
            dataset_info=dataset_info,
            tags=tags,
            description=description,
            author=author,
            file_size_mb=file_size_mb,
            hash=file_hash,
        )

    def save_model(
        self,
        model: nn.Module,
        model_name: str,
        config: ModelConfig,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        dataset_info: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        description: str = "",
        author: str = "",
        is_best: bool = False,
    ) -> Path:
        """保存模型"""
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "config": {
                "name": config.name,
                "architecture": config.architecture,
                "num_classes": config.num_classes,
                "input_size": config.input_size,
                "version": config.version,
            },
        }

        if optimizer:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        if metrics:
            checkpoint["metrics"] = metrics

        model_path = self.checkpoints_dir / f"{model_name}.pth"
        torch.save(checkpoint, model_path)

        metadata = self._generate_metadata(
            model_name=model_name,
            config=config,
            metrics=metrics or {},
            hyperparameters=hyperparameters or {},
            dataset_info=dataset_info or {},
            tags=tags or [],
            description=description,
            author=author,
        )

        metadata_path = self.metadata_dir / f"{model_name}_metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(asdict(metadata), f, ensure_ascii=False, indent=2)

        if is_best:
            best_path = self.checkpoints_dir / f"{model_name}_best.pth"
            shutil.copy(model_path, best_path)
            best_metadata_path = self.metadata_dir / f"{model_name}_best_metadata.json"
            shutil.copy(metadata_path, best_metadata_path)
            logger.info(f"保存最佳模型: {best_path}")

        logger.info(f"模型已保存: {model_path}")
        logger.info(f"元数据已保存: {metadata_path}")

        return model_path

    def load_model(
        self,
        model_name: str,
        config: ModelConfig,
        device: torch.device,
        optimizer: Optional[torch.optim.Optimizer] = None,
        load_best: bool = False,
    ) -> nn.Module:
        """加载模型"""
        if load_best:
            model_path = self.checkpoints_dir / f"{model_name}_best.pth"
        else:
            model_path = self.checkpoints_dir / f"{model_name}.pth"

        if not model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        checkpoint = torch.load(model_path, map_location=device, weights_only=False)

        model = ModelFactory.create_model(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)

        if optimizer and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        logger.info(f"模型已加载: {model_path}")
        logger.info(f"  Epoch: {checkpoint.get('epoch', 0)}")
        logger.info(f"  Metrics: {checkpoint.get('metrics', {})}")

        return model

    def list_models(self) -> List[Dict[str, Any]]:
        """列出所有模型"""
        models_list = []
        for metadata_file in sorted(self.metadata_dir.glob("*_metadata.json")):
            if "_best" in metadata_file.name:
                continue

            model_name = metadata_file.stem.replace("_metadata", "")

            with open(metadata_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            models_list.append(metadata)

        return models_list

    def get_model_info(self, model_name: str) -> Optional[ModelMetadata]:
        """获取模型信息"""
        metadata_path = self.metadata_dir / f"{model_name}_metadata.json"

        if not metadata_path.exists():
            return None

        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata_dict = json.load(f)

        return ModelMetadata(**metadata_dict)

    def delete_model(self, model_name: str, delete_best: bool = False) -> bool:
        """删除模型"""
        deleted = False

        model_path = self.checkpoints_dir / f"{model_name}.pth"
        if model_path.exists():
            model_path.unlink()
            logger.info(f"模型已删除: {model_path}")
            deleted = True

        metadata_path = self.metadata_dir / f"{model_name}_metadata.json"
        if metadata_path.exists():
            metadata_path.unlink()
            logger.info(f"元数据已删除: {metadata_path}")
            deleted = True

        if delete_best:
            best_path = self.checkpoints_dir / f"{model_name}_best.pth"
            if best_path.exists():
                best_path.unlink()
                logger.info(f"最佳模型已删除: {best_path}")
                deleted = True

            best_metadata_path = self.metadata_dir / f"{model_name}_best_metadata.json"
            if best_metadata_path.exists():
                best_metadata_path.unlink()
                logger.info(f"最佳元数据已删除: {best_metadata_path}")
                deleted = True

        return deleted

    def archive_model(self, model_name: str) -> bool:
        """归档模型"""
        archive_dir = self.model_dir / "archived"
        archive_dir.mkdir(exist_ok=True)

        model_path = self.checkpoints_dir / f"{model_name}.pth"
        metadata_path = self.metadata_dir / f"{model_name}_metadata.json"

        if model_path.exists():
            shutil.move(model_path, archive_dir / model_path.name)
            logger.info(f"模型已归档: {model_path}")

        if metadata_path.exists():
            shutil.move(metadata_path, archive_dir / metadata_path.name)
            logger.info(f"元数据已归档: {metadata_path}")

            metadata = self.get_model_info(model_name)
            if metadata:
                metadata.status = ModelStatus.ARCHIVED.value
                metadata.updated_at = datetime.now().isoformat()

                new_metadata_path = archive_dir / metadata_path.name
                with open(new_metadata_path, "w", encoding="utf-8") as f:
                    json.dump(asdict(metadata), f, ensure_ascii=False, indent=2)

        return True

    def deploy_model(
        self,
        model_name: str,
        format: ModelFormat = ModelFormat.PYTORCH,
        optimize: bool = False,
    ) -> Path:
        """部署模型"""
        model_path = self.checkpoints_dir / f"{model_name}.pth"
        metadata = self.get_model_info(model_name)

        if not model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        if format == ModelFormat.PYTORCH:
            deployed_path = self.deployed_dir / f"{model_name}_deployed.pth"
            shutil.copy(model_path, deployed_path)

        elif format == ModelFormat.ONNX:
            deployed_path = self._export_to_onnx(model_name, optimize)

        else:
            raise ValueError(f"不支持的格式: {format}")

        if metadata:
            metadata.status = ModelStatus.DEPLOYED.value
            metadata.updated_at = datetime.now().isoformat()

            metadata_path = self.metadata_dir / f"{model_name}_metadata.json"
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(asdict(metadata), f, ensure_ascii=False, indent=2)

        logger.info(f"模型已部署: {deployed_path}")
        return deployed_path

    def _export_to_onnx(self, model_name: str, optimize: bool = False) -> Path:
        """导出为ONNX格式"""
        metadata = self.get_model_info(model_name)
        if not metadata:
            raise ValueError(f"找不到模型元数据: {model_name}")

        config = ModelConfig(
            name=metadata.name,
            architecture=metadata.architecture,
            num_classes=metadata.num_classes,
            input_size=metadata.input_size,
        )

        model = ModelFactory.create_model(config)
        self.load_model(model_name, config, torch.device("cpu"))

        dummy_input = torch.randn(1, 3, metadata.input_size, metadata.input_size)

        onnx_path = self.deployed_dir / f"{model_name}.onnx"
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=optimize,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "output": {0: "batch_size"},
            },
        )

        logger.info(f"模型已导出为ONNX: {onnx_path}")
        return onnx_path

    def compare_models(self, model_names: List[str]) -> Dict[str, Any]:
        """比较多个模型的性能"""
        comparison = {
            "models": [],
            "best_accuracy": {"model": "", "value": 0.0},
            "best_loss": {"model": "", "value": float("inf")},
        }

        for model_name in model_names:
            metadata = self.get_model_info(model_name)
            if metadata:
                model_info = {
                    "name": metadata.name,
                    "version": metadata.version,
                    "architecture": metadata.architecture,
                    "metrics": metadata.metrics,
                    "file_size_mb": metadata.file_size_mb,
                }
                comparison["models"].append(model_info)

                if "accuracy" in metadata.metrics:
                    if metadata.metrics["accuracy"] > comparison["best_accuracy"]["value"]:
                        comparison["best_accuracy"] = {
                            "model": model_name,
                            "value": metadata.metrics["accuracy"],
                        }

                if "loss" in metadata.metrics:
                    if metadata.metrics["loss"] < comparison["best_loss"]["value"]:
                        comparison["best_loss"] = {
                            "model": model_name,
                            "value": metadata.metrics["loss"],
                        }

        return comparison

    def search_models(
        self,
        architecture: Optional[str] = None,
        min_accuracy: Optional[float] = None,
        tags: Optional[List[str]] = None,
        status: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """搜索模型"""
        all_models = self.list_models()
        filtered = []

        for model in all_models:
            if architecture and model.get("architecture") != architecture:
                continue

            if min_accuracy and model.get("metrics", {}).get("accuracy", 0) < min_accuracy:
                continue

            if tags and not any(tag in model.get("tags", []) for tag in tags):
                continue

            if status and model.get("status") != status:
                continue

            filtered.append(model)

        return filtered

    @contextmanager
    def model_context(self, model_name: str, config: ModelConfig, device: torch.device):
        """模型上下文管理器"""
        model = self.load_model(model_name, config, device)
        try:
            yield model
        finally:
            del model
            torch.cuda.empty_cache()


def init_default_models(num_classes: int) -> None:
    """初始化默认模型"""
    default_models = [
        ModelConfig("mobilenet_v2", "mobilenet_v2", num_classes, version="1.0.0"),
        ModelConfig("mobilenet_v2_pretrained", "mobilenet_v2", num_classes, pretrained=True, version="1.0.0"),
        ModelConfig("efficientnet_b3", "efficientnet_b3", num_classes, version="1.0.0"),
        ModelConfig("efficientnet_b3_pretrained", "efficientnet_b3", num_classes, pretrained=True, version="1.0.0"),
        ModelConfig("resnet50", "resnet50", num_classes, version="1.0.0"),
        ModelConfig("resnet50_pretrained", "resnet50", num_classes, pretrained=True, version="1.0.0"),
        ModelConfig("vit_b_16", "vit_b_16", num_classes, version="1.0.0"),
        ModelConfig("vit_b_16_pretrained", "vit_b_16", num_classes, pretrained=True, version="1.0.0"),
    ]

    for config in default_models:
        ModelRegistry.register(config)


def main():
    project_root = Path(__file__).parent.parent.parent
    model_dir = project_root / "models"

    manager = ModelManager(model_dir)

    logger.info("=" * 60)
    logger.info("模型管理器 - 成熟功能演示")
    logger.info("=" * 60)

    logger.info("\n已注册的模型:")
    for model_name in ModelRegistry.list_models():
        config = ModelRegistry.get(model_name)
        logger.info(f"  - {model_name}: {config.architecture} ({config.num_classes} 类)")

    logger.info(f"\n已保存的模型:")
    models_list = manager.list_models()
    for model in models_list:
        logger.info(f"  - {model['name']} v{model['version']}")
        logger.info(f"    架构: {model['architecture']}")
        logger.info(f"    状态: {model['status']}")
        logger.info(f"    大小: {model['file_size_mb']:.2f} MB")
        if model['metrics']:
            logger.info(f"    指标: {model['metrics']}")
        if model['tags']:
            logger.info(f"    标签: {', '.join(model['tags'])}")

    if models_list:
        logger.info(f"\n模型对比:")
        model_names = [model['name'] for model in models_list[:3]]
        comparison = manager.compare_models(model_names)
        logger.info(f"  最佳准确率: {comparison['best_accuracy']}")
        logger.info(f"  最佳损失: {comparison['best_loss']}")


if __name__ == "__main__":
    main()