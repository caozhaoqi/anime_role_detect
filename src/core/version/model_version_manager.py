#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
版本控制模块 - 完善超参数记录
为 v3 版本对比（如 CutMix/MixUp 数据增强）做准备
"""

import os
import sys
import json
from datetime import datetime
from typing import Dict, List, Optional

# 添加项目根目录
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.insert(0, project_root)


class ModelVersionManager:
    """模型版本管理器"""

    VERSION_FILE = "model_versions.json"

    def __init__(self, model_dir: str = None):
        """
        初始化版本管理器

        Args:
            model_dir: 模型目录路径
        """
        if model_dir is None:
            MODEL_NAME = "efficientnet_b3_loli_optimized_v2_20260529_133654"
            model_dir = os.path.join(project_root, "models", MODEL_NAME)

        self.model_dir = model_dir
        self.versions_file = os.path.join(model_dir, self.VERSION_FILE)
        self.versions = self._load_versions()

    def _load_versions(self) -> Dict:
        """加载版本记录"""
        if os.path.exists(self.versions_file):
            with open(self.versions_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {
            "model_name": os.path.basename(self.model_dir),
            "created_at": datetime.now().isoformat(),
            "versions": [],
        }

    def _save_versions(self):
        """保存版本记录"""
        with open(self.versions_file, "w", encoding="utf-8") as f:
            json.dump(self.versions, f, indent=2, ensure_ascii=False)

    def register_version(
        self,
        version_name: str,
        training_config: Dict,
        metrics: Dict,
        data_augmentation: List[str] = None,
        notes: str = "",
    ) -> str:
        """
        注册新版本

        Args:
            version_name: 版本名称 (如 v2, v3)
            training_config: 训练配置
            metrics: 评估指标
            data_augmentation: 数据增强方法
            notes: 备注

        Returns:
            版本 ID
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_id = f"{version_name}_{timestamp}"

        version_info = {
            "id": version_id,
            "name": version_name,
            "registered_at": datetime.now().isoformat(),
            "training_config": training_config,
            "metrics": metrics,
            "data_augmentation": data_augmentation or [],
            "notes": notes,
        }

        self.versions["versions"].append(version_info)
        self._save_versions()

        print(f"✅ 版本已注册: {version_id}")
        return version_id

    def get_latest_version(self) -> Optional[Dict]:
        """获取最新版本"""
        if not self.versions["versions"]:
            return None
        return self.versions["versions"][-1]

    def get_version(self, version_id: str) -> Optional[Dict]:
        """获取指定版本"""
        for v in self.versions["versions"]:
            if v["id"] == version_id:
                return v
        return None

    def compare_versions(self, version_id_1: str, version_id_2: str) -> Dict:
        """
        对比两个版本

        Args:
            version_id_1: 版本1 ID
            version_id_2: 版本2 ID

        Returns:
            对比结果
        """
        v1 = self.get_version(version_id_1)
        v2 = self.get_version(version_id_2)

        if not v1 or not v2:
            return {"error": "Version not found"}

        comparison = {
            "version_1": v1["id"],
            "version_2": v2["id"],
            "training_config_diff": self._diff_configs(
                v1.get("training_config", {}), v2.get("training_config", {})
            ),
            "metrics_comparison": {
                "metric": [],
                "version_1_value": [],
                "version_2_value": [],
                "difference": [],
            },
            "data_augmentation_diff": {"only_in_v1": [], "only_in_v2": [], "common": []},
        }

        # 对比数据增强
        aug1 = set(v1.get("data_augmentation", []))
        aug2 = set(v2.get("data_augmentation", []))
        comparison["data_augmentation_diff"]["only_in_v1"] = list(aug1 - aug2)
        comparison["data_augmentation_diff"]["only_in_v2"] = list(aug2 - aug1)
        comparison["data_augmentation_diff"]["common"] = list(aug1 & aug2)

        # 对比指标
        metrics1 = v1.get("metrics", {})
        metrics2 = v2.get("metrics", {})

        all_metrics = set(metrics1.keys()) | set(metrics2.keys())
        for metric in all_metrics:
            val1 = metrics1.get(metric, 0)
            val2 = metrics2.get(metric, 0)
            comparison["metrics_comparison"]["metric"].append(metric)
            comparison["metrics_comparison"]["version_1_value"].append(val1)
            comparison["metrics_comparison"]["version_2_value"].append(val2)
            comparison["metrics_comparison"]["difference"].append(val2 - val1)

        return comparison

    def _diff_configs(self, config1: Dict, config2: Dict) -> Dict:
        """对比配置差异"""
        all_keys = set(config1.keys()) | set(config2.keys())
        diff = {}

        for key in all_keys:
            val1 = config1.get(key, "N/A")
            val2 = config2.get(key, "N/A")
            if val1 != val2:
                diff[key] = {"v1": val1, "v2": val2}

        return diff

    def export_comparison_report(
        self, version_id_1: str, version_id_2: str, output_path: str = None
    ) -> str:
        """
        导出版本对比报告

        Args:
            version_id_1: 版本1 ID
            version_id_2: 版本2 ID
            output_path: 输出路径

        Returns:
            报告文件路径
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(
                project_root, "reports", f"version_comparison_{timestamp}.json"
            )

        comparison = self.compare_versions(version_id_1, version_id_2)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(comparison, f, indent=2, ensure_ascii=False)

        print(f"✅ 对比报告已导出: {output_path}")
        return output_path

    def list_versions(self) -> List[Dict]:
        """列出所有版本"""
        return self.versions["versions"]


def register_current_model_version():
    """注册当前模型的版本信息"""
    MODEL_NAME = "efficientnet_b3_loli_optimized_v2_20260529_133654"
    MODEL_DIR = os.path.join(project_root, "models", MODEL_NAME)

    # 读取训练结果
    training_results_path = os.path.join(MODEL_DIR, "training_results.json")
    benchmark_results_path = os.path.join(MODEL_DIR, "benchmark_results.json")

    with open(training_results_path, "r") as f:
        training_results = json.load(f)

    with open(benchmark_results_path, "r") as f:
        benchmark_results = json.load(f)

    # 创建版本管理器
    manager = ModelVersionManager(MODEL_DIR)

    # 注册 v2 版本
    training_config = {
        "model_name": training_results.get("model_name"),
        "num_classes": training_results.get("num_classes"),
        "image_size": training_results.get("image_size"),
        "batch_size": training_results.get("batch_size"),
        "learning_rate": training_results.get("learning_rate"),
        "epochs": training_results.get("epochs"),
        "weight_decay": training_results.get("weight_decay"),
        "label_smoothing": training_results.get("label_smoothing"),
        "train_samples": training_results.get("train_samples"),
        "val_samples": training_results.get("val_samples"),
        "timestamp": training_results.get("timestamp"),
    }

    metrics = {
        "best_accuracy": training_results.get("best_accuracy"),
        "top_1_accuracy": benchmark_results.get("top_1_accuracy"),
        "top_3_accuracy": benchmark_results.get("top_3_accuracy"),
        "top_5_accuracy": benchmark_results.get("top_5_accuracy"),
        "fps": benchmark_results.get("fps"),
    }

    # 检查是否已注册
    existing = manager.get_latest_version()
    if existing and existing.get("name") == "v2":
        print("⚠️ v2 版本已注册，跳过")
    else:
        version_id = manager.register_version(
            version_name="v2",
            training_config=training_config,
            metrics=metrics,
            data_augmentation=["RandomHorizontalFlip", "ColorJitter", "RandomCrop"],
            notes="EfficientNet-B3 优化版本，使用标准数据增强",
        )
        print(f"✅ v2 版本已注册: {version_id}")


def main():
    """测试版本控制模块"""
    print("=" * 60)
    print("📋 版本控制模块测试")
    print("=" * 60)

    # 注册当前模型版本
    register_current_model_version()

    # 创建版本管理器
    MODEL_NAME = "efficientnet_b3_loli_optimized_v2_20260529_133654"
    MODEL_DIR = os.path.join(project_root, "models", MODEL_NAME)
    manager = ModelVersionManager(MODEL_DIR)

    # 列出所有版本
    versions = manager.list_versions()
    print(f"\n📊 已注册版本数: {len(versions)}")

    for v in versions:
        print(f"\n版本: {v['id']}")
        print(f"  训练样本: {v.get('training_config', {}).get('train_samples', 'N/A')}")
        print(f"  Top-1 准确率: {v.get('metrics', {}).get('top_1_accuracy', 'N/A')}")
        print(f"  数据增强: {v.get('data_augmentation', [])}")


if __name__ == "__main__":
    main()
