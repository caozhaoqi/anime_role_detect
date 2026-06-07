#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动增量训练机制
实现模型的持续学习和自动更新

功能：
- 监控模型性能
- 自动收集新数据
- 触发增量训练
- 评估和部署新版本
"""

import os
import json
import time
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
import logging

import numpy as np
from PIL import Image

logger = logging.getLogger("incremental_trainer")


@dataclass
class TrainingConfig:
    """训练配置"""
    # 触发条件
    min_new_samples: int = 100  # 最小新样本数触发训练
    performance_threshold: float = 0.05  # 性能下降阈值
    check_interval_hours: int = 24  # 检查间隔（小时）
    
    # 训练参数
    learning_rate: float = 1e-4
    epochs: int = 10
    batch_size: int = 32
    warmup_epochs: int = 2
    
    # 数据参数
    min_samples_per_class: int = 10
    max_samples_per_class: int = 500
    validation_split: float = 0.2
    
    # 模型参数
    model_name: str = "ViT-B/32"
    embedding_dim: int = 512
    
    # 保存路径
    model_dir: str = "models/incremental"
    backup_dir: str = "models/backup"
    log_dir: str = "logs/training"


@dataclass
class PerformanceMetrics:
    """性能指标"""
    timestamp: str
    accuracy: float
    top5_accuracy: float
    avg_confidence: float
    num_test_samples: int
    confusion_score: float  # 混淆度，越高表示越难区分
    
    def is_degraded(self, baseline: "PerformanceMetrics", threshold: float = 0.05) -> bool:
        """检查性能是否下降"""
        acc_drop = baseline.accuracy - self.accuracy
        return acc_drop > threshold


class IncrementalDataCollector:
    """
    增量数据收集器
    收集用户反馈和新数据用于增量训练
    """
    
    def __init__(self, data_dir: str = "data/incremental"):
        """
        初始化
        
        Args:
            data_dir: 增量数据存储目录
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建子目录
        (self.data_dir / "new_samples").mkdir(exist_ok=True)
        (self.data_dir / "feedback").mkdir(exist_ok=True)
        (self.data_dir / "misclassified").mkdir(exist_ok=True)
        
        self.collected_count = 0
        
        logger.info(f"增量数据收集器初始化: {data_dir}")
    
    def add_new_sample(
        self,
        image: np.ndarray,
        character_name: str,
        source: str = "user_upload",
        metadata: Optional[Dict] = None,
    ) -> str:
        """
        添加新样本
        
        Args:
            image: 图片数组
            character_name: 角色名称
            source: 数据来源
            metadata: 附加元数据
            
        Returns:
            样本ID
        """
        sample_id = f"{character_name}_{int(time.time())}_{self.collected_count}"
        self.collected_count += 1
        
        # 保存图片
        char_dir = self.data_dir / "new_samples" / character_name
        char_dir.mkdir(exist_ok=True)
        
        img_path = char_dir / f"{sample_id}.jpg"
        Image.fromarray(image).save(img_path)
        
        # 保存元数据
        meta = {
            "sample_id": sample_id,
            "character": character_name,
            "source": source,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
        }
        
        meta_path = char_dir / f"{sample_id}.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        
        logger.info(f"添加新样本: {sample_id} -> {character_name}")
        return sample_id
    
    def add_feedback(
        self,
        image: np.ndarray,
        predicted_character: str,
        correct_character: str,
        confidence: float,
    ) -> str:
        """
        添加错误反馈
        
        Args:
            image: 图片
            predicted_character: 预测的角色
            correct_character: 正确的角色
            confidence: 预测置信度
            
        Returns:
            反馈ID
        """
        feedback_id = f"feedback_{int(time.time())}"
        
        # 保存到错误分类目录
        feedback_dir = self.data_dir / "misclassified" / correct_character
        feedback_dir.mkdir(parents=True, exist_ok=True)
        
        img_path = feedback_dir / f"{feedback_id}.jpg"
        Image.fromarray(image).save(img_path)
        
        # 保存反馈信息
        feedback = {
            "feedback_id": feedback_id,
            "predicted": predicted_character,
            "correct": correct_character,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat(),
        }
        
        feedback_path = feedback_dir / f"{feedback_id}.json"
        with open(feedback_path, "w", encoding="utf-8") as f:
            json.dump(feedback, f, indent=2)
        
        logger.info(f"添加反馈: {predicted_character} -> {correct_character}")
        return feedback_id
    
    def get_collected_data(self) -> Dict[str, List[str]]:
        """
        获取已收集的数据
        
        Returns:
            角色到图片路径的映射
        """
        data = {}
        
        # 收集新样本
        new_samples_dir = self.data_dir / "new_samples"
        if new_samples_dir.exists():
            for char_dir in new_samples_dir.iterdir():
                if char_dir.is_dir():
                    images = list(char_dir.glob("*.jpg"))
                    data[char_dir.name] = [str(p) for p in images]
        
        # 收集错误反馈
        feedback_dir = self.data_dir / "misclassified"
        if feedback_dir.exists():
            for char_dir in feedback_dir.iterdir():
                if char_dir.is_dir():
                    images = list(char_dir.glob("*.jpg"))
                    if char_dir.name in data:
                        data[char_dir.name].extend([str(p) for p in images])
                    else:
                        data[char_dir.name] = [str(p) for p in images]
        
        return data
    
    def get_total_samples(self) -> int:
        """获取总样本数"""
        data = self.get_collected_data()
        return sum(len(images) for images in data.values())
    
    def clear_collected_data(self):
        """清空已收集的数据（训练后调用）"""
        # 备份并清空
        backup_dir = self.data_dir / "archive" / datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        for subdir in ["new_samples", "misclassified"]:
            src = self.data_dir / subdir
            if src.exists():
                dst = backup_dir / subdir
                shutil.move(str(src), str(dst))
                src.mkdir(exist_ok=True)
        
        logger.info(f"已收集数据已归档: {backup_dir}")


class IncrementalTrainer:
    """
    增量训练器
    实现模型的增量更新
    """
    
    def __init__(self, config: Optional[TrainingConfig] = None):
        """
        初始化
        
        Args:
            config: 训练配置
        """
        self.config = config or TrainingConfig()
        self.data_collector = IncrementalDataCollector()
        
        # 创建目录
        Path(self.config.model_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.backup_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.log_dir).mkdir(parents=True, exist_ok=True)
        
        # 性能历史
        self.performance_history: List[PerformanceMetrics] = []
        self.baseline_metrics: Optional[PerformanceMetrics] = None
        
        # 加载历史
        self._load_history()
        
        logger.info("增量训练器初始化完成")
    
    def _load_history(self):
        """加载历史性能数据"""
        history_path = Path(self.config.log_dir) / "performance_history.json"
        if history_path.exists():
            with open(history_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.performance_history = [
                    PerformanceMetrics(**m) for m in data
                ]
            
            if self.performance_history:
                self.baseline_metrics = self.performance_history[-1]
                logger.info(f"加载历史性能数据: {len(self.performance_history)}条记录")
    
    def _save_history(self):
        """保存历史性能数据"""
        history_path = Path(self.config.log_dir) / "performance_history.json"
        data = [asdict(m) for m in self.performance_history]
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    
    def check_training_needed(self) -> Tuple[bool, str]:
        """
        检查是否需要训练
        
        Returns:
            (是否需要训练, 原因)
        """
        # 检查新样本数
        total_samples = self.data_collector.get_total_samples()
        if total_samples >= self.config.min_new_samples:
            return True, f"新样本数达到阈值: {total_samples}/{self.config.min_new_samples}"
        
        # 检查性能下降
        if self.baseline_metrics and len(self.performance_history) >= 2:
            recent = self.performance_history[-1]
            if recent.is_degraded(self.baseline_metrics, self.config.performance_threshold):
                return True, f"性能下降: {self.baseline_metrics.accuracy:.3f} -> {recent.accuracy:.3f}"
        
        return False, "未达到训练条件"
    
    def evaluate_model(
        self,
        embedder,
        feature_store,
        test_images_dir: str,
    ) -> PerformanceMetrics:
        """
        评估当前模型性能
        
        Args:
            embedder: 特征提取器
            feature_store: 特征库
            test_images_dir: 测试图片目录
            
        Returns:
            性能指标
        """
        from src.utils.memory_optimizer import MemoryMonitor
        
        monitor = MemoryMonitor()
        
        # 加载测试数据
        test_images = []
        true_labels = []
        
        test_dir = Path(test_images_dir)
        if test_dir.exists():
            for char_dir in test_dir.iterdir():
                if char_dir.is_dir():
                    for img_path in char_dir.glob("*.jpg"):
                        test_images.append(str(img_path))
                        true_labels.append(char_dir.name)
        
        if not test_images:
            logger.warning("没有找到测试数据")
            return PerformanceMetrics(
                timestamp=datetime.now().isoformat(),
                accuracy=0,
                top5_accuracy=0,
                avg_confidence=0,
                num_test_samples=0,
                confusion_score=0,
            )
        
        # 测试
        correct = 0
        top5_correct = 0
        confidences = []
        
        for img_path, true_label in zip(test_images, true_labels):
            feature = embedder.embed_image(img_path)
            if feature is None:
                continue
            
            results = feature_store.search(feature, top_k=5)
            
            if results:
                top1_char, top1_sim = results[0]
                confidences.append(top1_sim)
                
                if top1_char == true_label:
                    correct += 1
                
                # Top-5
                top5_chars = [char for char, _ in results]
                if true_label in top5_chars:
                    top5_correct += 1
        
        total = len(test_images)
        accuracy = correct / total if total > 0 else 0
        top5_accuracy = top5_correct / total if total > 0 else 0
        avg_confidence = np.mean(confidences) if confidences else 0
        
        # 计算混淆度（Top-1和Top-2的相似度差距）
        confusion_scores = []
        for img_path in test_images[:100]:  # 采样计算
            feature = embedder.embed_image(img_path)
            if feature is not None:
                results = feature_store.search(feature, top_k=2)
                if len(results) >= 2:
                    gap = results[0][1] - results[1][1]
                    confusion_scores.append(gap)
        
        confusion_score = 1 - np.mean(confusion_scores) if confusion_scores else 0
        
        metrics = PerformanceMetrics(
            timestamp=datetime.now().isoformat(),
            accuracy=accuracy,
            top5_accuracy=top5_accuracy,
            avg_confidence=avg_confidence,
            num_test_samples=total,
            confusion_score=confusion_score,
        )
        
        # 保存
        self.performance_history.append(metrics)
        self._save_history()
        
        if self.baseline_metrics is None:
            self.baseline_metrics = metrics
        
        logger.info(f"模型评估完成: Accuracy={accuracy:.3f}, Top5={top5_accuracy:.3f}")
        return metrics
    
    def incremental_train(
        self,
        embedder,
        feature_store,
    ) -> Dict:
        """
        执行增量训练
        
        Args:
            embedder: 特征提取器
            feature_store: 特征库
            
        Returns:
            训练结果
        """
        logger.info("开始增量训练...")
        start_time = time.time()
        
        # 1. 收集新数据
        new_data = self.data_collector.get_collected_data()
        if not new_data:
            return {"success": False, "message": "没有新数据"}
        
        logger.info(f"收集到新数据: {sum(len(v) for v in new_data.values())} 张图片")
        
        # 2. 提取特征并更新特征库
        updated_count = 0
        for character_name, image_paths in new_data.items():
            features = []
            for img_path in image_paths:
                feature = embedder.embed_image(img_path)
                if feature is not None:
                    features.append(feature)
            
            if features:
                feature_store.add_features_to_character(character_name, features)
                updated_count += len(features)
                logger.info(f"更新角色 '{character_name}': {len(features)} 个特征")
        
        # 3. 保存更新后的特征库
        model_path = Path(self.config.model_dir) / f"feature_store_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        feature_store.save(str(model_path))
        
        # 4. 清空已收集的数据
        self.data_collector.clear_collected_data()
        
        training_time = time.time() - start_time
        
        result = {
            "success": True,
            "message": "增量训练完成",
            "updated_features": updated_count,
            "updated_characters": len(new_data),
            "training_time_seconds": training_time,
            "model_path": str(model_path),
        }
        
        logger.info(f"增量训练完成: 更新 {updated_count} 个特征, 耗时 {training_time:.1f}秒")
        return result
    
    def run_auto_training_cycle(
        self,
        embedder,
        feature_store,
        test_images_dir: str,
    ) -> Dict:
        """
        运行自动训练周期
        
        Args:
            embedder: 特征提取器
            feature_store: 特征库
            test_images_dir: 测试图片目录
            
        Returns:
            周期结果
        """
        logger.info("=" * 60)
        logger.info("开始自动训练周期")
        logger.info("=" * 60)
        
        # 1. 评估当前性能
        current_metrics = self.evaluate_model(embedder, feature_store, test_images_dir)
        
        # 2. 检查是否需要训练
        need_train, reason = self.check_training_needed()
        
        result = {
            "timestamp": datetime.now().isoformat(),
            "current_metrics": asdict(current_metrics),
            "training_needed": need_train,
            "reason": reason,
        }
        
        if need_train:
            logger.info(f"触发训练: {reason}")
            
            # 执行训练
            train_result = self.incremental_train(embedder, feature_store)
            result["training_result"] = train_result
            
            # 重新评估
            new_metrics = self.evaluate_model(embedder, feature_store, test_images_dir)
            result["new_metrics"] = asdict(new_metrics)
            
            # 比较
            if new_metrics.accuracy >= current_metrics.accuracy:
                result["improvement"] = new_metrics.accuracy - current_metrics.accuracy
                logger.info(f"训练提升: +{result['improvement']:.3f}")
            else:
                result["improvement"] = new_metrics.accuracy - current_metrics.accuracy
                logger.warning(f"训练后性能下降: {result['improvement']:.3f}")
        else:
            logger.info(f"跳过训练: {reason}")
        
        # 保存周期结果
        cycle_path = Path(self.config.log_dir) / f"cycle_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(cycle_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        
        return result
    
    def schedule_training(
        self,
        embedder,
        feature_store,
        test_images_dir: str,
    ):
        """
        调度训练任务（定时运行）
        
        Args:
            embedder: 特征提取器
            feature_store: 特征库
            test_images_dir: 测试图片目录
        """
        import threading
        
        def training_loop():
            while True:
                try:
                    self.run_auto_training_cycle(embedder, feature_store, test_images_dir)
                except Exception as e:
                    logger.error(f"训练周期错误: {e}")
                
                # 等待下次检查
                time.sleep(self.config.check_interval_hours * 3600)
        
        thread = threading.Thread(target=training_loop, daemon=True)
        thread.start()
        
        logger.info(f"训练调度器已启动，检查间隔: {self.config.check_interval_hours}小时")


if __name__ == "__main__":
    # 测试
    config = TrainingConfig(
        min_new_samples=10,
        check_interval_hours=1,
    )
    
    trainer = IncrementalTrainer(config)
    
    # 模拟添加数据
    print(f"当前样本数: {trainer.data_collector.get_total_samples()}")
    print(f"需要训练: {trainer.check_training_needed()}")
