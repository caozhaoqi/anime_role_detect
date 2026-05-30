#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动化反馈闭环模块 - 模糊样本记录与增量训练
记录置信度在 0.5-0.7 之间的"模糊样本"，供人工标注后进行增量训练
"""

import os
import sys
import json
import shutil
from datetime import datetime
from typing import List, Dict, Optional
from PIL import Image
import numpy as np

# 添加项目根目录
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.insert(0, project_root)


class AmbiguousSampleRecorder:
    """模糊样本记录器"""

    def __init__(self, output_dir: str = None, fuzzy_low: float = 0.5, fuzzy_high: float = 0.7):
        """
        初始化模糊样本记录器

        Args:
            output_dir: 模糊样本输出目录
            fuzzy_low: 模糊样本置信度下限
            fuzzy_high: 模糊样本置信度上限
        """
        if output_dir is None:
            output_dir = os.path.join(project_root, "data", "ambiguous_samples")

        self.output_dir = output_dir
        self.fuzzy_low = fuzzy_low
        self.fuzzy_high = fuzzy_high

        # 创建目录结构
        self.pending_dir = os.path.join(output_dir, "pending")
        self.annotated_dir = os.path.join(output_dir, "annotated")
        self.rejected_dir = os.path.join(output_dir, "rejected")

        for d in [self.pending_dir, self.annotated_dir, self.rejected_dir]:
            os.makedirs(d, exist_ok=True)

        # 记录文件
        self.records_file = os.path.join(output_dir, "records.json")
        self.records = self._load_records()

    def _load_records(self) -> Dict:
        """加载记录"""
        if os.path.exists(self.records_file):
            with open(self.records_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {
            "created_at": datetime.now().isoformat(),
            "total_pending": 0,
            "total_annotated": 0,
            "total_rejected": 0,
            "samples": [],
        }

    def _save_records(self):
        """保存记录"""
        with open(self.records_file, "w", encoding="utf-8") as f:
            json.dump(self.records, f, indent=2, ensure_ascii=False)

    def is_fuzzy(self, confidence: float) -> bool:
        """判断是否为模糊样本"""
        return self.fuzzy_low <= confidence < self.fuzzy_high

    def is_unknown(self, confidence: float) -> bool:
        """判断是否为未知样本（低于下限）"""
        return confidence < self.fuzzy_low

    def record_sample(
        self, image: Image.Image, prediction: Dict, metadata: Dict = None
    ) -> Optional[str]:
        """
        记录模糊样本

        Args:
            image: PIL Image
            prediction: 预测结果，包含 role, confidence 等
            metadata: 额外元数据

        Returns:
            样本 ID 或 None（如果不是模糊样本）
        """
        confidence = prediction.get("confidence", 0.0)

        # 只记录模糊样本
        if not self.is_fuzzy(confidence):
            return None

        # 生成样本 ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        sample_id = f"fuzzy_{timestamp}"

        # 保存图像
        image_filename = f"{sample_id}.jpg"
        image_path = os.path.join(self.pending_dir, image_filename)
        image.save(image_path, quality=95)

        # 记录信息
        sample_info = {
            "id": sample_id,
            "image_path": image_path,
            "recorded_at": datetime.now().isoformat(),
            "status": "pending",
            "prediction": prediction,
            "metadata": metadata or {},
        }

        self.records["samples"].append(sample_info)
        self.records["total_pending"] += 1
        self._save_records()

        print(f"✅ 记录模糊样本: {sample_id} (置信度: {confidence:.3f})")
        return sample_id

    def batch_record(
        self, images: List[Image.Image], predictions: List[Dict], metadata_list: List[Dict] = None
    ) -> List[str]:
        """批量记录模糊样本"""
        sample_ids = []
        metadata_list = metadata_list or [None] * len(images)

        for image, prediction, metadata in zip(images, predictions, metadata_list):
            sample_id = self.record_sample(image, prediction, metadata)
            if sample_id:
                sample_ids.append(sample_id)

        return sample_ids

    def annotate_sample(self, sample_id: str, correct_role: str, notes: str = "") -> bool:
        """
        标注样本（人工确认正确角色）

        Args:
            sample_id: 样本 ID
            correct_role: 正确角色名
            notes: 备注

        Returns:
            是否成功
        """
        # 查找样本
        sample = None
        sample_idx = None
        for i, s in enumerate(self.records["samples"]):
            if s["id"] == sample_id:
                sample = s
                sample_idx = i
                break

        if sample is None:
            print(f"❌ 样本不存在: {sample_id}")
            return False

        # 移动图像到 annotated 目录
        old_path = sample["image_path"]
        new_filename = f"{correct_role}_{sample_id}.jpg"
        new_path = os.path.join(self.annotated_dir, new_filename)

        shutil.move(old_path, new_path)

        # 更新记录
        sample["status"] = "annotated"
        sample["annotated_at"] = datetime.now().isoformat()
        sample["correct_role"] = correct_role
        sample["new_image_path"] = new_path
        sample["notes"] = notes

        self.records["total_pending"] -= 1
        self.records["total_annotated"] += 1
        self._save_records()

        print(f"✅ 样本已标注: {sample_id} -> {correct_role}")
        return True

    def reject_sample(self, sample_id: str, reason: str = "") -> bool:
        """
        拒绝样本（无法标注，如未知角色）

        Args:
            sample_id: 样本 ID
            reason: 拒绝原因

        Returns:
            是否成功
        """
        # 查找样本
        sample = None
        for s in self.records["samples"]:
            if s["id"] == sample_id:
                sample = s
                break

        if sample is None:
            print(f"❌ 样本不存在: {sample_id}")
            return False

        # 移动图像到 rejected 目录
        old_path = sample["image_path"]
        new_filename = f"rejected_{sample_id}.jpg"
        new_path = os.path.join(self.rejected_dir, new_filename)

        shutil.move(old_path, new_path)

        # 更新记录
        sample["status"] = "rejected"
        sample["rejected_at"] = datetime.now().isoformat()
        sample["rejection_reason"] = reason

        self.records["total_pending"] -= 1
        self.records["total_rejected"] += 1
        self._save_records()

        print(f"✅ 样本已拒绝: {sample_id} ({reason})")
        return True

    def get_pending_samples(self) -> List[Dict]:
        """获取待标注样本"""
        return [s for s in self.records["samples"] if s["status"] == "pending"]

    def get_annotated_samples(self) -> List[Dict]:
        """获取已标注样本（可用于增量训练）"""
        return [s for s in self.records["samples"] if s["status"] == "annotated"]

    def export_for_training(self, output_path: str = None) -> str:
        """
        导出已标注样本用于增量训练

        Args:
            output_path: 输出路径

        Returns:
            导出文件路径
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(
                project_root, "data", "incremental_training", f"ambiguous_{timestamp}.json"
            )

        annotated = self.get_annotated_samples()

        export_data = {
            "exported_at": datetime.now().isoformat(),
            "total_samples": len(annotated),
            "samples": [],
        }

        for sample in annotated:
            export_data["samples"].append(
                {
                    "image_path": sample["new_image_path"],
                    "correct_role": sample["correct_role"],
                    "original_prediction": sample["prediction"],
                    "recorded_at": sample["recorded_at"],
                    "annotated_at": sample["annotated_at"],
                    "notes": sample.get("notes", ""),
                }
            )

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        print(f"✅ 已导出 {len(annotated)} 个样本到: {output_path}")
        return output_path

    def get_statistics(self) -> Dict:
        """获取统计信息"""
        total = len(self.records["samples"])
        pending = self.records["total_pending"]
        annotated = self.records["total_annotated"]
        rejected = self.records["total_rejected"]

        # 按角色统计
        role_counts = {}
        for sample in self.records["samples"]:
            if sample["status"] == "annotated":
                role = sample.get("correct_role", "unknown")
                role_counts[role] = role_counts.get(role, 0) + 1

        return {
            "total_samples": total,
            "pending": pending,
            "annotated": annotated,
            "rejected": rejected,
            "pending_ratio": pending / total if total > 0 else 0,
            "annotated_ratio": annotated / total if total > 0 else 0,
            "role_distribution": role_counts,
        }


def main():
    """测试模糊样本记录器"""
    print("=" * 60)
    print("🔍 模糊样本记录器测试")
    print("=" * 60)

    # 创建记录器
    recorder = AmbiguousSampleRecorder()

    print(f"\n📂 输出目录: {recorder.output_dir}")
    print(f"📊 模糊样本范围: {recorder.fuzzy_low} - {recorder.fuzzy_high}")

    # 测试记录（模拟一些预测结果）
    print("\n📝 测试记录模糊样本...")

    # 创建一个测试图像
    test_image = Image.new("RGB", (224, 224), color=(100, 150, 200))

    # 测试不同置信度的预测
    test_predictions = [
        {"role": "Hoshino", "confidence": 0.55},  # 模糊
        {"role": "Hoshino", "confidence": 0.65},  # 模糊
        {"role": "Hoshino", "confidence": 0.85},  # 不是模糊
        {"role": "Unknown", "confidence": 0.30},  # 不是模糊
    ]

    recorded_ids = recorder.batch_record([test_image] * len(test_predictions), test_predictions)

    print(f"\n✅ 记录了 {len(recorded_ids)} 个模糊样本")

    # 获取统计
    stats = recorder.get_statistics()
    print(f"\n📊 统计信息:")
    print(f"   总样本: {stats['total_samples']}")
    print(f"   待标注: {stats['pending']}")
    print(f"   已标注: {stats['annotated']}")
    print(f"   已拒绝: {stats['rejected']}")


if __name__ == "__main__":
    main()
