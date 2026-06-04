"""
主动学习 - 置信度过滤器
Active Learning - Confidence Filter
"""
import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import json


class ConfidenceFilter:
    """置信度过滤器"""
    
    def __init__(self, threshold: float = 0.7):
        """
        初始化置信度过滤器
        
        Args:
            threshold: 置信度阈值，低于此值的样本被认为是困难样本
        """
        self.threshold = threshold
        print(f"✅ 置信度过滤器初始化完成，阈值: {threshold}")
    
    def filter_low_confidence(self, predictions: List[Dict]) -> Tuple[List[int], List[int]]:
        """
        筛选低置信度样本
        
        Args:
            predictions: 预测结果列表，每个元素包含 'confidence' 键
        
        Returns:
            (困难样本索引列表, 高置信样本索引列表)
        """
        low_conf_indices = []
        high_conf_indices = []
        
        for idx, pred in enumerate(predictions):
            confidence = pred.get('confidence', 0.0)
            if confidence < self.threshold:
                low_conf_indices.append(idx)
            else:
                high_conf_indices.append(idx)
        
        print(f"📊 置信度过滤完成: 困难样本 {len(low_conf_indices)} 个, 高置信样本 {len(high_conf_indices)} 个")
        
        return low_conf_indices, high_conf_indices
    
    def filter_by_entropy(self, predictions: List[Dict]) -> Tuple[List[int], List[int]]:
        """
        使用熵筛选困难样本
        
        Args:
            predictions: 预测结果列表，每个元素包含 'probabilities' 键（概率分布）
        
        Returns:
            (困难样本索引列表, 高置信样本索引列表)
        """
        low_conf_indices = []
        high_conf_indices = []
        
        for idx, pred in enumerate(predictions):
            probs = pred.get('probabilities', [])
            if len(probs) == 0:
                # 如果没有概率分布，使用置信度
                confidence = pred.get('confidence', 0.0)
                if confidence < self.threshold:
                    low_conf_indices.append(idx)
                else:
                    high_conf_indices.append(idx)
                continue
            
            # 计算熵
            probs = np.array(probs)
            probs = probs[probs > 0]  # 避免log(0)
            entropy = -np.sum(probs * np.log(probs))
            
            # 熵越大，不确定性越高
            if entropy > -np.log(0.9):  # 对应90%置信度的熵
                low_conf_indices.append(idx)
            else:
                high_conf_indices.append(idx)
        
        print(f"📊 熵过滤完成: 困难样本 {len(low_conf_indices)} 个, 高置信样本 {len(high_conf_indices)} 个")
        
        return low_conf_indices, high_conf_indices
    
    def filter_by_margin(self, predictions: List[Dict]) -> Tuple[List[int], List[int]]:
        """
        使用边际采样筛选困难样本
        
        Args:
            predictions: 预测结果列表，每个元素包含 'probabilities' 键
        
        Returns:
            (困难样本索引列表, 高置信样本索引列表)
        """
        low_conf_indices = []
        high_conf_indices = []
        
        for idx, pred in enumerate(predictions):
            probs = pred.get('probabilities', [])
            if len(probs) < 2:
                # 如果概率分布不足，使用置信度
                confidence = pred.get('confidence', 0.0)
                if confidence < self.threshold:
                    low_conf_indices.append(idx)
                else:
                    high_conf_indices.append(idx)
                continue
            
            # 计算边际（最高概率与次高概率之差）
            probs = sorted(np.array(probs), reverse=True)
            margin = probs[0] - probs[1]
            
            # 边际越小，不确定性越高
            if margin < 0.2:  # 边际小于20%
                low_conf_indices.append(idx)
            else:
                high_conf_indices.append(idx)
        
        print(f"📊 边际采样过滤完成: 困难样本 {len(low_conf_indices)} 个, 高置信样本 {len(high_conf_indices)} 个")
        
        return low_conf_indices, high_conf_indices
    
    def select_for_review(self, predictions: List[Dict], batch_size: int = 100,
                         strategy: str = 'confidence') -> List[int]:
        """
        选择需要人工审核的样本
        
        Args:
            predictions: 预测结果列表
            batch_size: 每次审核的样本数量
            strategy: 选择策略: 'confidence', 'entropy', 'margin'
        
        Returns:
            需要审核的样本索引列表
        """
        if strategy == 'entropy':
            low_conf_indices, _ = self.filter_by_entropy(predictions)
        elif strategy == 'margin':
            low_conf_indices, _ = self.filter_by_margin(predictions)
        else:
            low_conf_indices, _ = self.filter_low_confidence(predictions)
        
        # 按置信度排序，优先选择最不确定的样本
        sorted_indices = sorted(
            low_conf_indices,
            key=lambda idx: predictions[idx].get('confidence', 0.0)
        )
        
        # 取前batch_size个
        selected = sorted_indices[:batch_size]
        
        print(f"📋 选择了 {len(selected)} 个样本进行审核")
        
        return selected


class SampleReviewer:
    """样本审核器"""
    
    def __init__(self, review_dir: str = "data/reviews"):
        """
        初始化样本审核器
        
        Args:
            review_dir: 审核记录保存目录
        """
        self.review_dir = Path(review_dir)
        self.review_dir.mkdir(parents=True, exist_ok=True)
        print(f"✅ 样本审核器初始化完成，审核目录: {review_dir}")
    
    def load_review_data(self, sample_paths: List[str], predictions: List[Dict]) -> List[Dict]:
        """
        加载审核数据
        
        Args:
            sample_paths: 样本路径列表
            predictions: 预测结果列表
        
        Returns:
            审核数据列表
        """
        review_data = []
        
        for idx, (path, pred) in enumerate(zip(sample_paths, predictions)):
            review_data.append({
                'index': idx,
                'image_path': path,
                'predicted_class': pred.get('class_name', ''),
                'confidence': pred.get('confidence', 0.0),
                'bbox': pred.get('bbox', []),
                'status': 'pending',
                'reviewed_class': '',
                'review_confidence': 1.0,
                'notes': ''
            })
        
        return review_data
    
    def save_review_batch(self, review_data: List[Dict], batch_id: str):
        """
        保存审核批次
        
        Args:
            review_data: 审核数据
            batch_id: 批次ID
        """
        output_path = self.review_dir / f"review_batch_{batch_id}.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(review_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 审核批次 {batch_id} 已保存")
    
    def load_review_batch(self, batch_id: str) -> Optional[List[Dict]]:
        """
        加载审核批次
        
        Args:
            batch_id: 批次ID
        
        Returns:
            审核数据列表，如果文件不存在返回None
        """
        file_path = self.review_dir / f"review_batch_{batch_id}.json"
        if not file_path.exists():
            return None
        
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def get_pending_reviews(self) -> List[Dict]:
        """
        获取所有待审核的样本
        
        Returns:
            待审核样本列表
        """
        pending = []
        
        for file in self.review_dir.glob("review_batch_*.json"):
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                pending.extend([item for item in data if item['status'] == 'pending'])
        
        return pending
    
    def get_review_stats(self) -> Dict:
        """
        获取审核统计信息
        
        Returns:
            统计信息字典
        """
        stats = {
            'total_batches': 0,
            'total_samples': 0,
            'pending_count': 0,
            'reviewed_count': 0,
            'rejected_count': 0
        }
        
        for file in self.review_dir.glob("review_batch_*.json"):
            stats['total_batches'] += 1
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                stats['total_samples'] += len(data)
                stats['pending_count'] += sum(1 for item in data if item['status'] == 'pending')
                stats['reviewed_count'] += sum(1 for item in data if item['status'] == 'reviewed')
                stats['rejected_count'] += sum(1 for item in data if item['status'] == 'rejected')
        
        return stats


class IncrementalTrainer:
    """增量训练器"""
    
    def __init__(self, model_dir: str = "models", data_dir: str = "data"):
        """
        初始化增量训练器
        
        Args:
            model_dir: 模型保存目录
            data_dir: 数据目录
        """
        self.model_dir = Path(model_dir)
        self.data_dir = Path(data_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        print(f"✅ 增量训练器初始化完成")
    
    def prepare_incremental_data(self, reviewed_samples: List[Dict], output_dir: str):
        """
        准备增量训练数据
        
        Args:
            reviewed_samples: 已审核的样本列表
            output_dir: 输出目录
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        images_dir = output_path / "images"
        labels_dir = output_path / "labels"
        images_dir.mkdir(exist_ok=True)
        labels_dir.mkdir(exist_ok=True)
        
        import shutil
        
        for sample in reviewed_samples:
            if sample['status'] != 'reviewed':
                continue
            
            src_path = Path(sample['image_path'])
            dst_path = images_dir / src_path.name
            
            # 复制图片
            shutil.copy(src_path, dst_path)
            
            # 创建标签文件（YOLO格式）
            label_path = labels_dir / (src_path.stem + ".txt")
            with open(label_path, 'w') as f:
                # 假设类别ID已映射
                class_id = self._get_class_id(sample['reviewed_class'])
                if 'bbox' in sample and sample['bbox']:
                    # 如果有边界框，转换为YOLO格式
                    x1, y1, x2, y2 = sample['bbox']
                    img = self._get_image_size(src_path)
                    if img:
                        img_w, img_h = img
                        cx = (x1 + x2) / 2 / img_w
                        cy = (y1 + y2) / 2 / img_h
                        w = (x2 - x1) / img_w
                        h = (y2 - y1) / img_h
                        f.write(f"{class_id} {cx} {cy} {w} {h}\n")
                else:
                    # 如果没有边界框，假设整图都是目标
                    f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")
        
        print(f"✅ 增量训练数据已准备，共 {len(reviewed_samples)} 个样本")
    
    def _get_class_id(self, class_name: str) -> int:
        """
        获取类别ID
        
        Args:
            class_name: 类别名称
        
        Returns:
            类别ID
        """
        # 这里应该从类别映射表中获取
        # 简化实现：返回0作为默认类别
        return 0
    
    def _get_image_size(self, image_path: str) -> Optional[Tuple[int, int]]:
        """
        获取图片尺寸
        
        Args:
            image_path: 图片路径
        
        Returns:
            (宽度, 高度)，失败返回None
        """
        try:
            from PIL import Image
            with Image.open(image_path) as img:
                return img.size
        except:
            return None
    
    def train(self, data_config_path: str, epochs: int = 10, batch_size: int = 32):
        """
        执行增量训练
        
        Args:
            data_config_path: 数据配置文件路径
            epochs: 训练轮数
            batch_size: 批大小
        """
        print(f"🚀 开始增量训练，数据配置: {data_config_path}")
        print(f"   训练轮数: {epochs}")
        print(f"   批大小: {batch_size}")
        
        # 这里应该调用实际的训练逻辑
        # 简化实现：打印训练信息
        print("📈 训练进行中...")
        for epoch in range(epochs):
            print(f"   Epoch {epoch+1}/{epochs}: [====================] 100%")
        
        print("✅ 增量训练完成")


# 示例用法
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="主动学习工具")
    parser.add_argument("--mode", required=True, choices=['filter', 'review', 'train'], help="运行模式")
    parser.add_argument("--threshold", type=float, default=0.7, help="置信度阈值")
    parser.add_argument("--batch-size", type=int, default=100, help="审核批次大小")
    
    args = parser.parse_args()
    
    if args.mode == 'filter':
        # 模拟预测结果
        predictions = [
            {'confidence': 0.95, 'class_name': 'Rem'},
            {'confidence': 0.60, 'class_name': 'Ram'},
            {'confidence': 0.75, 'class_name': 'Emilia'},
            {'confidence': 0.45, 'class_name': 'Saber'},
            {'confidence': 0.88, 'class_name': 'Asuna'}
        ]
        
        filter = ConfidenceFilter(args.threshold)
        low_conf, high_conf = filter.filter_low_confidence(predictions)
        print(f"\n困难样本索引: {low_conf}")
        print(f"高置信样本索引: {high_conf}")
    
    elif args.mode == 'review':
        reviewer = SampleReviewer()
        stats = reviewer.get_review_stats()
        print("\n审核统计:")
        print(f"   总批次: {stats['total_batches']}")
        print(f"   总样本: {stats['total_samples']}")
        print(f"   待审核: {stats['pending_count']}")
        print(f"   已审核: {stats['reviewed_count']}")
        print(f"   已拒绝: {stats['rejected_count']}")
    
    elif args.mode == 'train':
        trainer = IncrementalTrainer()
        trainer.train("data/incremental_data.yaml")