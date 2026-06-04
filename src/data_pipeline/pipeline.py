"""
数据流水线主类
Data Pipeline Main Class
"""
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data_pipeline.collector.deduplication import CLIPDeduplicator
from src.data_pipeline.annotator.yolo_detector import YOLODetector
from src.data_pipeline.active_learning.confidence_filter import (
    ConfidenceFilter,
    SampleReviewer,
    IncrementalTrainer
)
from src.data_pipeline.database.init_db import (
    init_database,
    Sample,
    Annotation,
    Character,
    CollectionTask
)


class DataPipeline:
    """数据流水线主类"""

    def __init__(self, config_path: str = "config/data_pipeline.yaml"):
        """
        初始化数据流水线

        Args:
            config_path: 配置文件路径
        """
        # 初始化数据库
        self.engine, self.Session = init_database()
        self.session = self.Session()

        # 初始化各个模块
        self.deduplicator = CLIPDeduplicator(model_name="ViT-B/32")
        self.detector = YOLODetector(model_path="yolov8n.pt")
        self.confidence_filter = ConfidenceFilter(threshold=0.7)
        self.reviewer = SampleReviewer(review_dir="data/review_batches")
        self.trainer = IncrementalTrainer(model_dir="data/models", data_dir="data/training")

        # 统计信息
        self.stats = {
            'start_time': None,
            'end_time': None,
            'total_samples': 0,
            'imported_samples': 0,
            'deduplicated_samples': 0,
            'annotated_samples': 0,
            'difficult_samples': 0,
            'errors': []
        }

        print("✅ 数据流水线初始化完成")

    def import_samples(self, data_dir: str = "data/final_dataset") -> int:
        """
        导入样本图片

        Args:
            data_dir: 数据目录

        Returns:
            导入的样本数量
        """
        print(f"\n📥 开始导入样本: {data_dir}")
        data_path = Path(data_dir)

        if not data_path.exists():
            print(f"❌ 数据目录不存在: {data_dir}")
            return 0

        imported_count = 0
        skipped_count = 0

        # 遍历每个角色目录
        for character_dir in data_path.iterdir():
            if not character_dir.is_dir():
                continue

            character_name = character_dir.name

            # 查找对应的角色
            character = self.session.query(Character).filter_by(name=character_name).first()
            if not character:
                print(f"⚠️ 未找到角色: {character_name}，跳过")
                continue

            # 遍历图片文件
            image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.webp')
            for image_file in character_dir.rglob('*'):
                if image_file.suffix.lower() not in image_extensions:
                    continue

                # 检查是否已存在
                existing = self.session.query(Sample).filter_by(image_path=str(image_file)).first()
                if existing:
                    skipped_count += 1
                    continue

                # 创建样本记录
                sample = Sample(
                    image_path=str(image_file),
                    character_id=character.id,
                    status='pending'
                )
                self.session.add(sample)
                imported_count += 1

                # 每100条提交一次
                if imported_count % 100 == 0:
                    self.session.commit()
                    print(f"   已导入 {imported_count} 个样本...")

        # 提交剩余数据
        self.session.commit()

        print(f"✅ 导入完成！新增: {imported_count} 个，跳过: {skipped_count} 个")

        self.stats['imported_samples'] = imported_count
        return imported_count

    def deduplicate_samples(self, phash_threshold: int = 5, clip_threshold: float = 0.98) -> Tuple[int, int]:
        """
        去重样本

        Args:
            phash_threshold: 感知哈希阈值
            clip_threshold: CLIP相似度阈值

        Returns:
            (去重后数量, 去除数量)
        """
        print(f"\n🔍 开始去重...")

        # 获取待去重的样本
        samples = self.session.query(Sample).filter(Sample.status == 'pending').all()
        image_paths = [s.image_path for s in samples]

        if not image_paths:
            print("⚠️ 没有待去重的样本")
            return 0, 0

        # 执行去重
        retained, stats = self.deduplicator.deduplicate(
            image_paths,
            phash_threshold=phash_threshold,
            clip_threshold=clip_threshold,
            batch_size=32
        )

        # 更新数据库状态
        removed_count = stats['total_removed']
        retained_set = set(retained)

        for sample in samples:
            if sample.image_path in retained_set:
                sample.status = 'deduplicated'
            else:
                sample.status = 'duplicate'

        self.session.commit()

        print(f"✅ 去重完成！保留: {len(retained)} 个，去除: {removed_count} 个")

        self.stats['deduplicated_samples'] = len(retained)
        return len(retained), removed_count

    def annotate_samples(self, conf_threshold: float = 0.5, limit: int = None) -> int:
        """
        自动标注样本

        Args:
            conf_threshold: 置信度阈值
            limit: 处理数量限制

        Returns:
            标注的样本数量
        """
        print(f"\n📝 开始自动标注...")

        # 获取待标注的样本
        query = self.session.query(Sample).filter(Sample.status == 'deduplicated')
        if limit:
            query = query.limit(limit)

        samples = query.all()
        print(f"📋 找到 {len(samples)} 个待标注样本")

        annotated_count = 0
        skipped_count = 0

        for sample in samples:
            # 检查是否已标注
            existing_annotation = self.session.query(Annotation).filter_by(sample_id=sample.id).first()
            if existing_annotation:
                skipped_count += 1
                continue

            # 执行检测
            detections = self.detector.detect(sample.image_path, conf_threshold=conf_threshold)

            if detections:
                # 使用置信度最高的检测结果
                best_detection = max(detections, key=lambda x: x['confidence'])

                # 创建标注记录
                annotation = Annotation(
                    sample_id=sample.id,
                    annotator='auto',
                    bbox=best_detection['bbox'],
                    confidence=best_detection['confidence'],
                    is_verified=False
                )
                self.session.add(annotation)

                # 更新样本信息
                sample.person_count = len(detections)
                sample.confidence = best_detection['confidence']
                sample.status = 'annotated'

                annotated_count += 1
            else:
                # 未检测到目标
                sample.status = 'no_detection'
                skipped_count += 1

            # 每50条提交一次
            if annotated_count % 50 == 0 and annotated_count > 0:
                self.session.commit()
                print(f"   已标注 {annotated_count} 个样本...")

        # 提交剩余数据
        self.session.commit()

        print(f"✅ 标注完成！已标注: {annotated_count} 个，跳过: {skipped_count} 个")

        self.stats['annotated_samples'] = annotated_count
        return annotated_count

    def filter_difficult_samples(self, batch_size: int = 10, strategy: str = 'confidence') -> List[int]:
        """
        筛选困难样本

        Args:
            batch_size: 批次大小
            strategy: 筛选策略 (confidence, entropy, margin)

        Returns:
            困难样本ID列表
        """
        print(f"\n🎯 开始筛选困难样本...")

        # 获取所有标注
        annotations = self.session.query(Annotation).filter_by(is_verified=False).all()

        if not annotations:
            print("⚠️ 没有待筛选的标注")
            return []

        # 构建预测结果
        predictions = [
            {
                'confidence': ann.confidence,
                'sample_id': ann.sample_id,
                'annotation_id': ann.id
            }
            for ann in annotations
        ]

        # 筛选困难样本
        selected_indices = self.confidence_filter.select_for_review(
            predictions,
            batch_size=batch_size,
            strategy=strategy
        )

        # 获取样本ID
        difficult_sample_ids = [predictions[idx]['sample_id'] for idx in selected_indices]

        print(f"✅ 筛选完成！找到 {len(difficult_sample_ids)} 个困难样本")

        self.stats['difficult_samples'] = len(difficult_sample_ids)
        return difficult_sample_ids

    def create_review_batch(self, sample_ids: List[int], batch_name: Optional[str] = None) -> str:
        """
        创建审核批次

        Args:
            sample_ids: 样本ID列表
            batch_name: 批次名称

        Returns:
            批次ID
        """
        print(f"\n📋 创建审核批次...")

        # 获取样本和标注信息
        samples = self.session.query(Sample).filter(Sample.id.in_(sample_ids)).all()
        sample_dict = {s.id: s for s in samples}

        review_data = []
        for sample_id in sample_ids:
            sample = sample_dict.get(sample_id)
            if not sample:
                continue

            annotation = self.session.query(Annotation).filter_by(sample_id=sample_id).first()
            if not annotation:
                continue

            review_data.append({
                'index': len(review_data),
                'sample_id': sample_id,
                'image_path': sample.image_path,
                'predicted_class': sample.character.name if sample.character else 'Unknown',
                'confidence': annotation.confidence,
                'bbox': annotation.bbox,
                'status': 'pending'
            })

        # 保存批次
        batch_id = batch_name or f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.reviewer.save_review_batch(review_data, batch_id)

        print(f"✅ 审核批次创建完成！批次ID: {batch_id}")

        return batch_id

    def run_full_pipeline(self, data_dir: str = "data/final_dataset", 
                         auto_review: bool = False) -> Dict:
        """
        运行完整流水线

        Args:
            data_dir: 数据目录
            auto_review: 是否自动审核

        Returns:
            流水线统计信息
        """
        print("\n" + "=" * 60)
        print("🚀 开始运行完整数据流水线")
        print("=" * 60)

        self.stats['start_time'] = datetime.now()

        try:
            # 1. 导入样本
            self.import_samples(data_dir)

            # 2. 去重
            self.deduplicate_samples()

            # 3. 自动标注
            self.annotate_samples()

            # 4. 筛选困难样本
            difficult_samples = self.filter_difficult_samples(batch_size=20)

            # 5. 创建审核批次
            if difficult_samples:
                self.create_review_batch(difficult_samples)

            # 6. 自动审核（可选）
            if auto_review and difficult_samples:
                print("\n⚠️ 自动审核功能待实现")

        except Exception as e:
            error_msg = f"流水线执行失败: {str(e)}"
            print(f"❌ {error_msg}")
            self.stats['errors'].append(error_msg)

        finally:
            self.stats['end_time'] = datetime.now()
            self.stats['total_samples'] = self.session.query(Sample).count()

            # 打印统计信息
            self.print_stats()

            # 关闭数据库连接
            self.session.close()
            self.engine.dispose()

        return self.stats

    def print_stats(self):
        """打印统计信息"""
        print("\n" + "=" * 60)
        print("📊 流水线执行统计")
        print("=" * 60)

        if self.stats['start_time'] and self.stats['end_time']:
            duration = (self.stats['end_time'] - self.stats['start_time']).total_seconds()
            print(f"执行时间: {duration:.2f} 秒")

        print(f"总样本数: {self.stats['total_samples']}")
        print(f"导入样本: {self.stats['imported_samples']}")
        print(f"去重后样本: {self.stats['deduplicated_samples']}")
        print(f"标注样本: {self.stats['annotated_samples']}")
        print(f"困难样本: {self.stats['difficult_samples']}")

        if self.stats['errors']:
            print(f"\n⚠️ 错误数量: {len(self.stats['errors'])}")
            for error in self.stats['errors']:
                print(f"  - {error}")

        print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="数据流水线工具")
    parser.add_argument("-d", "--dir", default="data/final_dataset", help="数据目录")
    parser.add_argument("--auto-review", action="store_true", help="自动审核")
    args = parser.parse_args()

    # 创建流水线
    pipeline = DataPipeline()

    # 运行完整流水线
    stats = pipeline.run_full_pipeline(args.dir, args.auto_review)