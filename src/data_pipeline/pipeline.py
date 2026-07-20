"""
数据流水线主类
Data Pipeline Main Class
"""
# 必须在导入任何其他模块之前设置环境变量
import os
import sys
import platform
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from functools import wraps
import time

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_pipeline.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Mac平台禁用CUDA，避免mutex错误
if platform.system() == "Darwin":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    os.environ["FORCE_CPU"] = "1"

# 添加项目路径
project_root = Path(__file__).parent.parent.parent

from src.data_pipeline.collector.deduplication import CLIPDeduplicator
from src.data_pipeline.annotator.yolo_detector import YOLODetector
from src.data_pipeline.active_learning.confidence_filter import (
    ConfidenceFilter,
    SampleReviewer,
    IncrementalTrainer
)
from src.data_pipeline.cleaner import (
    AnimeClassifier,
    QualityFilter,
    AIDetector,
    CharacterCropper,
    MultiTagger
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

        # 延迟初始化各个模块（只在需要时加载）
        self._deduplicator = None
        self._detector = None
        self._confidence_filter = None
        self._reviewer = None
        self._trainer = None
        self._anime_classifier = None
        self._quality_filter = None
        self._ai_detector = None
        self._character_cropper = None
        self._multi_tagger = None

        # 统计信息
        self.stats = {
            'start_time': None,
            'end_time': None,
            'total_samples': 0,
            'imported_samples': 0,
            'deduplicated_samples': 0,
            'cleaned_samples': 0,
            'annotated_samples': 0,
            'difficult_samples': 0,
            'errors': []
        }

        # 性能监控
        self.performance_metrics = {}

        logger.info("✅ 数据流水线初始化完成")
        print("✅ 数据流水线初始化完成")

    def retry_on_error(self, max_retries=3, delay=1.0, exceptions=(Exception,)):
        """
        重试装饰器

        Args:
            max_retries: 最大重试次数
            delay: 重试间隔（秒）
            exceptions: 需要重试的异常类型
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                last_exception = None
                for attempt in range(max_retries):
                    try:
                        return func(*args, **kwargs)
                    except exceptions as e:
                        last_exception = e
                        logger.warning(f"{func.__name__} 第 {attempt + 1} 次尝试失败: {e}")
                        if attempt < max_retries - 1:
                            time.sleep(delay * (attempt + 1))  # 指数退避
                logger.error(f"{func.__name__} 在 {max_retries} 次尝试后失败: {last_exception}")
                raise last_exception
            return wrapper
        return decorator

    def monitor_performance(self, func_name: str):
        """性能监控装饰器"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    elapsed = time.time() - start_time
                    self.performance_metrics[func_name] = {
                        'elapsed_time': elapsed,
                        'status': 'success',
                        'timestamp': datetime.now()
                    }
                    logger.info(f"{func_name} 完成，耗时: {elapsed:.2f}秒")
                    return result
                except Exception as e:
                    elapsed = time.time() - start_time
                    self.performance_metrics[func_name] = {
                        'elapsed_time': elapsed,
                        'status': 'failed',
                        'error': str(e),
                        'timestamp': datetime.now()
                    }
                    logger.error(f"{func_name} 失败，耗时: {elapsed:.2f}秒，错误: {e}")
                    raise
            return wrapper
        return decorator

    @property
    def deduplicator(self):
        if self._deduplicator is None:
            self._deduplicator = CLIPDeduplicator(model_name="ViT-B/32")
        return self._deduplicator

    @property
    def detector(self):
        if self._detector is None:
            self._detector = YOLODetector(model_path="yolov8n.pt")
        return self._detector

    @property
    def confidence_filter(self):
        if self._confidence_filter is None:
            self._confidence_filter = ConfidenceFilter(threshold=0.7)
        return self._confidence_filter

    @property
    def reviewer(self):
        if self._reviewer is None:
            self._reviewer = SampleReviewer(review_dir="data/review_batches")
        return self._reviewer

    @property
    def trainer(self):
        if self._trainer is None:
            self._trainer = IncrementalTrainer(model_dir="data/models", data_dir="data/training")
        return self._trainer

    @property
    def anime_classifier(self):
        if self._anime_classifier is None:
            self._anime_classifier = AnimeClassifier()
        return self._anime_classifier

    @property
    def quality_filter(self):
        if self._quality_filter is None:
            self._quality_filter = QualityFilter()
        return self._quality_filter

    @property
    def ai_detector(self):
        if self._ai_detector is None:
            self._ai_detector = AIDetector()
        return self._ai_detector

    @property
    def character_cropper(self):
        if self._character_cropper is None:
            self._character_cropper = CharacterCropper()
        return self._character_cropper

    @property
    def multi_tagger(self):
        if self._multi_tagger is None:
            self._multi_tagger = MultiTagger()
        return self._multi_tagger

    def _process_single_sample(self, sample_id: int, image_path: str, min_confidence: float = 0.5) -> Tuple[str, int, Dict]:
        """
        处理单个样本（用于并行处理） - 不使用数据库会话

        Returns:
            (status, sample_id, result_dict)
        """
        try:
            # 1. 质量过滤
            quality_ok, quality_info = self.quality_filter.filter(image_path)
            if not quality_ok:
                return 'filtered_quality', sample_id, {}

            # 2. 动漫分类
            anime_prob, anime_result = self.anime_classifier.classify(image_path)
            if anime_result != 'anime' or anime_prob < min_confidence:
                return 'filtered_non_anime', sample_id, {}

            # 3. AI检测
            ai_prob, ai_result = self.ai_detector.detect(image_path)

            # 4. 生成标签
            tags = self.multi_tagger.generate_comprehensive_tags(image_path)

            return 'cleaned', sample_id, {
                'is_ai_generated': (ai_result == 'ai-generated'),
                'anime_confidence': anime_prob,
                'ai_confidence': ai_prob,
                'attributes': tags.get('by_category', {})
            }

        except Exception as e:
            self.stats['errors'].append(f"样本 {sample_id} 处理失败: {e}")
            return 'error', sample_id, {}

    def clean_samples(self, min_confidence: float = 0.5, max_workers: int = 4) -> int:
        """
        数据清洗：过滤低质量、非动漫和AI生成的图片

        Args:
            min_confidence: 动漫分类最低置信度
            max_workers: 并行处理线程数

        Returns:
            清洗后的样本数量
        """
        print(f"\n🧹 开始数据清洗...")

        # 获取待清洗的样本（已去重但未清洗）
        samples = self.session.query(Sample).filter(
            Sample.status == 'deduplicated'
        ).all()

        if not samples:
            print("⚠️ 没有待清洗的样本")
            return 0

        cleaned_count = 0
        filtered_count = 0

        # 初始化清洗模块
        self.anime_classifier.initialize()
        self.ai_detector.initialize()
        self.multi_tagger.initialize()

        # 使用线程池并行处理 - 只传sample_id和image_path，不传session对象
        print(f"   使用 {max_workers} 个线程并行处理...")
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_sample = {
                executor.submit(self._process_single_sample, sample.id, sample.image_path, min_confidence): sample.id
                for sample in samples
            }

            # 收集结果
            for future in as_completed(future_to_sample):
                try:
                    result = future.result()
                    if result is not None and isinstance(result, tuple) and len(result) == 3:
                        results.append(result)
                except Exception as e:
                    sample_id = future_to_sample.get(future)
                    self.stats['errors'].append(f"收集结果失败 (样本 {sample_id}): {e}")
                
                if len(results) % 100 == 0:
                    print(f"   已处理 {len(results)} 个样本...")

        # 在主线程统一更新数据库（避免多线程会话问题）
        print(f"\n📝 更新数据库...")
        for item in results:
            # 安全解包
            if not isinstance(item, tuple) or len(item) != 3:
                continue
            status, sample_id, result_dict = item
            sample = self.session.query(Sample).get(sample_id)
            if not sample:
                continue
                
            if status == 'cleaned':
                sample.is_ai_generated = result_dict.get('is_ai_generated', False)
                sample.anime_confidence = result_dict.get('anime_confidence', 0.0)
                sample.ai_confidence = result_dict.get('ai_confidence', 0.0)
                sample.attributes = result_dict.get('attributes', {})
                sample.status = 'cleaned'
                cleaned_count += 1
            elif status.startswith('filtered'):
                sample.status = status
                filtered_count += 1
            
            # 每100条提交一次
            if (cleaned_count + filtered_count) % 100 == 0:
                self.session.commit()

        # 提交剩余数据
        self.session.commit()

        print(f"✅ 清洗完成！保留: {cleaned_count} 个，过滤: {filtered_count} 个")

        self.stats['cleaned_samples'] = cleaned_count
        return cleaned_count

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

            # 从目录名提取角色名（如 koharu_(blue_archive) -> koharu）
            if '_(' in character_name:
                character_name = character_name.rsplit('_(' , 1)[0]

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

    def _annotate_single_sample(self, sample_id: int, image_path: str, conf_threshold: float = 0.5) -> Tuple[str, int, Optional[Dict]]:
        """
        标注单个样本（用于并行处理） - 不使用数据库会话

        Returns:
            (status, sample_id, detection_result)
        """
        try:
            # 执行检测
            detections = self.detector.detect(image_path, conf_threshold=conf_threshold)

            if detections:
                # 使用置信度最高的检测结果
                best_detection = max(detections, key=lambda x: x['confidence'])
                return 'annotated', sample_id, {
                    'detections': detections,
                    'best': best_detection
                }
            else:
                return 'no_detection', sample_id, None

        except Exception as e:
            self.stats['errors'].append(f"标注样本 {sample_id} 失败: {e}")
            return 'error', sample_id, None

    def annotate_samples(self, conf_threshold: float = 0.5, limit: int = None, max_workers: int = 4) -> int:
        """
        自动标注样本

        Args:
            conf_threshold: 置信度阈值
            limit: 处理数量限制
            max_workers: 并行处理线程数

        Returns:
            标注的样本数量
        """
        print(f"\n📝 开始自动标注...")

        # 获取待标注的样本（已清洗的样本）
        query = self.session.query(Sample).filter(Sample.status == 'cleaned')
        if limit:
            query = query.limit(limit)

        samples = query.all()
        print(f"📋 找到 {len(samples)} 个待标注样本")

        # 先过滤已标注的样本
        existing_sample_ids = {
            a.sample_id for a in self.session.query(Annotation.sample_id).all()
        }
        unannotated_samples = [s for s in samples if s.id not in existing_sample_ids]
        print(f"   跳过已标注 {len(samples) - len(unannotated_samples)} 个")

        annotated_count = 0
        skipped_count = 0
        no_detection_count = 0

        # 使用线程池并行处理 - 只传sample_id和image_path
        print(f"   使用 {max_workers} 个线程并行处理...")
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_sample = {
                executor.submit(self._annotate_single_sample, sample.id, sample.image_path, conf_threshold): sample.id
                for sample in unannotated_samples
            }

            # 收集结果
            for future in as_completed(future_to_sample):
                try:
                    result = future.result()
                    if result is not None and isinstance(result, tuple) and len(result) == 3:
                        results.append(result)
                except Exception as e:
                    sample_id = future_to_sample.get(future)
                    self.stats['errors'].append(f"收集标注结果失败 (样本 {sample_id}): {e}")
                
                if len(results) % 50 == 0:
                    print(f"   已处理 {len(results)} 个样本...")

        # 在主线程统一更新数据库
        print(f"\n📝 更新数据库...")
        for item in results:
            # 安全解包
            if not isinstance(item, tuple) or len(item) != 3:
                continue
            status, sample_id, result = item
            sample = self.session.query(Sample).get(sample_id)
            if not sample:
                continue
                
            if status == 'annotated' and result:
                # 创建标注记录
                annotation = Annotation(
                    sample_id=sample.id,
                    annotator='auto',
                    bbox=result['best']['bbox'],
                    confidence=result['best']['confidence'],
                    is_verified=False
                )
                self.session.add(annotation)

                # 更新样本信息
                sample.person_count = len(result['detections'])
                sample.confidence = result['best']['confidence']
                sample.status = 'annotated'
                annotated_count += 1
            elif status == 'no_detection':
                sample.status = 'no_detection'
                no_detection_count += 1
            elif status == 'skipped':
                skipped_count += 1
            
            # 每50条提交一次
            if (annotated_count + no_detection_count + skipped_count) % 50 == 0:
                self.session.commit()

        # 提交剩余数据
        self.session.commit()

        print(f"✅ 标注完成！已标注: {annotated_count} 个，未检测到: {no_detection_count} 个")

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

            # 3. 数据清洗
            self.clean_samples()

            # 4. 自动标注
            self.annotate_samples()

            # 5. 筛选困难样本
            difficult_samples = self.filter_difficult_samples(batch_size=20)

            # 6. 创建审核批次
            if difficult_samples:
                self.create_review_batch(difficult_samples)

            # 7. 自动审核（可选）
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
        print(f"清洗后样本: {self.stats['cleaned_samples']}")
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
