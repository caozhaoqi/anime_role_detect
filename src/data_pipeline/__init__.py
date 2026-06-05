"""
数据流水线模块
Data Pipeline Module

提供完整的数据采集、清洗、标注流水线功能
"""

__version__ = "1.0.0"
__author__ = "ARD Team"

# 只导入数据库模块（不触发PyTorch）
from .database.init_db import init_database, Character, Sample, CollectionTask, Annotation

# 其他模块延迟导入，避免触发PyTorch CUDA初始化
# 使用时请单独导入：
# from src.data_pipeline.collector.deduplication import CLIPDeduplicator
# from src.data_pipeline.annotator.yolo_detector import YOLODetector
# from src.data_pipeline.active_learning.confidence_filter import ConfidenceFilter
# from src.data_pipeline.pipeline import DataPipeline

__all__ = [
    "init_database",
    "Character",
    "Sample",
    "CollectionTask",
    "Annotation",
    # 延迟导入的模块
    "CLIPDeduplicator",
    "YOLODetector",
    "ConfidenceFilter",
    "DataPipeline"
]


def __getattr__(name):
    """延迟导入模块"""
    if name == "CLIPDeduplicator":
        from .collector.deduplication import CLIPDeduplicator
        return CLIPDeduplicator
    elif name == "YOLODetector":
        from .annotator.yolo_detector import YOLODetector
        return YOLODetector
    elif name == "ConfidenceFilter":
        from .active_learning.confidence_filter import ConfidenceFilter
        return ConfidenceFilter
    elif name == "DataPipeline":
        from .pipeline import DataPipeline
        return DataPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")