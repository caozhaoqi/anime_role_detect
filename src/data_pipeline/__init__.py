"""
数据流水线模块
Data Pipeline Module

提供完整的数据采集、清洗、标注流水线功能
"""

__version__ = "1.0.0"
__author__ = "ARD Team"

# 数据库模块
from .database.init_db import init_database, Character, Sample, CollectionTask

# 采集模块
from .collector.deduplication import CLIPDeduplicator

# 标注模块
from .annotator.yolo_detector import YOLODetector

# 主动学习模块
from .active_learning.confidence_filter import ConfidenceFilter, SampleReviewer, IncrementalTrainer

# 流水线主类
from .pipeline import DataPipeline

__all__ = [
    "init_database",
    "Character",
    "Sample",
    "CollectionTask",
    "CLIPDeduplicator",
    "YOLODetector",
    "ConfidenceFilter",
    "SampleReviewer",
    "IncrementalTrainer",
    "DataPipeline"
]