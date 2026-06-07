#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WebUI模块 - 提供数据流水线管理控制台的页面组件
"""

__version__ = "1.0.0"

# 导入所有页面模块
from .pages.overview import display_overview
from .pages.characters import display_characters
from .pages.samples import display_samples
from .pages.annotation_review import display_annotation_review
from .pages.difficult_samples import display_difficult_samples
from .pages.pipeline import display_pipeline
from .pages.annotations import display_annotations
from .pages.data_quality import display_data_quality
from .pages.log_viewer import display_log_viewer
from .pages.data_export import display_data_export

__all__ = [
    "display_overview",
    "display_characters", 
    "display_samples",
    "display_annotation_review",
    "display_difficult_samples",
    "display_pipeline",
    "display_annotations",
    "display_data_quality",
    "display_log_viewer",
    "display_data_export"
]
