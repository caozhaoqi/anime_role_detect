#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WebUI工具模块 - 提供共享功能
"""

# 必须在导入任何其他模块之前设置环境变量
import os
import sys
import platform
from pathlib import Path

# Mac平台禁用CUDA，避免mutex错误
if platform.system() == "Darwin":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    # 强制禁用CUDA
    os.environ["FORCE_CPU"] = "1"

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import json
from PIL import Image, ImageDraw

# 全局状态
DB_AVAILABLE = False
PIPELINE_AVAILABLE = False


def get_db_modules():
    """延迟导入数据库模块"""
    global DB_AVAILABLE
    if DB_AVAILABLE:
        return True
    try:
        # 直接导入数据库模块，避免触发PyTorch
        import importlib.util
        
        # 先导入SQLAlchemy相关
        spec = importlib.util.spec_from_file_location(
            "sqlalchemy_types", 
            str(project_root / "src" / "data_pipeline" / "database" / "init_db.py")
        )
        init_db_module = importlib.util.module_from_spec(spec)
        
        # 设置环境变量，确保PyTorch不会初始化CUDA
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["FORCE_CPU"] = "1"
        
        spec.loader.exec_module(init_db_module)
        
        globals()['Character'] = init_db_module.Character
        globals()['Sample'] = init_db_module.Sample
        globals()['Annotation'] = init_db_module.Annotation
        globals()['init_database'] = init_db_module.init_database
        DB_AVAILABLE = True
        return True
    except Exception as e:
        import streamlit as st
        st.error(f"数据库模块加载失败: {e}")
        return False


def get_pipeline_module():
    """延迟导入Pipeline模块"""
    global PIPELINE_AVAILABLE
    if PIPELINE_AVAILABLE:
        return True
    try:
        # 设置环境变量，确保PyTorch不会初始化CUDA
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["FORCE_CPU"] = "1"
        
        from src.data_pipeline.pipeline import DataPipeline
        globals()['DataPipeline'] = DataPipeline
        PIPELINE_AVAILABLE = True
        return True
    except Exception as e:
        import streamlit as st
        st.error(f"Pipeline模块加载失败: {e}")
        return False


def load_stats():
    """加载统计信息"""
    stats_path = "data/annotation_stats.json"
    if os.path.exists(stats_path):
        with open(stats_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def get_db_stats_optimized():
    """优化的数据库统计查询"""
    if not get_db_modules():
        return {
            'total_characters': 0,
            'total_samples': 0,
            'total_annotations': 0,
            'status_counts': {},
            'verified_count': 0
        }
    
    try:
        from sqlalchemy import func
        engine, Session = init_database()
        session = Session()
        
        total_characters = session.query(Character).count()
        total_samples = session.query(Sample).count()
        total_annotations = session.query(Annotation).count()
        verified_count = session.query(Annotation).filter(Annotation.is_verified == True).count()
        
        status_counts = {}
        results = session.query(Sample.status, func.count(Sample.status)).group_by(Sample.status).all()
        for status, count in results:
            status_counts[status] = count
        
        session.close()
        engine.dispose()
        
        return {
            'total_characters': total_characters,
            'total_samples': total_samples,
            'total_annotations': total_annotations,
            'status_counts': status_counts,
            'verified_count': verified_count
        }
    except Exception as e:
        import streamlit as st
        st.error(f"获取数据库统计失败: {e}")
        return {
            'total_characters': 0,
            'total_samples': 0,
            'total_annotations': 0,
            'status_counts': {},
            'verified_count': 0
        }


def draw_bbox(image_path, annotations):
    """在图像上绘制BBox，包含角色名和置信度"""
    try:
        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        
        # 使用更清晰的字体
        from PIL import ImageFont
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 14)
        except:
            font = ImageFont.load_default()
        
        for ann in annotations:
            bbox = ann.bbox
            if bbox and len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                
                # 绘制边界框（脸部框选）
                draw.rectangle([x1, y1, x2, y2], outline="#FF6B6B", width=3)
                
                # 准备标签文本
                label_parts = []
                
                # 添加角色名
                if hasattr(ann, 'character_name') and ann.character_name:
                    label_parts.append(f"{ann.character_name}")
                
                # 添加检测置信度
                if hasattr(ann, 'confidence') and ann.confidence:
                    label_parts.append(f"detect:{ann.confidence:.2f}")
                
                # 添加角色识别置信度
                if hasattr(ann, 'character_confidence') and ann.character_confidence:
                    label_parts.append(f"char:{ann.character_confidence:.2f}")
                
                # 绘制标签
                if label_parts:
                    label_text = " | ".join(label_parts)
                    
                    # 计算文本尺寸
                    try:
                        text_bbox = draw.textbbox((0, 0), label_text, font=font)
                        text_width = text_bbox[2] - text_bbox[0]
                        text_height = text_bbox[3] - text_bbox[1]
                    except:
                        text_width, text_height = 100, 20
                    
                    # 确定标签位置（在边界框上方，超出则显示在下方）
                    label_x = x1
                    label_y = y1 - text_height - 5
                    
                    if label_y < 5:
                        label_y = y2 + 5
                    
                    # 绘制半透明背景
                    bg_padding = 5
                    draw.rectangle(
                        [label_x - bg_padding, label_y - bg_padding, 
                         label_x + text_width + bg_padding, label_y + text_height + bg_padding],
                        fill=(0, 0, 0, 180)  # 半透明黑色
                    )
                    
                    # 绘制文本
                    draw.text((label_x, label_y), label_text, fill="#FFFFFF", font=font)
        
        return img
    except Exception as e:
        import traceback
        print(f"⚠️ 绘制BBox失败: {e}")
        print(traceback.format_exc())
        return None
