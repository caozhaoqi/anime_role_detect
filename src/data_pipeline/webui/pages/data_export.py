#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据导出页面
"""

import streamlit as st
import os
import json
from pathlib import Path
from src.data_pipeline.webui.utils import get_db_modules


def export_yolo_format(output_dir="dataset/yolo"):
    """导出YOLO格式数据"""
    if not get_db_modules():
        return False, "数据库不可用"
    
    try:
        from src.data_pipeline.webui.utils import init_database, Sample, Annotation
        engine, Session = init_database()
        session = Session()
        
        os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)
        
        samples = session.query(Sample).filter(Sample.status == 'annotated').all()
        
        for sample in samples:
            annotations = session.query(Annotation).filter(Annotation.sample_id == sample.id).all()
            
            img_path = Path(sample.image_path)
            label_path = os.path.join(output_dir, "labels", img_path.stem + ".txt")
            
            with open(label_path, 'w') as f:
                for ann in annotations:
                    if ann.bbox and len(ann.bbox) == 4:
                        x1, y1, x2, y2 = ann.bbox
                        width = x2 - x1
                        height = y2 - y1
                        center_x = (x1 + width / 2) / sample.width if sample.width else 0.5
                        center_y = (y1 + height / 2) / sample.height if sample.height else 0.5
                        width_norm = width / sample.width if sample.width else 0
                        height_norm = height / sample.height if sample.height else 0
                        f.write(f"0 {center_x} {center_y} {width_norm} {height_norm}\n")
        
        session.close()
        engine.dispose()
        
        return True, f"成功导出 {len(samples)} 条样本到 {output_dir}"
    
    except Exception as e:
        return False, f"导出失败: {e}"


def export_coco_format(output_dir="dataset/coco"):
    """导出COCO格式数据"""
    if not get_db_modules():
        return False, "数据库不可用"
    
    try:
        from src.data_pipeline.webui.utils import init_database, Sample, Annotation
        engine, Session = init_database()
        session = Session()
        
        os.makedirs(output_dir, exist_ok=True)
        
        images = []
        annotations = []
        annotation_id = 1
        
        samples = session.query(Sample).filter(Sample.status == 'annotated').all()
        
        for img_id, sample in enumerate(samples, 1):
            img_path = Path(sample.image_path)
            images.append({
                "id": img_id,
                "file_name": img_path.name,
                "width": sample.width or 640,
                "height": sample.height or 640
            })
            
            anns = session.query(Annotation).filter(Annotation.sample_id == sample.id).all()
            for ann in anns:
                if ann.bbox and len(ann.bbox) == 4:
                    x1, y1, x2, y2 = ann.bbox
                    width = x2 - x1
                    height = y2 - y1
                    annotations.append({
                        "id": annotation_id,
                        "image_id": img_id,
                        "category_id": 1,
                        "bbox": [x1, y1, width, height],
                        "area": width * height,
                        "iscrowd": 0
                    })
                    annotation_id += 1
        
        coco_data = {
            "images": images,
            "annotations": annotations,
            "categories": [{"id": 1, "name": "anime_character", "supercategory": "character"}]
        }
        
        with open(os.path.join(output_dir, "annotations.json"), 'w') as f:
            json.dump(coco_data, f)
        
        session.close()
        engine.dispose()
        
        return True, f"成功导出 {len(samples)} 条样本到 {output_dir}"
    
    except Exception as e:
        return False, f"导出失败: {e}"


def display_data_export():
    """显示数据导出页面"""
    st.title("📤 数据导出")
    
    st.subheader("导出格式")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**YOLO格式**")
        st.write("适合YOLO系列模型训练")
        if st.button("🚀 导出YOLO格式"):
            success, msg = export_yolo_format()
            if success:
                st.success(msg)
            else:
                st.error(msg)
    
    with col2:
        st.markdown("**COCO格式**")
        st.write("适合COCO风格模型训练")
        if st.button("🚀 导出COCO格式"):
            success, msg = export_coco_format()
            if success:
                st.success(msg)
            else:
                st.error(msg)
    
    st.subheader("📁 导出目录")
    st.write("数据集将导出到 `dataset/` 目录")
    if os.path.exists("dataset"):
        for root, dirs, files in os.walk("dataset"):
            level = root.replace("dataset", "").count(os.sep)
            indent = " " * 2 * level
            st.write(f"{indent}{os.path.basename(root)}/")
            subindent = " " * 2 * (level + 1)
            for file in files[:5]:
                st.write(f"{subindent}{file}")
            if len(files) > 5:
                st.write(f"{subindent}...")
