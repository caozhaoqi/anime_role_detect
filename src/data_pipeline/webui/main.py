#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit Web界面入口 - 动漫角色检测数据集生产平台
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
    os.environ["FORCE_CPU"] = "1"

project_root = Path(__file__).parent.parent.parent.parent

import streamlit as st

# 导入模块化页面
from src.data_pipeline.webui.pages.overview import display_overview
from src.data_pipeline.webui.pages.characters import display_characters
from src.data_pipeline.webui.pages.samples import display_samples
from src.data_pipeline.webui.pages.annotation_review import display_annotation_review
from src.data_pipeline.webui.pages.difficult_samples import display_difficult_samples
from src.data_pipeline.webui.pages.pipeline import display_pipeline
from src.data_pipeline.webui.pages.annotations import display_annotations
from src.data_pipeline.webui.pages.data_quality import display_data_quality
from src.data_pipeline.webui.pages.log_viewer import display_log_viewer
from src.data_pipeline.webui.pages.data_export import display_data_export


def main():
    """主函数"""
    st.set_page_config(
        page_title="动漫角色识别系统",
        page_icon="🎬",
        layout="wide"
    )
    
    st.sidebar.title("导航")
    menu_options = [
        "概览",
        "角色管理",
        "样本管理",
        "✅ 标注审核",
        "🎯 困难样本",
        "流水线控制",
        "标注统计",
        "📊 数据质量",
        "📝 日志查看",
        "📤 数据导出"
    ]
    selected_menu = st.sidebar.radio("选择功能", menu_options)
    
    if selected_menu == "概览":
        display_overview()
    elif selected_menu == "角色管理":
        display_characters()
    elif selected_menu == "样本管理":
        display_samples()
    elif selected_menu == "✅ 标注审核":
        display_annotation_review()
    elif selected_menu == "🎯 困难样本":
        display_difficult_samples()
    elif selected_menu == "流水线控制":
        display_pipeline()
    elif selected_menu == "标注统计":
        display_annotations()
    elif selected_menu == "📊 数据质量":
        display_data_quality()
    elif selected_menu == "📝 日志查看":
        display_log_viewer()
    elif selected_menu == "📤 数据导出":
        display_data_export()


if __name__ == "__main__":
    main()
