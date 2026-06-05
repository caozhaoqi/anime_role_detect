#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline控制页面（异步执行）
"""

# 必须在导入任何其他模块之前设置环境变量
import os
import platform

# Mac平台禁用CUDA，避免mutex错误
if platform.system() == "Darwin":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    os.environ["FORCE_CPU"] = "1"

import streamlit as st
import threading
import time
from src.data_pipeline.webui.utils import get_pipeline_module, get_db_stats_optimized


# 全局状态
pipeline_status = "idle"
pipeline_progress = 0
pipeline_log = []


def run_pipeline_async():
    """异步运行Pipeline"""
    global pipeline_status, pipeline_progress, pipeline_log
    
    try:
        pipeline_status = "running"
        pipeline_progress = 0
        pipeline_log = ["开始执行数据流水线..."]
        
        # 确保Pipeline模块已加载
        if not get_pipeline_module():
            pipeline_log.append("❌ Pipeline模块加载失败")
            pipeline_status = "error"
            return
        
        # 从utils模块获取DataPipeline类
        from src.data_pipeline.webui.utils import DataPipeline
        pipeline = DataPipeline()
        
        steps = [
            ("导入样本", pipeline.import_samples),
            ("去重处理", pipeline.deduplicate),
            ("数据清洗", pipeline.clean),
            ("自动标注", pipeline.annotate),
            ("困难样本筛选", pipeline.filter_difficult_samples)
        ]
        
        for i, (step_name, step_func) in enumerate(steps):
            pipeline_log.append(f"[{time.strftime('%H:%M:%S')}] {step_name}...")
            pipeline_progress = int((i + 1) / len(steps) * 30)
            
            try:
                step_func()
                pipeline_log.append(f"[{time.strftime('%H:%M:%S')}] ✅ {step_name}完成")
            except Exception as e:
                pipeline_log.append(f"[{time.strftime('%H:%M:%S')}] ❌ {step_name}失败: {e}")
        
        pipeline_progress = 100
        pipeline_log.append(f"[{time.strftime('%H:%M:%S')}] 🏁 数据流水线执行完成！")
        pipeline_status = "completed"
        
    except Exception as e:
        pipeline_log.append(f"[{time.strftime('%H:%M:%S')}] 💥 执行出错: {e}")
        pipeline_status = "error"


def display_pipeline():
    """显示Pipeline控制页面"""
    global pipeline_status, pipeline_progress, pipeline_log
    
    st.title("⚙️ Pipeline控制")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("当前状态")
        status_color = {
            "idle": "gray",
            "running": "green",
            "completed": "blue",
            "error": "red"
        }
        st.markdown(f"**状态:** <span style='color:{status_color[pipeline_status]}'>{pipeline_status}</span>", unsafe_allow_html=True)
        
        if pipeline_status == "running":
            st.progress(pipeline_progress)
    
    with col2:
        st.subheader("快捷操作")
        
        if pipeline_status != "running":
            if st.button("🚀 运行完整Pipeline"):
                threading.Thread(target=run_pipeline_async, daemon=True).start()
        
        if st.button("🔄 刷新状态"):
            st.rerun()
    
    st.subheader("📋 执行日志")
    log_container = st.container()
    
    if pipeline_log:
        for log in pipeline_log[-50:]:
            log_container.write(log)
    else:
        log_container.info("等待执行...")
    
    st.subheader("📊 数据统计")
    stats = get_db_stats_optimized()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("样本数", stats['total_samples'])
    with col2:
        st.metric("标注数", stats['total_annotations'])
    with col3:
        st.metric("已审核", stats['verified_count'])
