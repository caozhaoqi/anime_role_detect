#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline控制页面（异步执行）
"""

import streamlit as st
import threading
import time
import subprocess
from src.data_pipeline.webui.utils import get_db_stats_optimized


# 全局状态
pipeline_status = "idle"
pipeline_progress = 0
pipeline_log = []


def run_pipeline_script():
    """运行Pipeline脚本（子进程方式，避免CUDA问题）"""
    global pipeline_status, pipeline_progress, pipeline_log
    
    try:
        pipeline_status = "running"
        pipeline_progress = 0
        pipeline_log = ["开始执行数据流水线..."]
        
        # 使用子进程运行Pipeline脚本
        process = subprocess.Popen(
            ["python3", "-m", "src.data_pipeline.pipeline"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd="/Users/caozhaoqi/PycharmProjects/anime_role_detect"
        )
        
        # 读取输出
        for line in process.stdout:
            pipeline_log.append(line.strip())
            if len(pipeline_log) > 100:
                pipeline_log = pipeline_log[-100:]
        
        process.wait()
        
        if process.returncode == 0:
            pipeline_progress = 100
            pipeline_log.append(f"[{time.strftime('%H:%M:%S')}] 🏁 数据流水线执行完成！")
            pipeline_status = "completed"
        else:
            pipeline_log.append(f"[{time.strftime('%H:%M:%S')}] ❌ 执行失败，返回码: {process.returncode}")
            pipeline_status = "error"
        
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
                threading.Thread(target=run_pipeline_script, daemon=True).start()
        
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
