#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
标注统计页面（含热力图）
"""

import streamlit as st
import plotly.express as px
import pandas as pd
from src.data_pipeline.webui.utils import get_db_modules, load_stats


def display_annotations():
    """显示标注统计页面"""
    st.title("📊 标注统计")
    
    stats = load_stats()
    
    st.subheader("📈 置信度分布")
    conf_stats = stats.get('confidence_stats', {})
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("平均置信度", f"{conf_stats.get('avg', 0):.2f}")
    with col2:
        st.metric("最低置信度", f"{conf_stats.get('min', 0):.2f}")
    with col3:
        st.metric("最高置信度", f"{conf_stats.get('max', 0):.2f}")
    with col4:
        st.metric("标准差", f"{conf_stats.get('std', 0):.2f}")
    
    st.subheader("📍 目标中心分布热力图")
    center_x_list = stats.get('center_x_list', [])
    center_y_list = stats.get('center_y_list', [])
    
    if center_x_list and center_y_list:
        df = pd.DataFrame({
            'center_x': center_x_list,
            'center_y': center_y_list
        })
        
        fig = px.density_heatmap(
            df,
            x="center_x",
            y="center_y",
            title="目标中心分布",
            labels={'center_x': 'X坐标', 'center_y': 'Y坐标'},
            width=700,
            height=500
        )
        st.plotly_chart(fig)
        
        avg_center_x = stats.get('avg_center_x', 0)
        avg_center_y = stats.get('avg_center_y', 0)
        st.write(f"平均中心位置: ({avg_center_x:.2f}, {avg_center_y:.2f})")
        
        if avg_center_x > 0.4 and avg_center_x < 0.6 and avg_center_y > 0.4 and avg_center_y < 0.6:
            st.warning("⚠️ 检测到明显的中心偏置（Center Bias）")
    
    st.subheader("📦 BBox面积统计")
    area_stats = stats.get('area_stats', {})
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("平均面积比", f"{area_stats.get('avg', 0):.4f}")
    with col2:
        st.metric("最小面积比", f"{area_stats.get('min', 0):.4f}")
    with col3:
        st.metric("最大面积比", f"{area_stats.get('max', 0):.4f}")
    
    st.subheader("👥 多人检测统计")
    multi_stats = stats.get('multi_person_stats', {})
    st.write(f"多人样本数: {multi_stats.get('multi_person_count', 0)}")
    st.write(f"单人样本数: {multi_stats.get('single_person_count', 0)}")
    
    if multi_stats.get('total_count', 0) > 0:
        multi_ratio = multi_stats.get('multi_person_count', 0) / multi_stats.get('total_count', 1)
        st.write(f"多人占比: {multi_ratio:.1%}")
