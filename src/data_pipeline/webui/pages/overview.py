#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
概览页面
"""

import streamlit as st
import pandas as pd
from src.data_pipeline.webui.utils import get_db_stats_optimized, load_stats


def display_overview():
    """显示概览页面"""
    st.title("🎬 Anime Role detect")
    st.subheader("Data Pipeline Management Console")
    
    db_stats = get_db_stats_optimized()
    annot_stats = load_stats()
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("👥 角色数量", db_stats['total_characters'])
    
    with col2:
        st.metric("🖼️ 样本数量", db_stats['total_samples'])
    
    with col3:
        st.metric("🏷️ 标注数量", db_stats['total_annotations'])
    
    with col4:
        avg_conf = annot_stats.get('confidence_stats', {}).get('avg', 0)
        st.metric("📊 平均置信度", f"{avg_conf:.2f}")
    
    with col5:
        st.metric("✅ 已审核", db_stats['verified_count'])
    
    st.subheader("📈 样本状态分布")
    status_counts = db_stats['status_counts']
    if status_counts:
        status_df = pd.DataFrame(list(status_counts.items()), columns=['状态', '数量'])
        status_df['占比'] = (status_df['数量'] / status_df['数量'].sum() * 100).round(1)
        st.dataframe(status_df, width='stretch')
        
        status_descriptions = {
            'pending': '待处理',
            'deduplicated': '已去重',
            'cleaned': '已清洗',
            'annotated': '已标注',
            'no_detection': '未检测到',
            'filtered_quality': '质量过滤',
            'filtered_non_anime': '非动漫过滤',
            'duplicate': '重复'
        }
        st.markdown("**状态说明:**")
        for status, desc in status_descriptions.items():
            if status in status_counts:
                st.write(f"- `{status}`: {desc}")
    else:
        st.info("暂无样本数据")
