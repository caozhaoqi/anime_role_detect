#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据质量评分页面
"""

import streamlit as st
from src.data_pipeline.webui.utils import get_db_stats_optimized, load_stats


def calculate_quality_score(db_stats, annot_stats):
    """计算数据质量评分"""
    scores = []
    
    # 数据量评分 (0-10)
    total_samples = db_stats['total_samples']
    if total_samples >= 10000:
        data_score = 10
    elif total_samples >= 5000:
        data_score = 8
    elif total_samples >= 1000:
        data_score = 6
    elif total_samples >= 100:
        data_score = 4
    else:
        data_score = 2
    scores.append(('数据量', data_score, f'{total_samples} 条'))
    
    # 标注覆盖率评分 (0-10)
    total_annotations = db_stats['total_annotations']
    if total_annotations >= total_samples:
        coverage_score = 10
    elif total_annotations >= total_samples * 0.8:
        coverage_score = 8
    elif total_annotations >= total_samples * 0.5:
        coverage_score = 6
    else:
        coverage_score = 3
    scores.append(('标注覆盖率', coverage_score, f'{total_annotations} 个'))
    
    # 审核率评分 (0-10)
    verified_count = db_stats['verified_count']
    if total_annotations > 0:
        verified_ratio = verified_count / total_annotations
        if verified_ratio >= 0.8:
            verify_score = 10
        elif verified_ratio >= 0.5:
            verify_score = 7
        elif verified_ratio >= 0.1:
            verify_score = 4
        else:
            verify_score = 1
    else:
        verify_score = 1
    scores.append(('审核率', verify_score, f'{verified_ratio:.1%}'))
    
    # 置信度评分 (0-10)
    avg_conf = annot_stats.get('confidence_stats', {}).get('avg', 0)
    if avg_conf >= 0.9:
        conf_score = 10
    elif avg_conf >= 0.7:
        conf_score = 8
    elif avg_conf >= 0.5:
        conf_score = 5
    else:
        conf_score = 3
    scores.append(('置信度', conf_score, f'{avg_conf:.2f}'))
    
    # 位置均匀度评分 (0-10)
    avg_center_x = annot_stats.get('avg_center_x', 0.5)
    avg_center_y = annot_stats.get('avg_center_y', 0.5)
    center_bias = abs(avg_center_x - 0.5) + abs(avg_center_y - 0.5)
    if center_bias < 0.1:
        position_score = 10
    elif center_bias < 0.2:
        position_score = 7
    elif center_bias < 0.3:
        position_score = 4
    else:
        position_score = 2
    scores.append(('位置均匀度', position_score, f'偏置: {center_bias:.2f}'))
    
    # 综合评分
    weights = [0.25, 0.25, 0.20, 0.20, 0.10]
    total_score = sum(s * w for (_, s, _), w in zip(scores, weights))
    
    return scores, total_score


def display_data_quality():
    """显示数据质量评分页面"""
    st.title("🎯 数据质量评分")
    
    db_stats = get_db_stats_optimized()
    annot_stats = load_stats()
    
    scores, total_score = calculate_quality_score(db_stats, annot_stats)
    
    st.subheader("📊 综合评分")
    st.markdown(f"<h1 style='text-align: center; color: #4CAF50;'>{total_score:.1f} / 10</h1>", unsafe_allow_html=True)
    
    if total_score >= 8:
        st.success("🎉 数据质量优秀！")
    elif total_score >= 6:
        st.info("👍 数据质量良好，建议继续优化")
    elif total_score >= 4:
        st.warning("⚠️ 数据质量一般，建议加强审核")
    else:
        st.error("🚨 数据质量较差，需要重点改进")
    
    st.subheader("📈 各项指标")
    for name, score, desc in scores:
        st.markdown(f"**{name}:**")
        st.progress(score / 10)
        st.write(f"  得分: {score}/10 | {desc}")
    
    st.subheader("💡 优化建议")
    if db_stats['verified_count'] == 0:
        st.write("- ⭐ 优先进行人工审核，至少审核1000条低置信度样本")
    if annot_stats.get('avg_center_x', 0.5) > 0.55 or annot_stats.get('avg_center_y', 0.5) > 0.55:
        st.write("- ⭐ 检测到中心偏置，建议增加边缘位置的样本")
    if annot_stats.get('confidence_stats', {}).get('avg', 0) < 0.7:
        st.write("- ⭐ 平均置信度较低，建议检查标注模型或增加训练数据")
