#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日志查看器页面
"""

import streamlit as st
import os
import time


def display_log_viewer():
    """显示日志查看器页面"""
    st.title("📝 日志查看器")
    
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_files = [f for f in os.listdir(log_dir) if f.endswith('.log')]
    
    if not log_files:
        st.info("暂无日志文件")
        return
    
    selected_log = st.selectbox("选择日志文件", log_files, index=0)
    log_path = os.path.join(log_dir, selected_log)
    
    if st.button("🔄 刷新日志"):
        st.rerun()
    
    st.subheader(f"日志内容: {selected_log}")
    
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 显示最近的日志
        lines = content.split('\n')[-200:]
        log_content = '\n'.join(lines)
        
        st.code(log_content, language='text')
        
        if len(lines) < len(content.split('\n')):
            total_lines = len(content.split('\n'))
            st.info(f"仅显示最近200行，共 {total_lines} 行")
    except Exception as e:
        st.error(f"读取日志失败: {e}")
    
    st.subheader("📊 日志统计")
    if os.path.exists(log_path):
        file_size = os.path.getsize(log_path)
        st.write(f"文件大小: {file_size / 1024:.2f} KB")
        
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()
            line_count = len(content.split('\n'))
            error_count = content.count('ERROR') + content.count('error') + content.count('Error')
            warning_count = content.count('WARNING') + content.count('warning') + content.count('Warning')
        
        st.write(f"总行数: {line_count}")
        st.write(f"错误数: {error_count}")
        st.write(f"警告数: {warning_count}")
