#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
样本管理页面
"""

import os
import streamlit as st
from PIL import Image
from src.data_pipeline.webui.utils import get_db_modules


def display_samples():
    """显示样本管理页面（带分页）"""
    st.title("🖼️ 样本管理")
    
    if not get_db_modules():
        st.warning("数据库不可用")
        return
    
    try:
        from src.data_pipeline.webui.utils import init_database, Sample
        engine, Session = init_database()
        session = Session()
        
        status_options = ['all', 'pending', 'deduplicated', 'cleaned', 'annotated']
        selected_status = st.selectbox("筛选状态", status_options, index=0)
        
        query = session.query(Sample)
        if selected_status != 'all':
            query = query.filter(Sample.status == selected_status)
        
        total_count = query.count()
        items_per_page = 15
        
        total_pages = max(1, (total_count + items_per_page - 1) // items_per_page)
        page = st.number_input("页码", min_value=1, max_value=total_pages, value=1)
        offset = (page - 1) * items_per_page
        
        samples = query.offset(offset).limit(items_per_page).all()
        
        st.subheader(f"样本列表 ({total_count} 条)")
        
        cols = st.columns(3)
        for i, sample in enumerate(samples):
            with cols[i % 3]:
                try:
                    img = Image.open(sample.image_path)
                    img.thumbnail((150, 150))
                    st.image(img, caption=os.path.basename(sample.image_path))
                except Exception:
                    st.write("📷 图片加载失败")
                
                st.write(f"状态: {sample.status}")
                if sample.character:
                    st.write(f"角色: {sample.character.name}")
                if sample.confidence:
                    st.write(f"置信度: {sample.confidence:.2f}")
        
        session.close()
        engine.dispose()
    except Exception as e:
        st.error(f"加载样本列表失败: {e}")
