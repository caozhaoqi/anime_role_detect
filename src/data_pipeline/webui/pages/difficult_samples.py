#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
困难样本审核页面
"""

import streamlit as st
from PIL import Image
from src.data_pipeline.webui.utils import get_db_modules, draw_bbox


def display_difficult_samples():
    """困难样本审核页面"""
    st.title("⚠️ 困难样本审核")
    
    if not get_db_modules():
        st.warning("数据库不可用")
        return
    
    try:
        from src.data_pipeline.webui.utils import init_database, Sample, Annotation
        engine, Session = init_database()
        session = Session()
        
        query = session.query(Sample).filter(
            Sample.status == 'annotated',
            Sample.confidence < 0.7
        ).order_by(Sample.confidence.asc())
        
        total_count = query.count()
        items_per_page = 6
        
        total_pages = max(1, (total_count + items_per_page - 1) // items_per_page)
        page = st.number_input("页码", min_value=1, max_value=total_pages, value=1)
        offset = (page - 1) * items_per_page
        
        samples = query.offset(offset).limit(items_per_page).all()
        
        st.subheader(f"低置信度样本 ({total_count} 条，置信度 < 0.7)")
        st.markdown("**提示:** 这些是最需要人工审核的样本")
        
        cols = st.columns(3)
        for i, sample in enumerate(samples):
            with cols[i % 3]:
                annotations = session.query(Annotation).filter(Annotation.sample_id == sample.id).all()
                
                img_with_bbox = draw_bbox(sample.image_path, annotations)
                if img_with_bbox:
                    img_with_bbox.thumbnail((200, 200))
                    st.image(img_with_bbox, caption=f"置信度: {sample.confidence:.2f}")
                else:
                    try:
                        img = Image.open(sample.image_path)
                        img.thumbnail((200, 200))
                        st.image(img, caption=f"置信度: {sample.confidence:.2f}")
                    except:
                        st.write("📷 图片加载失败")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button(f"✅ 通过 {sample.id}", key=f"diff_pass_{sample.id}"):
                        for ann in annotations:
                            ann.is_verified = True
                        session.commit()
                        st.success("已通过！")
                
                with col2:
                    if st.button(f"❌ 拒绝 {sample.id}", key=f"diff_reject_{sample.id}"):
                        sample.status = 'filtered_quality'
                        session.commit()
                        st.success("已拒绝！")
                
                with col3:
                    if st.button(f"🔄 重标 {sample.id}", key=f"diff_reatt_{sample.id}"):
                        sample.status = 'pending'
                        session.commit()
                        st.success("已标记重新标注！")
        
        session.close()
        engine.dispose()
    except Exception as e:
        st.error(f"加载困难样本页面失败: {e}")
