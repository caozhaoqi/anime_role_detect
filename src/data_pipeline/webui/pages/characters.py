#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
角色管理页面
"""

import streamlit as st
from src.data_pipeline.webui.utils import get_db_modules


def display_characters():
    """显示角色管理页面"""
    st.title("👥 角色管理")
    
    if not get_db_modules():
        st.warning("数据库不可用")
        return
    
    try:
        from src.data_pipeline.webui.utils import init_database, Character, Sample
        engine, Session = init_database()
        session = Session()
        
        characters = session.query(Character).order_by(Character.name).all()
        
        search_query = st.text_input("搜索角色", "")
        if search_query:
            characters = [c for c in characters if search_query.lower() in c.name.lower()]
        
        st.subheader(f"角色列表 ({len(characters)})")
        
        items_per_page = 20
        total_pages = max(1, (len(characters) + items_per_page - 1) // items_per_page)
        page = st.number_input("页码", min_value=1, max_value=total_pages, value=1)
        
        start_idx = (page - 1) * items_per_page
        end_idx = start_idx + items_per_page
        display_chars = characters[start_idx:end_idx]
        
        cols = st.columns(4)
        for i, char in enumerate(display_chars):
            with cols[i % 4]:
                st.markdown(f"**{char.name}**")
                st.write(f"系列: {char.series}")
                if char.aliases:
                    st.write(f"别名: {', '.join(char.aliases)[:30]}...")
                
                sample_count = session.query(Sample).filter(Sample.character_id == char.id).count()
                st.write(f"样本数: {sample_count}")
        
        session.close()
        engine.dispose()
    except Exception as e:
        st.error(f"加载角色列表失败: {e}")
