#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查faiss索引文件的内容
"""

import faiss
import numpy as np
import os

def check_faiss_index(index_path):
    """检查faiss索引文件"""
    if not os.path.exists(index_path):
        print(f"索引文件不存在: {index_path}")
        return
    
    print(f"加载索引文件: {index_path}")
    index = faiss.read_index(index_path)
    
    print(f"索引类型: {type(index)}")
    print(f"索引维度: {index.d}")
    print(f"索引中的向量数量: {index.ntotal}")
    
    # 检查索引是否支持训练
    print(f"是否支持训练: {index.is_trained}")
    
    # 检查索引的参数
    if hasattr(index, 'nlist'):
        print(f"聚类中心数量: {index.nlist}")
    
    print(f"\n索引文件检查完成")

if __name__ == "__main__":
    check_faiss_index("role_index.faiss")
    print("\n" + "="*50 + "\n")
    check_faiss_index("tests/docs/role_index.faiss")
