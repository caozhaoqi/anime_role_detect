#!/usr/bin/env python3
import os
import sys
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# 测试 KMeans 多原型逻辑
from sklearn.cluster import KMeans

# 生成测试数据
np.random.seed(42)
features = np.random.randn(10, 512)  # 10个特征，512维

print(f"原始特征数: {len(features)}")

# 测试多原型逻辑
k = min(3, len(features))
print(f"k = {k}")

kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
kmeans.fit(features)

prototypes = kmeans.cluster_centers_
print(f"生成的原型数: {len(prototypes)}")

# 验证归一化
norms = np.linalg.norm(prototypes, axis=1, keepdims=True)
print(f"原型范数: {norms.flatten()}")
