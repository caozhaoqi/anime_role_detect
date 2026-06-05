#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
角色识别模块 - 基于CLIP Embedding + Faiss 检索
Character Recognition Module - Based on CLIP Embedding + Faiss Retrieval

替代传统分类模型，实现：
- 新角色无需重新训练
- 直接更新特征库
- 适合动漫角色长尾场景
"""

from .clip_embedder import CLIPEmbedder
from .feature_store import FeatureStore
from .character_retriever import CharacterRetriever

__all__ = ['CLIPEmbedder', 'FeatureStore', 'CharacterRetriever']
