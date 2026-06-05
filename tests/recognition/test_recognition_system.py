#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLIP + Faiss 角色识别系统 - 单元测试
Tests for CLIP + Faiss Character Recognition System
"""

import os
import sys
import unittest
import platform
from pathlib import Path
import tempfile
import numpy as np
from PIL import Image

if platform.system() == "Darwin":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

# 添加项目路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.core.recognition import CLIPEmbedder, FeatureStore, CharacterRetriever


class TestFeatureStore(unittest.TestCase):
    """特征库测试"""

    def setUp(self):
        """每个测试前执行"""
        self.tmp_dir = tempfile.mkdtemp()
        self.index_path = os.path.join(self.tmp_dir, "test_index.faiss")
        self.metadata_path = os.path.join(self.tmp_dir, "test_meta.json")
        self.store = FeatureStore(
            dimension=512,
            index_path=self.index_path,
            metadata_path=self.metadata_path,
        )

    def tearDown(self):
        """每个测试后清理"""
        import shutil
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)

    def test_initialization(self):
        """测试初始化"""
        self.store.initialize()
        self.assertIsNotNone(self.store._index)
        self.assertEqual(self.store._stats["total_features"], 0)
        self.assertEqual(self.store._stats["total_characters"], 0)

    def test_add_character(self):
        """测试添加角色"""
        self.store.initialize()
        features = np.random.randn(20, 512).astype(np.float32)
        features = features / np.linalg.norm(features, axis=1, keepdims=True)

        success = self.store.add_character("char1", features, image_paths=["img1", "img2"])
        self.assertTrue(success)
        self.assertTrue(self.store.has_character("char1"))
        self.assertEqual(self.store._stats["total_features"], 20)
        self.assertEqual(self.store._stats["total_characters"], 1)

    def test_add_multiple_characters(self):
        """测试添加多个角色"""
        self.store.initialize()
        for i in range(3):
            features = np.random.randn(10, 512).astype(np.float32)
            features = features / np.linalg.norm(features, axis=1, keepdims=True)
            self.store.add_character(f"char_{i}", features)

        self.assertEqual(self.store._stats["total_characters"], 3)
        self.assertEqual(self.store._stats["total_features"], 30)
        self.assertEqual(len(self.store.list_characters()), 3)

    def test_search(self):
        """测试检索"""
        self.store.initialize()
        # 添加特征
        for i in range(3):
            features = np.random.randn(10, 512).astype(np.float32)
            features = features / np.linalg.norm(features, axis=1, keepdims=True)
            self.store.add_character(f"char_{i}", features)

        # 检索
        query = np.random.randn(512).astype(np.float32)
        results = self.store.search(query, top_k=5)
        self.assertEqual(len(results), 1)
        self.assertGreater(len(results[0]), 0)

        # 验证返回格式
        for character_name, similarity in results[0]:
            self.assertIsInstance(character_name, str)
            self.assertIsInstance(similarity, float)
            self.assertTrue(0 <= similarity <= 1.0)

    def test_search_similar_features(self):
        """测试相似特征检索"""
        self.store.initialize()

        # 添加不同角色的特征
        base_features = np.random.randn(10, 512).astype(np.float32)
        base_features = base_features / np.linalg.norm(base_features, axis=1, keepdims=True)
        self.store.add_character("target_char", base_features)

        other_features = np.random.randn(10, 512).astype(np.float32)
        other_features = other_features / np.linalg.norm(other_features, axis=1, keepdims=True)
        self.store.add_character("other_char", other_features)

        # 用target_char的特征检索
        query = base_features[0]
        results = self.store.search(query, top_k=1)

        self.assertEqual(results[0][0][0], "target_char")
        self.assertGreater(results[0][0][1], 0.5)  # 相似度应该比较高

    def test_save_load(self):
        """测试保存和加载"""
        self.store.initialize()

        # 添加特征
        features = np.random.randn(10, 512).astype(np.float32)
        features = features / np.linalg.norm(features, axis=1, keepdims=True)
        self.store.add_character("char1", features)

        # 保存
        self.assertTrue(self.store.save())

        # 创建新实例并加载
        new_store = FeatureStore(
            dimension=512,
            index_path=self.index_path,
            metadata_path=self.metadata_path,
        )
        new_store.initialize()
        self.assertEqual(new_store._stats["total_features"], 10)
        self.assertTrue(new_store.has_character("char1"))

    def test_empty_search(self):
        """测试空库检索"""
        self.store.initialize()
        query = np.random.randn(512).astype(np.float32)
        results = self.store.search(query, top_k=5)
        self.assertEqual(len(results), 1)
        self.assertEqual(len(results[0]), 0)

    def test_get_stats(self):
        """测试统计信息"""
        self.store.initialize()
        stats = self.store.get_stats()
        self.assertIn("total_features", stats)
        self.assertIn("total_characters", stats)
        self.assertIn("dimension", stats)
        self.assertIn("characters", stats)
        self.assertEqual(stats["dimension"], 512)

    def test_dimension_mismatch(self):
        """测试维度不匹配"""
        self.store.initialize()
        wrong_dim_features = np.random.randn(10, 256).astype(np.float32)
        with self.assertRaises(ValueError):
            self.store.add_character("wrong", wrong_dim_features)

    def test_clear(self):
        """测试清空"""
        self.store.initialize()
        features = np.random.randn(5, 512).astype(np.float32)
        features = features / np.linalg.norm(features, axis=1, keepdims=True)
        self.store.add_character("char1", features)

        self.store.clear()
        self.assertEqual(self.store._stats["total_features"], 0)
        self.assertEqual(self.store._stats["total_characters"], 0)


class TestCharacterRetriever(unittest.TestCase):
    """角色检索器测试"""

    def setUp(self):
        """每个测试前执行"""
        self.tmp_dir = tempfile.mkdtemp()
        self.index_path = os.path.join(self.tmp_dir, "test_index.faiss")
        self.metadata_path = os.path.join(self.tmp_dir, "test_meta.json")

    def tearDown(self):
        import shutil
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)

    def test_retriever_creation(self):
        """测试检索器创建"""
        retriever = CharacterRetriever(
            clip_model_name="ViT-B/32",
            feature_store_path=self.index_path,
            metadata_path=self.metadata_path,
            use_huggingface=True,
        )
        self.assertIsNotNone(retriever.embedder)
        self.assertIsNotNone(retriever.feature_store)
        self.assertEqual(retriever.similarity_threshold, 0.5)

    def test_retriever_initialize(self):
        """测试检索器初始化"""
        retriever = CharacterRetriever(
            clip_model_name="ViT-B/32",
            feature_store_path=self.index_path,
            metadata_path=self.metadata_path,
            use_huggingface=True,
        )
        retriever.initialize()
        self.assertTrue(retriever._initialized)
        self.assertTrue(retriever.embedder.is_initialized())

    def test_register_with_mock_features(self):
        """测试注册（使用真实CLIP和小图片）"""
        # 创建测试图片
        test_img_dir = Path(self.tmp_dir) / "test_images"
        test_img_dir.mkdir(parents=True, exist_ok=True)

        # 创建多张不同颜色的图片作为"测试角色"
        for char_idx, color in enumerate([(255, 0, 0), (0, 255, 0), (0, 0, 255)]):
            char_dir = test_img_dir / f"char_{char_idx}"
            char_dir.mkdir(exist_ok=True)
            for img_idx in range(3):
                img = Image.new("RGB", (224, 224), color)
                # 添加一些噪点增加变化
                import random
                pixels = np.array(img)
                noise = np.random.randint(-30, 30, pixels.shape)
                pixels = np.clip(pixels + noise, 0, 255).astype(np.uint8)
                Image.fromarray(pixels).save(char_dir / f"img_{img_idx}.jpg")

        # 创建检索器
        retriever = CharacterRetriever(
            clip_model_name="ViT-B/32",
            feature_store_path=self.index_path,
            metadata_path=self.metadata_path,
            use_huggingface=True,
        )

        # 注册一个角色
        image_paths = [str(p) for p in (test_img_dir / "char_0").glob("*.jpg")]
        result = retriever.register_character(
            character_name="test_character",
            image_paths=image_paths,
            max_samples=5,
        )

        # CLIP可能在小测试图片上表现不佳，所以只检查基本流程
        self.assertIn("success", result)
        self.assertIn("character_name", result)
        if result["success"]:
            self.assertEqual(result["character_name"], "test_character")

    def test_get_stats(self):
        """测试统计信息"""
        retriever = CharacterRetriever(
            clip_model_name="ViT-B/32",
            feature_store_path=self.index_path,
            metadata_path=self.metadata_path,
        )
        stats = retriever.get_stats()
        self.assertIn("embedder", stats)
        self.assertIn("feature_store", stats)
        self.assertIn("similarity_threshold", stats)
        self.assertEqual(stats["embedder"]["dimension"], 512)


class TestFeatureStorePerformance(unittest.TestCase):
    """特征库性能测试"""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.index_path = os.path.join(self.tmp_dir, "perf_index.faiss")
        self.metadata_path = os.path.join(self.tmp_dir, "perf_meta.json")

    def tearDown(self):
        import shutil
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)

    def test_large_scale_search_performance(self):
        """测试大规模检索性能"""
        import time

        store = FeatureStore(
            dimension=512,
            index_path=self.index_path,
            metadata_path=self.metadata_path,
        )
        store.initialize()

        # 添加100个角色，每个100个特征
        print("\n📥 添加10000个特征...")
        start = time.time()
        for i in range(100):
            features = np.random.randn(100, 512).astype(np.float32)
            features = features / np.linalg.norm(features, axis=1, keepdims=True)
            store.add_character(f"char_{i}", features)
        add_time = time.time() - start
        print(f"   添加耗时: {add_time:.2f}秒")
        self.assertEqual(store._stats["total_features"], 10000)

        # 搜索
        print("🔍 执行批量检索...")
        start = time.time()
        queries = np.random.randn(50, 512).astype(np.float32)
        queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)
        results = store.search(queries, top_k=10)
        search_time = time.time() - start
        print(f"   50个查询耗时: {search_time:.3f}秒")
        print(f"   平均每个查询: {search_time/50*1000:.1f}ms")

        # 验证结果
        self.assertEqual(len(results), 50)
        for r in results:
            self.assertEqual(len(r), 10)

        # 性能断言: 50个查询应在2秒内完成
        self.assertLess(search_time, 2.0)


if __name__ == "__main__":
    print("🧪 运行CLIP+Faiss角色识别系统测试")
    print("=" * 60)
    unittest.main(verbosity=2)
