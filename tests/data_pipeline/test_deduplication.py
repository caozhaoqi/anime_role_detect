"""
CLIP去重系统单元测试
Unit Tests for CLIP Deduplication System
"""
import os
import sys
import unittest
import numpy as np
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data_pipeline.collector.deduplication import CLIPDeduplicator


class TestCLIPDeduplication(unittest.TestCase):
    """CLIP去重器测试类"""
    
    @classmethod
    def setUpClass(cls):
        """在所有测试之前执行一次"""
        print("📥 初始化CLIP去重器...")
        try:
            cls.deduplicator = CLIPDeduplicator(model_name="ViT-B/32")
            cls.deduplicator_available = True
        except Exception as e:
            print(f"⚠️ CLIP模型加载失败（可能需要联网下载）: {e}")
            cls.deduplicator_available = False
        
        # 获取测试图片路径
        cls.test_data_dir = project_root / "data" / "final_dataset" / "Tsukiyo"
        cls.test_images = list(cls.test_data_dir.glob("*.jpg"))[:100]  # 取前10张测试
    
    def test_compute_phash(self):
        """测试感知哈希计算"""
        if not self.test_images:
            self.skipTest("没有测试图片")
        
        for img_path in self.test_images[:3]:
            phash = self.deduplicator.compute_phash(str(img_path))
            self.assertIsNotNone(phash)
            self.assertEqual(len(phash), 16)  # dhash的长度是16位十六进制
    
    def test_compute_embedding(self):
        """测试图片向量计算"""
        if not self.deduplicator_available:
            self.skipTest("CLIP模型不可用")
        if not self.test_images:
            self.skipTest("没有测试图片")
        
        for img_path in self.test_images[:3]:
            embedding = self.deduplicator.compute_embedding(str(img_path))
            self.assertIsNotNone(embedding)
            self.assertEqual(embedding.shape[0], 512)  # ViT-B/32输出512维向量
            # 验证向量归一化
            self.assertAlmostEqual(float(np.linalg.norm(embedding)), 1.0, places=5)
    
    def test_compute_embeddings_batch(self):
        """测试批量向量计算"""
        if not self.deduplicator_available:
            self.skipTest("CLIP模型不可用")
        if len(self.test_images) < 5:
            self.skipTest("测试图片不足")
        
        paths = [str(p) for p in self.test_images[:5]]
        results = self.deduplicator.compute_embeddings(paths, batch_size=3)
        
        self.assertEqual(len(results), 5)
        for path, embedding in results:
            self.assertIsInstance(path, str)
            self.assertEqual(embedding.shape[0], 512)
    
    def test_deduplicate_by_phash(self):
        """测试感知哈希去重"""
        if len(self.test_images) < 2:
            self.skipTest("测试图片不足")
        
        paths = [str(p) for p in self.test_images[:5]]
        retained, duplicates = self.deduplicator.deduplicate_by_phash(paths, threshold=5)
        
        self.assertIsInstance(retained, list)
        self.assertIsInstance(duplicates, list)
        self.assertTrue(len(retained) <= len(paths))
    
    def test_deduplicate_by_clip(self):
        """测试CLIP向量去重"""
        if not self.deduplicator_available:
            self.skipTest("CLIP模型不可用")
        if len(self.test_images) < 2:
            self.skipTest("测试图片不足")
        
        # 先计算向量
        paths = [str(p) for p in self.test_images[:5]]
        embeddings = self.deduplicator.compute_embeddings(paths)
        
        retained, duplicates = self.deduplicator.deduplicate_by_clip(embeddings, threshold=0.98)
        
        self.assertIsInstance(retained, list)
        self.assertIsInstance(duplicates, list)
        self.assertTrue(len(retained) <= len(embeddings))
        
        # 验证重复对包含相似度
        for dup in duplicates:
            self.assertEqual(len(dup), 3)  # (重复路径, 原始路径, 相似度)
            self.assertIsInstance(dup[2], float)
            self.assertGreaterEqual(dup[2], 0.98)
    
    def test_full_deduplication(self):
        """测试完整去重流程"""
        if not self.deduplicator_available:
            self.skipTest("CLIP模型不可用")
        if len(self.test_images) < 3:
            self.skipTest("测试图片不足")
        
        paths = [str(p) for p in self.test_images[:5]]
        retained, stats = self.deduplicator.deduplicate(
            paths,
            phash_threshold=5,
            clip_threshold=0.98,
            batch_size=3
        )
        
        # 验证结果
        self.assertIsInstance(retained, list)
        self.assertIsInstance(stats, dict)
        
        # 验证统计信息
        self.assertEqual(stats['original_count'], 5)
        self.assertLessEqual(stats['after_phash_count'], 5)
        self.assertLessEqual(stats['after_clip_count'], stats['after_phash_count'])
        self.assertEqual(stats['total_removed'], 5 - stats['after_clip_count'])


if __name__ == "__main__":
    # 设置日志级别
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # 运行测试
    unittest.main(verbosity=2)
