"""
YOLO检测器单元测试
Unit Tests for YOLO Detector
"""
import os
import sys
import unittest
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent.parent

from src.data_pipeline.annotator.yolo_detector import YOLODetector


class TestYOLODetector(unittest.TestCase):
    """YOLO检测器测试类"""
    
    @classmethod
    def setUpClass(cls):
        """在所有测试之前执行一次"""
        print("📥 初始化YOLO检测器...")
        try:
            cls.detector = YOLODetector(model_path="yolov8n.pt")
            cls.detector_available = True
        except Exception as e:
            print(f"⚠️ YOLO模型加载失败: {e}")
            cls.detector_available = False
        
        # 获取测试图片路径
        cls.test_data_dir = project_root / "data" / "final_dataset" / "Tsukiyo"
        cls.test_images = list(cls.test_data_dir.glob("*.jpg"))[:10]
    
    def test_detect_single_image(self):
        """测试单张图片检测"""
        if not self.detector_available:
            self.skipTest("YOLO模型不可用")
        if not self.test_images:
            self.skipTest("没有测试图片")
        
        img_path = str(self.test_images[0])
        detections = self.detector.detect(img_path, conf_threshold=0.5)
        
        self.assertIsInstance(detections, list)
    
    def test_detect_multiple_images(self):
        """测试批量图片检测"""
        if not self.detector_available:
            self.skipTest("YOLO模型不可用")
        if len(self.test_images) < 3:
            self.skipTest("测试图片不足")
        
        paths = [str(p) for p in self.test_images[:3]]
        results = self.detector.detect_batch(paths, conf_threshold=0.5)
        
        self.assertIsInstance(results, dict)
        self.assertEqual(len(results), 3)
        
        for path, detections in results.items():
            self.assertIsInstance(detections, list)
    
    def test_detection_result_format(self):
        """测试检测结果格式"""
        if not self.detector_available:
            self.skipTest("YOLO模型不可用")
        if not self.test_images:
            self.skipTest("没有测试图片")
        
        img_path = str(self.test_images[0])
        detections = self.detector.detect(img_path)
        
        for det in detections:
            # 验证检测结果结构
            self.assertIn('bbox', det)
            self.assertIn('confidence', det)
            self.assertIn('class_id', det)
            self.assertIn('class_name', det)
            
            # 验证边界框格式
            bbox = det['bbox']
            self.assertEqual(len(bbox), 4)
            self.assertGreaterEqual(bbox[2], bbox[0])  # x2 >= x1
            self.assertGreaterEqual(bbox[3], bbox[1])  # y2 >= y1
            
            # 验证置信度范围
            self.assertGreaterEqual(det['confidence'], 0.0)
            self.assertLessEqual(det['confidence'], 1.0)
    
    def test_get_person_count(self):
        """测试人数统计"""
        if not self.detector_available:
            self.skipTest("YOLO模型不可用")
        if not self.test_images:
            self.skipTest("没有测试图片")
        
        img_path = str(self.test_images[0])
        count = self.detector.get_person_count(img_path)
        
        self.assertIsInstance(count, int)
        self.assertGreaterEqual(count, 0)
    
    def test_get_bbox_area_ratio(self):
        """测试边界框面积比例"""
        if not self.detector_available:
            self.skipTest("YOLO模型不可用")
        if not self.test_images:
            self.skipTest("没有测试图片")
        
        img_path = str(self.test_images[0])
        ratio = self.detector.get_bbox_area_ratio(img_path)
        
        self.assertIsInstance(ratio, float)
        self.assertGreaterEqual(ratio, 0.0)
        self.assertLessEqual(ratio, 1.0)
    
    def test_crop_and_save(self):
        """测试裁剪保存功能"""
        if not self.detector_available:
            self.skipTest("YOLO模型不可用")
        if not self.test_images:
            self.skipTest("没有测试图片")
        
        import tempfile
        
        img_path = str(self.test_images[0])
        detections = self.detector.detect(img_path)
        
        if detections:
            bbox = detections[0]['bbox']
            
            with tempfile.TemporaryDirectory() as tmp_dir:
                output_path = os.path.join(tmp_dir, "cropped.jpg")
                result = self.detector.crop_and_save(img_path, bbox, output_path)
                
                self.assertTrue(result)
                self.assertTrue(os.path.exists(output_path))
    
    def test_draw_bboxes(self):
        """测试绘制边界框功能"""
        if not self.detector_available:
            self.skipTest("YOLO模型不可用")
        if not self.test_images:
            self.skipTest("没有测试图片")
        
        import tempfile
        
        img_path = str(self.test_images[0])
        detections = self.detector.detect(img_path)
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = os.path.join(tmp_dir, "annotated.jpg")
            result = self.detector.draw_bboxes(img_path, detections, output_path)
            
            self.assertTrue(result)
            self.assertTrue(os.path.exists(output_path))
    
    def test_confidence_threshold(self):
        """测试置信度阈值效果"""
        if not self.detector_available:
            self.skipTest("YOLO模型不可用")
        if not self.test_images:
            self.skipTest("没有测试图片")
        
        img_path = str(self.test_images[0])
        
        # 低阈值应该返回更多检测结果
        detections_low = self.detector.detect(img_path, conf_threshold=0.1)
        # 高阈值应该返回更少或相同数量的检测结果
        detections_high = self.detector.detect(img_path, conf_threshold=0.9)
        
        self.assertGreaterEqual(len(detections_low), len(detections_high))


if __name__ == "__main__":
    # 设置日志级别
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # 运行测试
    unittest.main(verbosity=2)
