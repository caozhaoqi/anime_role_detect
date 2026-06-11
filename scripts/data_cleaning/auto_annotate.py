#!/usr/bin/env python3
"""自动标注样本图片 - 支持角色识别"""

import os
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_pipeline.database.init_db import init_database, Sample, Annotation
from src.data_pipeline.annotator.yolo_detector import YOLODetector


class CharacterRecognizer:
    """角色识别器 - 基于CLIP特征匹配"""
    
    def __init__(self):
        self.recognizer = None
        self.clip_embedder = None
        
    def _lazy_init(self):
        """延迟初始化识别器"""
        if self.recognizer is None:
            try:
                from src.core.recognition.open_set_recognizer import OpenSetRecognizer
                from src.core.recognition.clip_embedder import CLIPEmbedder
                
                MODEL_NAME = "efficientnet_b3_loli_optimized_v2_20260529_133654"
                MODEL_DIR = os.path.join(project_root, "models", MODEL_NAME)
                
                index_path = os.path.join(MODEL_DIR, "role_index_final.faiss")
                mapping_path = os.path.join(MODEL_DIR, "role_index_final_mapping.json")
                role_info_path = os.path.join(project_root, "src", "core", "data", "role_info.json")
                
                if os.path.exists(index_path):
                    self.recognizer = OpenSetRecognizer(
                        index_path=index_path,
                        mapping_path=mapping_path,
                        role_info_path=role_info_path,
                        unknown_threshold=0.3,
                        fuzzy_threshold=0.5,
                    )
                    print("✅ 角色识别器加载成功")
                else:
                    print("⚠️ 角色索引文件不存在，跳过角色识别")
                
                self.clip_embedder = CLIPEmbedder()
            except Exception as e:
                print(f"⚠️ 角色识别器初始化失败: {e}")
    
    def recognize(self, image_path: str, bbox=None):
        """
        识别图片中的角色
        
        Args:
            image_path: 图片路径
            bbox: 可选的裁剪区域 [x1, y1, x2, y2]
        
        Returns:
            (角色名, 置信度) 或 (None, None)
        """
        self._lazy_init()
        
        if self.recognizer is None or self.clip_embedder is None:
            return None, None
        
        try:
            # 获取图片特征
            feature = self.clip_embedder.get_image_features(image_path, bbox=bbox)
            
            if feature is None:
                return None, None
            
            # 执行识别
            result = self.recognizer.recognize(feature, top_k=1)
            
            if result["predictions"]:
                top_pred = result["predictions"][0]
                return top_pred["role"], top_pred["similarity"]
            
            return None, None
            
        except Exception as e:
            print(f"⚠️ 角色识别失败 {image_path}: {e}")
            return None, None


def auto_annotate_samples(conf_threshold: float = 0.5, limit: int = None, enable_recognition: bool = True):
    """自动标注样本"""
    # 初始化数据库
    engine, Session = init_database()
    session = Session()

    # 创建YOLO检测器
    detector = YOLODetector()
    
    # 创建角色识别器（延迟初始化）
    character_recognizer = CharacterRecognizer() if enable_recognition else None

    # 获取待标注的样本
    query = session.query(Sample).filter(Sample.status == 'pending')
    if limit:
        query = query.limit(limit)
    
    samples = query.all()
    print(f"📋 找到 {len(samples)} 个待标注样本")
    if enable_recognition:
        print(f"🔍 已启用角色识别")

    annotated_count = 0
    skipped_count = 0
    recognized_count = 0

    for sample in samples:
        # 检查是否已标注
        existing_annotation = session.query(Annotation).filter_by(sample_id=sample.id).first()
        if existing_annotation:
            skipped_count += 1
            continue

        # 执行检测
        detections = detector.detect(sample.image_path, conf_threshold=conf_threshold)

        if detections:
            # 使用置信度最高的检测结果
            best_detection = max(detections, key=lambda x: x['confidence'])
            
            # 角色识别
            character_name = None
            character_confidence = None
            if character_recognizer:
                character_name, character_confidence = character_recognizer.recognize(
                    sample.image_path, 
                    best_detection['bbox']
                )
                if character_name:
                    recognized_count += 1
            
            # 创建标注记录
            annotation = Annotation(
                sample_id=sample.id,
                annotator='auto',
                bbox=best_detection['bbox'],
                confidence=best_detection['confidence'],
                character_name=character_name,
                character_confidence=character_confidence,
                is_verified=False
            )
            session.add(annotation)

            # 更新样本信息
            sample.person_count = len(detections)
            sample.confidence = best_detection['confidence']
            sample.status = 'annotated'
            
            annotated_count += 1
        else:
            # 未检测到目标
            sample.status = 'no_detection'
            skipped_count += 1

        # 每50条提交一次
        if annotated_count % 50 == 0 and annotated_count > 0:
            session.commit()
            print(f"📝 已标注 {annotated_count} 个样本...")

    # 提交剩余数据
    session.commit()

    print(f"\n✅ 标注完成！")
    print(f"   已标注: {annotated_count} 个样本")
    print(f"   跳过: {skipped_count} 个样本")
    if enable_recognition:
        print(f"   已识别角色: {recognized_count} 个样本")

    session.close()
    engine.dispose()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="自动标注样本图片")
    parser.add_argument("--conf", type=float, default=0.5, help="置信度阈值")
    parser.add_argument("--limit", type=int, default=None, help="处理数量限制")
    args = parser.parse_args()
    
    auto_annotate_samples(args.conf, args.limit)
