#!/usr/bin/env python3
"""自动标注样本图片"""

import os
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_pipeline.database.init_db import init_database, Sample, Annotation
from src.data_pipeline.annotator.yolo_detector import YOLODetector


def auto_annotate_samples(conf_threshold: float = 0.5, limit: int = None):
    """自动标注样本"""
    # 初始化数据库
    engine, Session = init_database()
    session = Session()

    # 创建YOLO检测器
    detector = YOLODetector()

    # 获取待标注的样本
    query = session.query(Sample).filter(Sample.status == 'pending')
    if limit:
        query = query.limit(limit)
    
    samples = query.all()
    print(f"📋 找到 {len(samples)} 个待标注样本")

    annotated_count = 0
    skipped_count = 0

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
            
            # 创建标注记录
            annotation = Annotation(
                sample_id=sample.id,
                annotator='auto',
                bbox=best_detection['bbox'],
                confidence=best_detection['confidence'],
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

    session.close()
    engine.dispose()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="自动标注样本图片")
    parser.add_argument("--conf", type=float, default=0.5, help="置信度阈值")
    parser.add_argument("--limit", type=int, default=None, help="处理数量限制")
    args = parser.parse_args()
    
    auto_annotate_samples(args.conf, args.limit)
