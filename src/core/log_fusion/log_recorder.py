#!/usr/bin/env python3
"""
日志记录模块
记录分类过程中的日志，用于后续的日志融合
"""
import os
import json
import logging
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('log_recorder')

class LogRecorder:
    """日志记录类"""
    
    def __init__(self, log_dir='./logs'):
        """初始化日志记录器
        
        Args:
            log_dir: 日志目录
        """
        self.log_dir = log_dir
        
        # 创建日志目录
        os.makedirs(self.log_dir, exist_ok=True)
    
    def record_log(self, image_path, role, similarity, feature, boxes=None, metadata=None):
        """记录分类日志
        
        Args:
            image_path: 图片路径
            role: 分类结果角色
            similarity: 相似度
            feature: 特征向量
            boxes: 检测框
            metadata: 其他元数据
            
        Returns:
            str: 日志文件路径
        """
        try:
            # 生成日志文件名
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            log_filename = f"classification_{timestamp}.json"
            log_path = os.path.join(self.log_dir, log_filename)
            
            # 构建日志数据
            log_data = {
                'timestamp': datetime.now().isoformat(),
                'image_path': image_path,
                'role': role,
                'similarity': float(similarity),
                'feature': feature.tolist() if hasattr(feature, 'tolist') else feature,
                'boxes': boxes or [],
                'metadata': metadata or {}
            }
            
            # 写入日志文件
            with open(log_path, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"分类日志记录成功: {log_path}")
            return log_path
        except Exception as e:
            logger.error(f"记录分类日志失败: {e}")
            return None
    
    def record_batch_logs(self, batch_results):
        """批量记录分类日志
        
        Args:
            batch_results: 批量分类结果列表，每个元素为字典
            
        Returns:
            list: 日志文件路径列表
        """
        log_paths = []
        
        for result in batch_results:
            try:
                log_path = self.record_log(
                    image_path=result.get('image_path'),
                    role=result.get('role'),
                    similarity=result.get('similarity'),
                    feature=result.get('feature'),
                    boxes=result.get('boxes'),
                    metadata=result.get('metadata')
                )
                if log_path:
                    log_paths.append(log_path)
            except Exception as e:
                logger.error(f"批量记录日志失败: {e}")
                continue
        
        logger.info(f"批量日志记录完成，共记录 {len(log_paths)} 条日志")
        return log_paths
    
    def get_log_count(self):
        """获取日志数量
        
        Returns:
            int: 日志数量
        """
        try:
            log_files = [f for f in os.listdir(self.log_dir) if f.endswith('.json')]
            return len(log_files)
        except Exception as e:
            logger.error(f"获取日志数量失败: {e}")
            return 0
    
    def clear_logs(self, keep_days=7):
        """清理过期日志
        
        Args:
            keep_days: 保留天数
            
        Returns:
            int: 删除的日志数量
        """
        try:
            import time
            delete_count = 0
            cutoff_time = time.time() - (keep_days * 24 * 3600)
            
            for file_name in os.listdir(self.log_dir):
                if file_name.endswith('.json'):
                    file_path = os.path.join(self.log_dir, file_name)
                    if os.path.getmtime(file_path) < cutoff_time:
                        os.remove(file_path)
                        delete_count += 1
            
            logger.info(f"清理过期日志完成，共删除 {delete_count} 条日志")
            return delete_count
        except Exception as e:
            logger.error(f"清理过期日志失败: {e}")
            return 0

# 全局日志记录器实例
_global_log_recorder = None

def get_log_recorder():
    """获取全局日志记录器实例
    
    Returns:
        LogRecorder: 日志记录器实例
    """
    global _global_log_recorder
    if _global_log_recorder is None:
        _global_log_recorder = LogRecorder()
    return _global_log_recorder

def record_classification_log(image_path, role, similarity, feature, boxes=None, metadata=None):
    """记录分类日志的便捷函数
    
    Args:
        image_path: 图片路径
        role: 分类结果角色
        similarity: 相似度
        feature: 特征向量
        boxes: 检测框
        metadata: 其他元数据
        
    Returns:
        str: 日志文件路径
    """
    recorder = get_log_recorder()
    return recorder.record_log(image_path, role, similarity, feature, boxes, metadata)

if __name__ == "__main__":
    """测试日志记录模块"""
    import numpy as np
    
    # 初始化日志记录器
    recorder = LogRecorder()
    
    # 生成测试数据
    test_image_path = "test_image.jpg"
    test_role = "蔚蓝档案_星野"
    test_similarity = 0.95
    test_feature = np.random.randn(768)  # 模拟CLIP特征
    test_boxes = [{"x": 100, "y": 100, "w": 200, "h": 200}]
    test_metadata = {"model_version": "v1.0"}
    
    # 记录测试日志
    print("测试记录分类日志...")
    log_path = recorder.record_log(
        test_image_path, test_role, test_similarity, test_feature, test_boxes, test_metadata
    )
    
    if log_path:
        print(f"日志记录成功: {log_path}")
        
        # 读取并显示日志内容
        with open(log_path, 'r', encoding='utf-8') as f:
            log_data = json.load(f)
        print("日志内容:")
        print(json.dumps(log_data, ensure_ascii=False, indent=2))
    else:
        print("日志记录失败")
    
    # 测试批量记录
    print("\n测试批量记录分类日志...")
    batch_results = [
        {
            'image_path': "test_image1.jpg",
            'role': "蔚蓝档案_星野",
            'similarity': 0.95,
            'feature': np.random.randn(768),
            'boxes': [{"x": 100, "y": 100, "w": 200, "h": 200}]
        },
        {
            'image_path': "test_image2.jpg",
            'role': "蔚蓝档案_白子",
            'similarity': 0.92,
            'feature': np.random.randn(768),
            'boxes': [{"x": 150, "y": 150, "w": 180, "h": 180}]
        }
    ]
    
    log_paths = recorder.record_batch_logs(batch_results)
    print(f"批量日志记录完成，共记录 {len(log_paths)} 条日志")
    
    # 测试获取日志数量
    print(f"\n当前日志数量: {recorder.get_log_count()}")
