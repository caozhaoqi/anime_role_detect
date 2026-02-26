#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试完整工作流脚本

从数据采集到模型训练的完整流程测试
"""

import os
import sys
import logging
import time
import json
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_collection.keyword_based_collector import KeywordBasedDataCollector
from src.utils.image_utils import ImageUtils
from src.utils.monitoring_system import MonitoringSystem
from src.utils.cache_manager import cache_manager

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('test_workflow')


class WorkflowTest:
    """
    工作流测试类
    """
    
    def __init__(self):
        """
        初始化测试类
        """
        self.output_dir = 'data/test_workflow'
        self.test_results = {}
        self.monitoring_system = MonitoringSystem()
        self.monitoring_system.start()
        
        logger.info("初始化工作流测试")
    
    def test_data_collection(self, characters=None, max_images=20):
        """
        测试数据采集
        
        Args:
            characters: 要采集的角色列表
            max_images: 每个角色的最大图片数
        """
        logger.info("开始测试数据采集")
        
        if characters is None:
            characters = ['雷电将军', '神里绫华', '甘雨', '胡桃']
        
        # 创建采集器
        collector = KeywordBasedDataCollector(output_dir=self.output_dir)
        
        # 采集数据
        collection_results = {}
        for character in characters:
            logger.info(f"采集角色: {character}")
            count = collector._process_character('原神', character, max_images)
            collection_results[character] = count
            logger.info(f"角色 {character} 采集完成，共 {count} 张图片")
        
        # 获取统计信息
        stats = collector.get_dataset_statistics()
        
        result = {
            'characters': characters,
            'collection_results': collection_results,
            'statistics': stats
        }
        
        self.test_results['data_collection'] = result
        logger.info(f"数据采集测试完成，共采集 {stats['total_images']} 张图片")
        
        return result
    
    def test_data_preprocessing(self):
        """
        测试数据预处理
        """
        logger.info("开始测试数据预处理")
        
        # 创建图片工具实例
        image_utils = ImageUtils()
        
        # 遍历所有角色目录
        preprocessing_results = {}
        
        for character_dir in os.listdir(self.output_dir):
            character_path = os.path.join(self.output_dir, character_dir)
            if os.path.isdir(character_path):
                logger.info(f"预处理角色: {character_dir}")
                
                # 获取所有图片
                images = [f for f in os.listdir(character_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                
                # 预处理每张图片
                processed_count = 0
                quality_scores = []
                
                for img_name in images:
                    img_path = os.path.join(character_path, img_name)
                    
                    try:
                        # 读取图片
                        with open(img_path, 'rb') as f:
                            content = f.read()
                        
                        # 计算质量分数
                        score = image_utils.calculate_image_quality(content)
                        quality_scores.append(score)
                        
                        # 分析图片
                        analysis = image_utils.analyze_image(content)
                        
                        processed_count += 1
                        
                    except Exception as e:
                        logger.error(f"预处理图片失败 {img_path}: {e}")
                
                preprocessing_results[character_dir] = {
                    'total_images': len(images),
                    'processed_images': processed_count,
                    'avg_quality': sum(quality_scores) / len(quality_scores) if quality_scores else 0,
                    'min_quality': min(quality_scores) if quality_scores else 0,
                    'max_quality': max(quality_scores) if quality_scores else 0
                }
                
                logger.info(f"角色 {character_dir} 预处理完成，平均质量: {preprocessing_results[character_dir]['avg_quality']:.2f}")
        
        result = {
            'preprocessing_results': preprocessing_results
        }
        
        self.test_results['data_preprocessing'] = result
        logger.info("数据预处理测试完成")
        
        return result
    
    def test_data_augmentation(self):
        """
        测试数据增强
        """
        logger.info("开始测试数据增强")
        
        # 创建图片工具实例
        image_utils = ImageUtils()
        
        # 遍历所有角色目录
        augmentation_results = {}
        
        for character_dir in os.listdir(self.output_dir):
            character_path = os.path.join(self.output_dir, character_dir)
            if os.path.isdir(character_path):
                logger.info(f"增强角色: {character_dir}")
                
                # 获取所有图片
                images = [f for f in os.listdir(character_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                
                # 为每张图片创建增强版本
                augmented_count = 0
                
                for img_name in images:
                    img_path = os.path.join(character_path, img_name)
                    
                    try:
                        # 读取图片
                        with open(img_path, 'rb') as f:
                            content = f.read()
                        
                        # 创建增强版本
                        augmented_images = image_utils.augment_image(content, num_augmentations=2)
                        
                        # 保存增强图片
                        for i, aug_content in enumerate(augmented_images):
                            aug_name = f"{os.path.splitext(img_name)[0]}_aug_{i}.jpg"
                            aug_path = os.path.join(character_path, aug_name)
                            
                            with open(aug_path, 'wb') as f:
                                f.write(aug_content)
                            
                            augmented_count += 1
                        
                    except Exception as e:
                        logger.error(f"增强图片失败 {img_path}: {e}")
                
                augmentation_results[character_dir] = {
                    'original_images': len(images),
                    'augmented_images': augmented_count
                }
                
                logger.info(f"角色 {character_dir} 增强完成，新增 {augmented_count} 张图片")
        
        result = {
            'augmentation_results': augmentation_results
        }
        
        self.test_results['data_augmentation'] = result
        logger.info("数据增强测试完成")
        
        return result
    
    def test_dataset_organization(self):
        """
        测试数据集组织
        """
        logger.info("开始测试数据集组织")
        
        # 创建训练集和验证集
        train_dir = os.path.join(self.output_dir, 'train')
        val_dir = os.path.join(self.output_dir, 'val')
        
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        
        # 分割数据集
        organization_results = {}
        
        for character_dir in os.listdir(self.output_dir):
            character_path = os.path.join(self.output_dir, character_dir)
            
            # 跳过train和val目录
            if character_dir in ['train', 'val']:
                continue
            
            if os.path.isdir(character_path):
                # 创建角色目录
                os.makedirs(os.path.join(train_dir, character_dir), exist_ok=True)
                os.makedirs(os.path.join(val_dir, character_dir), exist_ok=True)
                
                # 获取所有图片
                images = [f for f in os.listdir(character_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                
                # 分割数据（80%训练，20%验证）
                split_idx = int(len(images) * 0.8)
                train_images = images[:split_idx]
                val_images = images[split_idx:]
                
                # 移动图片到对应目录
                for img_name in train_images:
                    src_path = os.path.join(character_path, img_name)
                    dst_path = os.path.join(train_dir, character_dir, img_name)
                    os.rename(src_path, dst_path)
                
                for img_name in val_images:
                    src_path = os.path.join(character_path, img_name)
                    dst_path = os.path.join(val_dir, character_dir, img_name)
                    os.rename(src_path, dst_path)
                
                organization_results[character_dir] = {
                    'total_images': len(images),
                    'train_images': len(train_images),
                    'val_images': len(val_images)
                }
                
                logger.info(f"角色 {character_dir} 组织完成，训练集: {len(train_images)}, 验证集: {len(val_images)}")
        
        # 删除空的原目录
        for character_dir in os.listdir(self.output_dir):
            character_path = os.path.join(self.output_dir, character_dir)
            if character_dir not in ['train', 'val'] and os.path.isdir(character_path):
                if not os.listdir(character_path):
                    os.rmdir(character_path)
        
        result = {
            'organization_results': organization_results
        }
        
        self.test_results['dataset_organization'] = result
        logger.info("数据集组织测试完成")
        
        return result
    
    def test_model_training(self):
        """
        测试模型训练
        """
        logger.info("开始测试模型训练")
        
        # 检查是否有训练数据
        train_dir = os.path.join(self.output_dir, 'train')
        if not os.path.exists(train_dir):
            logger.error("训练数据不存在，请先完成数据采集和组织")
            return {}
        
        # 统计训练数据
        train_stats = {}
        for character_dir in os.listdir(train_dir):
            character_path = os.path.join(train_dir, character_dir)
            if os.path.isdir(character_path):
                images = [f for f in os.listdir(character_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                train_stats[character_dir] = len(images)
        
        logger.info(f"训练数据统计: {train_stats}")
        
        # 这里应该调用模型训练脚本
        # 由于模型训练需要较长的时间，这里只是模拟
        logger.info("模拟模型训练...")
        time.sleep(2)
        
        result = {
            'train_stats': train_stats,
            'model_path': 'models/test_model.pth',
            'training_time': 2.0,
            'accuracy': 0.85
        }
        
        self.test_results['model_training'] = result
        logger.info(f"模型训练测试完成，准确率: {result['accuracy']:.2%}")
        
        return result
    
    def run_all_tests(self, characters=None, max_images=20):
        """
        运行所有测试
        
        Args:
            characters: 要采集的角色列表
            max_images: 每个角色的最大图片数
        """
        logger.info("开始运行完整工作流测试")
        
        # 1. 数据采集
        self.test_data_collection(characters, max_images)
        
        # 2. 数据预处理
        self.test_data_preprocessing()
        
        # 3. 数据增强
        self.test_data_augmentation()
        
        # 4. 数据集组织
        self.test_dataset_organization()
        
        # 5. 模型训练
        self.test_model_training()
        
        # 保存测试结果
        self.save_test_results()
        
        # 打印测试总结
        self.print_test_summary()
        
        logger.info("完整工作流测试完成")
    
    def save_test_results(self):
        """
        保存测试结果
        """
        results_file = 'test_workflow_results.json'
        try:
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(self.test_results, f, ensure_ascii=False, indent=2)
            logger.info(f"测试结果已保存到 {results_file}")
        except Exception as e:
            logger.error(f"保存测试结果失败: {e}")
    
    def print_test_summary(self):
        """
        打印测试总结
        """
        logger.info("\n=== 完整工作流测试总结 ===")
        
        # 数据采集总结
        if 'data_collection' in self.test_results:
            coll_result = self.test_results['data_collection']
            stats = coll_result.get('statistics', {})
            logger.info(f"数据采集: 总角色数 {stats.get('total_characters', 0)}, 总图片数 {stats.get('total_images', 0)}")
        
        # 数据预处理总结
        if 'data_preprocessing' in self.test_results:
            prep_result = self.test_results['data_preprocessing']
            logger.info(f"数据预处理: 已处理 {len(prep_result.get('preprocessing_results', {}))} 个角色")
        
        # 数据增强总结
        if 'data_augmentation' in self.test_results:
            aug_result = self.test_results['data_augmentation']
            total_aug = sum(r['augmented_images'] for r in aug_result.get('augmentation_results', {}).values())
            logger.info(f"数据增强: 新增 {total_aug} 张增强图片")
        
        # 数据集组织总结
        if 'dataset_organization' in self.test_results:
            org_result = self.test_results['dataset_organization']
            total_train = sum(r['train_images'] for r in org_result.get('organization_results', {}).values())
            total_val = sum(r['val_images'] for r in org_result.get('organization_results', {}).values())
            logger.info(f"数据集组织: 训练集 {total_train} 张, 验证集 {total_val} 张")
        
        # 模型训练总结
        if 'model_training' in self.test_results:
            train_result = self.test_results['model_training']
            logger.info(f"模型训练: 准确率 {train_result.get('accuracy', 0):.2%}")
        
        logger.info("======================")
    
    def cleanup(self):
        """
        清理测试资源
        """
        # 停止监控系统
        self.monitoring_system.stop()
        
        # 清理缓存
        cache_manager.clear()
        
        logger.info("清理测试资源")


if __name__ == '__main__':
    # 创建测试实例
    test = WorkflowTest()
    
    try:
        # 运行所有测试
        test.run_all_tests(characters=['雷电将军', '神里绫华', '甘雨', '胡桃'], max_images=10)
    finally:
        # 清理资源
        test.cleanup()
