#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据质量控制脚本

用于验证采集的图像质量，过滤低质量图像，确保数据集的质量和一致性
"""

import os
import argparse
import logging
import PIL
from PIL import Image
import numpy as np
import shutil
from tqdm import tqdm

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('data_quality_controller')


class DataQualityController:
    def __init__(self, data_dir='data/train', min_resolution=512, min_aspect_ratio=0.7, max_aspect_ratio=1.4):
        """
        初始化数据质量控制器
        
        Args:
            data_dir: 数据集目录
            min_resolution: 最小图像分辨率
            min_aspect_ratio: 最小宽高比
            max_aspect_ratio: 最大宽高比
        """
        self.data_dir = data_dir
        self.min_resolution = min_resolution
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        
        # 创建质量报告目录
        self.report_dir = os.path.join(data_dir, 'quality_reports')
        os.makedirs(self.report_dir, exist_ok=True)
        
        # 创建低质量图像目录
        self.low_quality_dir = os.path.join(data_dir, 'low_quality')
        os.makedirs(self.low_quality_dir, exist_ok=True)
    
    def check_image_quality(self, image_path):
        """
        检查单个图像的质量
        
        Args:
            image_path: 图像路径
            
        Returns:
            tuple: (是否通过质量检查, 质量问题列表)
        """
        issues = []
        
        try:
            # 打开图像
            image = Image.open(image_path)
            
            # 检查图像格式
            if image.format not in ['JPEG', 'PNG']:
                issues.append(f"不支持的图像格式: {image.format}")
            
            # 检查图像大小
            width, height = image.size
            
            if width < self.min_resolution or height < self.min_resolution:
                issues.append(f"分辨率过低: {width}x{height}")
            
            # 检查宽高比
            aspect_ratio = width / height
            if aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio:
                issues.append(f"宽高比异常: {aspect_ratio:.2f}")
            
            # 检查图像模式
            if image.mode not in ['RGB', 'L']:
                issues.append(f"不支持的图像模式: {image.mode}")
            
            # 检查图像数据
            image_data = np.array(image)
            
            # 检查图像是否全黑或全白
            if np.min(image_data) == np.max(image_data):
                issues.append("图像全黑或全白")
            
            # 检查图像清晰度（简单的边缘检测）
            if image.mode == 'RGB':
                gray = image.convert('L')
                gray_data = np.array(gray)
                
                # 计算梯度
                gradient_x = np.abs(np.diff(gray_data, axis=1))
                gradient_y = np.abs(np.diff(gray_data, axis=0))
                edge_density = (np.sum(gradient_x > 20) + np.sum(gradient_y > 20)) / (gray_data.size)
                
                if edge_density < 0.01:
                    issues.append("图像可能模糊")
            
            # 检查文件大小
            file_size = os.path.getsize(image_path) / 1024  # KB
            if file_size < 50:
                issues.append(f"文件大小过小: {file_size:.2f}KB")
            elif file_size > 5000:
                issues.append(f"文件大小过大: {file_size:.2f}KB")
            
        except PIL.UnidentifiedImageError:
            issues.append("无法识别的图像文件")
        except Exception as e:
            issues.append(f"图像处理错误: {str(e)}")
        
        return len(issues) == 0, issues
    
    def validate_dataset_quality(self):
        """
        验证整个数据集的质量
        
        Returns:
            dict: 质量报告
        """
        report = {
            'total_characters': 0,
            'total_images': 0,
            'passed_images': 0,
            'failed_images': 0,
            'character_reports': {},
            'quality_issues': {}
        }
        
        # 遍历所有角色目录
        character_dirs = [d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d)) and d not in ['quality_reports', 'low_quality']]
        report['total_characters'] = len(character_dirs)
        
        logger.info(f"开始验证数据集质量，共 {len(character_dirs)} 个角色目录")
        
        for character_dir in tqdm(character_dirs):
            character_path = os.path.join(self.data_dir, character_dir)
            image_files = [f for f in os.listdir(character_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
            
            character_report = {
                'total_images': len(image_files),
                'passed_images': 0,
                'failed_images': 0,
                'issues': {}
            }
            
            for image_file in image_files:
                image_path = os.path.join(character_path, image_file)
                report['total_images'] += 1
                
                passed, issues = self.check_image_quality(image_path)
                
                if passed:
                    report['passed_images'] += 1
                    character_report['passed_images'] += 1
                else:
                    report['failed_images'] += 1
                    character_report['failed_images'] += 1
                    
                    # 记录问题
                    for issue in issues:
                        if issue not in report['quality_issues']:
                            report['quality_issues'][issue] = 0
                        report['quality_issues'][issue] += 1
                        
                        if issue not in character_report['issues']:
                            character_report['issues'][issue] = []
                        character_report['issues'][issue].append(image_file)
            
            report['character_reports'][character_dir] = character_report
        
        # 生成质量报告
        self.generate_quality_report(report)
        
        return report
    
    def move_low_quality_images(self):
        """
        移动低质量图像到低质量目录
        """
        logger.info("开始移动低质量图像")
        
        moved_count = 0
        
        # 遍历所有角色目录
        character_dirs = [d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d)) and d not in ['quality_reports', 'low_quality']]
        
        for character_dir in tqdm(character_dirs):
            character_path = os.path.join(self.data_dir, character_dir)
            image_files = [f for f in os.listdir(character_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
            
            # 创建角色对应的低质量目录
            character_low_quality_dir = os.path.join(self.low_quality_dir, character_dir)
            os.makedirs(character_low_quality_dir, exist_ok=True)
            
            for image_file in image_files:
                image_path = os.path.join(character_path, image_file)
                passed, _ = self.check_image_quality(image_path)
                
                if not passed:
                    # 移动到低质量目录
                    dest_path = os.path.join(character_low_quality_dir, image_file)
                    shutil.move(image_path, dest_path)
                    moved_count += 1
                    logger.debug(f"移动低质量图像: {image_file} -> {character_dir}")
        
        logger.info(f"完成移动低质量图像，共移动 {moved_count} 张")
        return moved_count
    
    def generate_quality_report(self, report):
        """
        生成质量报告文件
        
        Args:
            report: 质量报告数据
        """
        report_path = os.path.join(self.report_dir, 'quality_report.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 数据集质量报告\n\n")
            f.write(f"## 总体统计\n")
            f.write(f"- 总角色数: {report['total_characters']}\n")
            f.write(f"- 总图像数: {report['total_images']}\n")
            f.write(f"- 通过质量检查: {report['passed_images']} ({report['passed_images']/report['total_images']*100:.2f}%)\n")
            f.write(f"- 未通过质量检查: {report['failed_images']} ({report['failed_images']/report['total_images']*100:.2f}%)\n\n")
            
            f.write(f"## 质量问题分布\n")
            for issue, count in sorted(report['quality_issues'].items(), key=lambda x: x[1], reverse=True):
                f.write(f"- {issue}: {count} 张 ({count/report['total_images']*100:.2f}%)\n")
            f.write("\n")
            
            f.write(f"## 角色质量报告\n")
            for character, char_report in report['character_reports'].items():
                f.write(f"### {character}\n")
                f.write(f"- 总图像数: {char_report['total_images']}\n")
                f.write(f"- 通过质量检查: {char_report['passed_images']}\n")
                f.write(f"- 未通过质量检查: {char_report['failed_images']}\n")
                
                if char_report['issues']:
                    f.write("- 质量问题:\n")
                    for issue, images in char_report['issues'].items():
                        f.write(f"  - {issue}: {len(images)} 张\n")
                f.write("\n")
        
        logger.info(f"生成质量报告成功: {report_path}")
    
    def analyze_data_distribution(self):
        """
        分析数据集分布
        
        Returns:
            dict: 分布报告
        """
        distribution_report = {
            'character_distribution': {},
            'resolution_distribution': {},
            'total_characters': 0,
            'total_images': 0,
            'average_images_per_character': 0
        }
        
        # 遍历所有角色目录
        character_dirs = [d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d)) and d not in ['quality_reports', 'low_quality']]
        distribution_report['total_characters'] = len(character_dirs)
        
        total_images = 0
        
        for character_dir in character_dirs:
            character_path = os.path.join(self.data_dir, character_dir)
            image_files = [f for f in os.listdir(character_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
            image_count = len(image_files)
            
            distribution_report['character_distribution'][character_dir] = image_count
            total_images += image_count
            
            # 分析分辨率分布
            for image_file in image_files:
                try:
                    image_path = os.path.join(character_path, image_file)
                    image = Image.open(image_path)
                    resolution = f"{image.width}x{image.height}"
                    
                    if resolution not in distribution_report['resolution_distribution']:
                        distribution_report['resolution_distribution'][resolution] = 0
                    distribution_report['resolution_distribution'][resolution] += 1
                except Exception as e:
                    logger.debug(f"分析分辨率失败 {image_file}: {e}")
        
        distribution_report['total_images'] = total_images
        if distribution_report['total_characters'] > 0:
            distribution_report['average_images_per_character'] = total_images / distribution_report['total_characters']
        
        # 生成分布报告
        self.generate_distribution_report(distribution_report)
        
        return distribution_report
    
    def generate_distribution_report(self, distribution_report):
        """
        生成分布报告文件
        
        Args:
            distribution_report: 分布报告数据
        """
        report_path = os.path.join(self.report_dir, 'distribution_report.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 数据集分布报告\n\n")
            f.write(f"## 总体统计\n")
            f.write(f"- 总角色数: {distribution_report['total_characters']}\n")
            f.write(f"- 总图像数: {distribution_report['total_images']}\n")
            f.write(f"- 平均每个角色图像数: {distribution_report['average_images_per_character']:.2f}\n\n")
            
            f.write(f"## 角色图像分布\n")
            sorted_characters = sorted(distribution_report['character_distribution'].items(), key=lambda x: x[1], reverse=True)
            
            for character, count in sorted_characters:
                f.write(f"- {character}: {count} 张\n")
            f.write("\n")
            
            f.write(f"## 分辨率分布\n")
            sorted_resolutions = sorted(distribution_report['resolution_distribution'].items(), key=lambda x: x[1], reverse=True)[:20]
            
            for resolution, count in sorted_resolutions:
                f.write(f"- {resolution}: {count} 张 ({count/distribution_report['total_images']*100:.2f}%)\n")
            f.write("\n")
            
            # 分析数据平衡情况
            image_counts = list(distribution_report['character_distribution'].values())
            if image_counts:
                min_count = min(image_counts)
                max_count = max(image_counts)
                median_count = sorted(image_counts)[len(image_counts)//2]
                
                f.write(f"## 数据平衡分析\n")
                f.write(f"- 最少图像数: {min_count} 张\n")
                f.write(f"- 最多图像数: {max_count} 张\n")
                f.write(f"- 中位数图像数: {median_count} 张\n")
                if min_count > 0:
                    f.write(f"- 数据不平衡度: {max_count/min_count:.2f} (理想值接近1)\n")
                else:
                    f.write(f"- 数据不平衡度: 无穷大 (存在角色无图像)\n")
        
        logger.info(f"生成分布报告成功: {report_path}")
    
    def cleanup_low_quality(self):
        """
        清理低质量图像
        
        Returns:
            int: 清理的图像数量
        """
        logger.info("开始清理低质量图像")
        
        # 验证数据集质量
        quality_report = self.validate_dataset_quality()
        
        # 移动低质量图像
        moved_count = self.move_low_quality_images()
        
        # 分析数据分布
        distribution_report = self.analyze_data_distribution()
        
        logger.info("数据质量控制完成！")
        logger.info(f"质量检查结果: {quality_report['passed_images']} 张通过, {quality_report['failed_images']} 张未通过")
        logger.info(f"移动低质量图像: {moved_count} 张")
        logger.info(f"数据集分布: 平均每个角色 {distribution_report['average_images_per_character']:.2f} 张图像")
        
        return moved_count


def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='数据质量控制脚本')
    parser.add_argument('--data-dir', type=str, default='data/train', help='数据集目录')
    parser.add_argument('--min-resolution', type=int, default=512, help='最小图像分辨率')
    parser.add_argument('--cleanup', action='store_true', help='是否清理低质量图像')
    
    args = parser.parse_args()
    
    controller = DataQualityController(
        data_dir=args.data_dir,
        min_resolution=args.min_resolution
    )
    
    if args.cleanup:
        controller.cleanup_low_quality()
    else:
        # 只生成报告，不清理
        controller.validate_dataset_quality()
        controller.analyze_data_distribution()


if __name__ == '__main__':
    main()
