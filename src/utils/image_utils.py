#!/usr/bin/env python3
"""
图片处理工具类，封装图片验证、转换和保存等逻辑
"""
import os
import logging
import math
import random
import numpy as np
from PIL import Image, ImageFilter
from io import BytesIO
from skimage import filters, measure
from skimage.color import rgb2gray

logger = logging.getLogger(__name__)


class ImageUtils:
    """图片处理工具类"""
    
    @staticmethod
    def validate_image(content):
        """
        验证图片是否有效
        
        Args:
            content: 图片内容
            
        Returns:
            bool: 图片是否有效
        """
        try:
            img = Image.open(BytesIO(content))
            img.verify()
            img.close()
            return True
        except Exception as e:
            logger.error(f"图片验证失败: {e}")
            return False
    
    @staticmethod
    def get_image_size(content):
        """
        获取图片尺寸
        
        Args:
            content: 图片内容
            
        Returns:
            tuple: (width, height) 或 None
        """
        try:
            img = Image.open(BytesIO(content))
            size = img.size
            img.close()
            return size
        except Exception as e:
            logger.error(f"获取图片尺寸失败: {e}")
            return None
    
    @staticmethod
    def check_image_size(content, min_size=300):
        """
        检查图片尺寸是否满足要求
        
        Args:
            content: 图片内容
            min_size: 最小边长
            
        Returns:
            bool: 图片尺寸是否满足要求
        """
        size = ImageUtils.get_image_size(content)
        if not size:
            return False
        
        width, height = size
        if width < min_size or height < min_size:
            logger.warning(f"图片尺寸过小: {width}x{height}")
            return False
        
        return True
    
    @staticmethod
    def convert_image(content, target_mode='RGB'):
        """
        转换图片模式
        
        Args:
            content: 图片内容
            target_mode: 目标模式
            
        Returns:
            bytes: 转换后的图片内容
        """
        try:
            img = Image.open(BytesIO(content))
            
            # 转换模式
            if img.mode != target_mode:
                img = img.convert(target_mode)
                logger.info(f"图片模式转换: {img.mode} -> {target_mode}")
            
            # 保存为字节流
            buffer = BytesIO()
            img.save(buffer, format='JPEG', quality=95)
            img.close()
            
            return buffer.getvalue()
        except Exception as e:
            logger.error(f"图片转换失败: {e}")
            return content
    
    @staticmethod
    def save_image(content, output_dir, filename, min_size=300):
        """
        保存图片
        
        Args:
            content: 图片内容
            output_dir: 输出目录
            filename: 文件名
            min_size: 最小边长
            
        Returns:
            bool: 保存是否成功
        """
        try:
            # 验证图片
            if not ImageUtils.validate_image(content):
                logger.warning("图片验证失败，跳过保存")
                return False
            
            # 检查图片大小
            if not ImageUtils.check_image_size(content, min_size):
                logger.warning("图片尺寸不满足要求，跳过保存")
                return False
            
            # 创建输出目录
            os.makedirs(output_dir, exist_ok=True)
            
            # 转换图片
            converted_content = ImageUtils.convert_image(content)
            
            # 保存图片
            img_path = os.path.join(output_dir, filename)
            with open(img_path, 'wb') as f:
                f.write(converted_content)
            
            logger.info(f"图片保存成功: {img_path}")
            return True
        except Exception as e:
            logger.error(f"图片保存失败: {e}")
            return False
    
    @staticmethod
    def process_image_batch(image_contents, output_dir, filenames, min_size=300, batch_size=10):
        """
        批量处理图片
        
        Args:
            image_contents: 图片内容列表
            output_dir: 输出目录
            filenames: 文件名列表
            min_size: 最小边长
            batch_size: 批次大小
            
        Returns:
            list: 保存成功的文件名列表
        """
        import concurrent.futures
        
        success_files = []
        total_processed = 0
        
        # 分批次处理
        for i in range(0, len(image_contents), batch_size):
            batch_contents = image_contents[i:i+batch_size]
            batch_filenames = filenames[i:i+batch_size]
            
            # 使用线程池并发处理批次
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(batch_size, 5)) as executor:
                future_to_file = {}
                
                for content, filename in zip(batch_contents, batch_filenames):
                    future = executor.submit(ImageUtils.save_image, content, output_dir, filename, min_size)
                    future_to_file[future] = filename
                
                for future in concurrent.futures.as_completed(future_to_file):
                    filename = future_to_file[future]
                    try:
                        if future.result():
                            success_files.append(filename)
                            total_processed += 1
                    except Exception as e:
                        logger.error(f"处理图片失败 {filename}: {e}")
            
            logger.info(f"批次处理完成，已处理: {total_processed}/{len(image_contents)}")
        
        logger.info(f"批量处理完成，成功: {len(success_files)}/{len(image_contents)}")
        return success_files
    
    @staticmethod
    def calculate_image_quality(image_content):
        """
        计算图片质量分数
        
        Args:
            image_content: 图片内容
            
        Returns:
            float: 质量分数 (0-100)
        """
        try:
            img = Image.open(BytesIO(image_content))
            
            # 计算基础分数
            width, height = img.size
            size_score = min(width * height / (1024 * 1024), 1.0) * 30  # 尺寸分数
            
            # 计算色彩分数
            color_score = 0
            if img.mode in ['RGB', 'RGBA']:
                # 更精确的色彩丰富度评估
                histogram = img.histogram()
                color_diversity = len([h for h in histogram if h > 0])
                color_score = min(color_diversity / 1500, 1.0) * 25
            
            # 计算清晰度分数（使用边缘检测）
            sharpness_score = ImageUtils._calculate_sharpness(image_content) * 25
            
            # 计算对比度分数
            contrast_score = ImageUtils._calculate_contrast(image_content) * 15
            
            # 计算纵横比分数（避免极端比例的图片）
            aspect_ratio = max(width, height) / min(width, height)
            aspect_score = max(0, 1 - (aspect_ratio - 1) / 3) * 5
            
            # 总分数
            total_score = size_score + color_score + sharpness_score + contrast_score + aspect_score
            img.close()
            
            return min(total_score, 100)
        except Exception as e:
            logger.error(f"计算图片质量失败: {e}")
            return 0
    
    @staticmethod
    def _calculate_sharpness(image_content):
        """
        计算图片清晰度
        
        Args:
            image_content: 图片内容
            
        Returns:
            float: 清晰度分数 (0-1)
        """
        try:
            img = Image.open(BytesIO(image_content))
            
            # 转换为灰度图
            if img.mode != 'L':
                img = img.convert('L')
            
            # 应用拉普拉斯边缘检测
            laplacian = img.filter(ImageFilter.Laplacian)
            
            # 计算边缘强度的方差
            np_img = np.array(laplacian)
            variance = np.var(np_img)
            
            # 归一化到 0-1
            sharpness = min(variance / 10000, 1.0)
            
            img.close()
            return sharpness
        except Exception as e:
            logger.error(f"计算清晰度失败: {e}")
            return 0.5  # 默认值
    
    @staticmethod
    def _calculate_contrast(image_content):
        """
        计算图片对比度
        
        Args:
            image_content: 图片内容
            
        Returns:
            float: 对比度分数 (0-1)
        """
        try:
            img = Image.open(BytesIO(image_content))
            
            # 转换为灰度图
            if img.mode != 'L':
                img = img.convert('L')
            
            # 计算像素值的标准差
            np_img = np.array(img)
            std_dev = np.std(np_img)
            
            # 归一化到 0-1
            contrast = min(std_dev / 80, 1.0)
            
            img.close()
            return contrast
        except Exception as e:
            logger.error(f"计算对比度失败: {e}")
            return 0.5  # 默认值
    
    @staticmethod
    def calculate_content_relevance(image_content, query):
        """
        计算图片内容与查询的相关性
        
        Args:
            image_content: 图片内容
            query: 搜索查询
            
        Returns:
            float: 相关性分数 (0-100)
        """
        try:
            # 这里可以集成更复杂的内容识别算法
            # 例如使用CLIP模型进行图文匹配
            # 目前使用简单的启发式方法
            
            # 基础相关性分数
            relevance_score = 70.0
            
            # 检查图片尺寸
            img = Image.open(BytesIO(image_content))
            width, height = img.size
            img.close()
            
            # 尺寸过小的图片相关性降低
            if width < 500 or height < 500:
                relevance_score -= 20
            
            # 这里可以添加更复杂的内容分析
            # 例如：
            # 1. 人脸检测
            # 2. 角色识别
            # 3. 场景分析
            
            return max(0, min(100, relevance_score))
        except Exception as e:
            logger.error(f"计算内容相关性失败: {e}")
            return 50.0  # 默认值
    
    @staticmethod
    def deduplicate_images(image_contents, threshold=0.9):
        """
        去重图片（使用感知哈希）
        
        Args:
            image_contents: 图片内容列表
            threshold: 相似度阈值
            
        Returns:
            list: 去重后的图片内容列表
        """
        try:
            from PIL import Image
            import imagehash
            
            unique_images = []
            seen_hashes = []
            
            for content in image_contents:
                try:
                    img = Image.open(BytesIO(content))
                    # 计算感知哈希
                    img_hash = imagehash.phash(img)
                    img.close()
                    
                    # 检查是否与已有的图片相似
                    is_duplicate = False
                    for existing_hash in seen_hashes:
                        if img_hash - existing_hash < 10:  # 哈希距离小于10视为相似
                            is_duplicate = True
                            break
                    
                    if not is_duplicate:
                        unique_images.append(content)
                        seen_hashes.append(img_hash)
                except Exception as e:
                    logger.error(f"计算图片哈希失败: {e}")
                    # 保留无法计算哈希的图片
                    unique_images.append(content)
            
            logger.info(f"图片去重完成，从 {len(image_contents)} 张减少到 {len(unique_images)} 张")
            return unique_images
        except ImportError:
            # 如果没有安装imagehash，使用简单的MD5去重
            import hashlib
            
            unique_images = []
            seen_hashes = set()
            
            for content in image_contents:
                # 计算图片哈希值
                img_hash = hashlib.md5(content).hexdigest()
                
                if img_hash not in seen_hashes:
                    seen_hashes.add(img_hash)
                    unique_images.append(content)
            
            logger.info(f"图片去重完成，从 {len(image_contents)} 张减少到 {len(unique_images)} 张")
            return unique_images
    
    @staticmethod
    def filter_low_quality_images(image_contents, min_quality=70):
        """
        过滤低质量图片
        
        Args:
            image_contents: 图片内容列表
            min_quality: 最低质量分数
            
        Returns:
            list: 高质量图片内容列表
        """
        high_quality_images = []
        
        for content in image_contents:
            quality_score = ImageUtils.calculate_image_quality(content)
            if quality_score >= min_quality:
                high_quality_images.append(content)
        
        logger.info(f"图片质量筛选完成，从 {len(image_contents)} 张筛选出 {len(high_quality_images)} 张高质量图片")
        return high_quality_images
    
    @staticmethod
    def analyze_image_content(image_content):
        """
        分析图片内容
        
        Args:
            image_content: 图片内容
            
        Returns:
            dict: 内容分析结果
        """
        try:
            img = Image.open(BytesIO(image_content))
            width, height = img.size
            
            analysis = {
                'width': width,
                'height': height,
                'mode': img.mode,
                'aspect_ratio': max(width, height) / min(width, height),
                'size_kb': len(image_content) / 1024,
                'quality_score': ImageUtils.calculate_image_quality(image_content),
                'is_potential_duplicate': False,
                'content_type': 'unknown'
            }
            
            # 简单的内容类型分析
            if width < 300 or height < 300:
                analysis['content_type'] = 'small_image'
            elif analysis['aspect_ratio'] > 3:
                analysis['content_type'] = 'extreme_aspect_ratio'
            else:
                analysis['content_type'] = 'potential_character_image'
            
            img.close()
            return analysis
        except Exception as e:
            logger.error(f"分析图片内容失败: {e}")
            return {
                'width': 0,
                'height': 0,
                'mode': 'unknown',
                'aspect_ratio': 0,
                'size_kb': 0,
                'quality_score': 0,
                'is_potential_duplicate': False,
                'content_type': 'error'
            }
    
    @staticmethod
    def batch_analyze_images(image_contents):
        """
        批量分析图片
        
        Args:
            image_contents: 图片内容列表
            
        Returns:
            list: 分析结果列表
        """
        import concurrent.futures
        
        results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_content = {executor.submit(ImageUtils.analyze_image_content, content): content for content in image_contents}
            
            for future in concurrent.futures.as_completed(future_to_content):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"分析图片失败: {e}")
                    results.append({
                        'width': 0,
                        'height': 0,
                        'mode': 'unknown',
                        'aspect_ratio': 0,
                        'size_kb': 0,
                        'quality_score': 0,
                        'is_potential_duplicate': False,
                        'content_type': 'error'
                    })
        
        logger.info(f"批量分析完成，分析了 {len(results)} 张图片")
        return results
    

    
    @staticmethod
    def resize_image(content, max_size=1024):
        """
        调整图片大小
        
        Args:
            content: 图片内容
            max_size: 最大边长
            
        Returns:
            bytes: 调整后的图片内容
        """
        try:
            img = Image.open(BytesIO(content))
            
            # 计算新尺寸
            width, height = img.size
            if max(width, height) <= max_size:
                # 不需要调整大小
                img.close()
                return content
            
            # 计算缩放比例
            ratio = max_size / max(width, height)
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            
            # 调整大小
            img = img.resize((new_width, new_height), Image.LANCZOS)
            
            # 保存为字节流
            buffer = BytesIO()
            img.save(buffer, format='JPEG', quality=95)
            img.close()
            
            return buffer.getvalue()
        except Exception as e:
            logger.error(f"调整图片大小失败: {e}")
            return content
    
    @staticmethod
    def augment_image(content, num_augmentations=2):
        """
        数据增强
        
        Args:
            content: 图片内容
            num_augmentations: 增强数量
            
        Returns:
            list: 增强后的图片内容列表
        """
        try:
            img = Image.open(BytesIO(content))
            
            augmented_images = []
            
            for i in range(num_augmentations):
                aug_img = img.copy()
                
                # 随机选择增强方法
                aug_method = i % 4
                
                if aug_method == 0:
                    # 水平翻转
                    aug_img = aug_img.transpose(Image.FLIP_LEFT_RIGHT)
                elif aug_method == 1:
                    # 旋转
                    angle = random.choice([90, 180, 270])
                    aug_img = aug_img.rotate(angle, expand=False)
                elif aug_method == 2:
                    # 调整亮度
                    from PIL import ImageEnhance
                    enhancer = ImageEnhance.Brightness(aug_img)
                    factor = random.uniform(0.8, 1.2)
                    aug_img = enhancer.enhance(factor)
                elif aug_method == 3:
                    # 调整对比度
                    from PIL import ImageEnhance
                    enhancer = ImageEnhance.Contrast(aug_img)
                    factor = random.uniform(0.8, 1.2)
                    aug_img = enhancer.enhance(factor)
                
                # 保存为字节流
                buffer = BytesIO()
                aug_img.save(buffer, format='JPEG', quality=95)
                augmented_images.append(buffer.getvalue())
                
                aug_img.close()
            
            img.close()
            return augmented_images
        except Exception as e:
            logger.error(f"数据增强失败: {e}")
            return []
