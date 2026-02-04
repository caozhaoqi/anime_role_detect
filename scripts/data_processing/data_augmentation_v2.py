#!/usr/bin/env python3
"""
增强数据增强脚本 v2
生成更多样化的训练样本，包括翻转、旋转、亮度调整、对比度调整、裁剪等
"""
import os
import cv2
import numpy as np
import logging
from tqdm import tqdm

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('data_augmentation_v2')


def augment_image(image, output_prefix):
    """
    对单张图像进行多种增强
    
    Args:
        image: 原始图像
        output_prefix: 输出文件前缀
    
    Returns:
        生成的增强图像数量
    """
    augmented_images = []
    
    # 1. 原始图像
    augmented_images.append((f"{output_prefix}_original.jpg", image))
    
    # 2. 水平翻转
    h_flip = cv2.flip(image, 1)
    augmented_images.append((f"{output_prefix}_hflip.jpg", h_flip))
    
    # 3. 垂直翻转
    v_flip = cv2.flip(image, 0)
    augmented_images.append((f"{output_prefix}_vflip.jpg", v_flip))
    
    # 4. 旋转 (多种角度)
    angles = [-15, -10, -5, 5, 10, 15]
    for angle in angles:
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        augmented_images.append((f"{output_prefix}_rotate_{angle}.jpg", rotated))
    
    # 5. 亮度调整
    brightness_factors = [0.7, 0.8, 0.9, 1.1, 1.2, 1.3]
    for factor in brightness_factors:
        bright = cv2.convertScaleAbs(image, alpha=factor, beta=0)
        augmented_images.append((f"{output_prefix}_bright_{factor:.1f}.jpg", bright))
    
    # 6. 对比度调整
    contrast_factors = [0.7, 0.8, 0.9, 1.1, 1.2, 1.3]
    for factor in contrast_factors:
        contrast = cv2.convertScaleAbs(image, alpha=factor, beta=0)
        augmented_images.append((f"{output_prefix}_contrast_{factor:.1f}.jpg", contrast))
    
    # 7. 随机裁剪
    for i in range(3):
        h, w = image.shape[:2]
        crop_h = int(h * 0.8)
        crop_w = int(w * 0.8)
        start_h = np.random.randint(0, h - crop_h + 1)
        start_w = np.random.randint(0, w - crop_w + 1)
        cropped = image[start_h:start_h+crop_h, start_w:start_w+crop_w]
        # 调整回原始大小
        cropped_resized = cv2.resize(cropped, (w, h))
        augmented_images.append((f"{output_prefix}_crop_{i}.jpg", cropped_resized))
    
    # 8. 高斯模糊
    for ksize in [(3, 3), (5, 5)]:
        blurred = cv2.GaussianBlur(image, ksize, 0)
        augmented_images.append((f"{output_prefix}_blur_{ksize[0]}.jpg", blurred))
    
    # 9. 颜色通道调整
    # 随机调整RGB通道
    for i in range(3):
        img_copy = image.copy()
        # 随机调整每个通道的强度
        for c in range(3):
            alpha = np.random.uniform(0.8, 1.2)
            img_copy[:, :, c] = np.clip(img_copy[:, :, c] * alpha, 0, 255).astype(np.uint8)
        augmented_images.append((f"{output_prefix}_color_{i}.jpg", img_copy))
    
    return augmented_images


def process_directory(input_dir, output_dir):
    """
    处理整个目录的图像
    
    Args:
        input_dir: 输入目录
        output_dir: 输出目录
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有子目录（角色）
    characters = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    logger.info(f"发现 {len(characters)} 个角色")
    
    total_images = 0
    total_augmented = 0
    
    for character in characters:
        char_input_dir = os.path.join(input_dir, character)
        char_output_dir = os.path.join(output_dir, character)
        
        # 确保角色输出目录存在
        os.makedirs(char_output_dir, exist_ok=True)
        
        # 获取所有图像文件
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            image_files.extend([f for f in os.listdir(char_input_dir) if f.lower().endswith(ext)])
        
        logger.info(f"处理角色: {character}, 原始图像数量: {len(image_files)}")
        total_images += len(image_files)
        
        # 处理每张图像
        for img_file in tqdm(image_files, desc=f"增强 {character}"):
            img_path = os.path.join(char_input_dir, img_file)
            
            try:
                # 读取图像
                image = cv2.imread(img_path)
                if image is None:
                    logger.warning(f"无法读取图像: {img_path}")
                    continue
                
                # 转换为RGB（OpenCV默认读取为BGR）
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # 生成输出前缀
                base_name = os.path.splitext(img_file)[0]
                output_prefix = os.path.join(char_output_dir, base_name)
                
                # 增强图像
                augmented_images = augment_image(image, output_prefix)
                
                # 保存增强后的图像
                for output_path, aug_image in augmented_images:
                    # 转换回BGR保存
                    aug_image_bgr = cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(output_path, aug_image_bgr)
                
                total_augmented += len(augmented_images)
                
            except Exception as e:
                logger.error(f"处理图像 {img_path} 时出错: {str(e)}")
                continue
    
    logger.info(f"数据增强完成！")
    logger.info(f"原始图像总数: {total_images}")
    logger.info(f"增强后图像总数: {total_augmented}")
    logger.info(f"增强倍数: {total_augmented / total_images:.2f}x")


def main():
    """
    主函数
    """
    input_dir = 'data/all_characters'
    output_dir = 'data/augmented_characters_v2'
    
    logger.info('开始执行增强数据增强...')
    logger.info(f'输入目录: {input_dir}')
    logger.info(f'输出目录: {output_dir}')
    
    process_directory(input_dir, output_dir)
    
    logger.info('增强数据增强完成！')


if __name__ == "__main__":
    main()
