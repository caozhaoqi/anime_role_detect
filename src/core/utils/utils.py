#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一的工具函数模块
"""

import os
import hashlib
import cv2
import numpy as np
from PIL import Image
import logging
import json

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('utils')


def calculate_hash(image, hash_size=8):
    """计算图像的感知哈希
    
    Args:
        image: PIL图像对象
        hash_size: 哈希大小
    
    Returns:
        str: 哈希值
    """
    # 转换为灰度图
    image = image.convert('L')
    # 调整大小
    image = image.resize((hash_size + 1, hash_size))
    # 计算差异
    pixels = list(image.getdata())
    diff = []
    for i in range(hash_size):
        for j in range(hash_size):
            pixel_left = pixels[i * (hash_size + 1) + j]
            pixel_right = pixels[i * (hash_size + 1) + j + 1]
            diff.append(pixel_left > pixel_right)
    # 生成哈希
    hash_value = 0
    for i, value in enumerate(diff):
        if value:
            hash_value |= 1 << i
    return str(hash_value)


def detect_faces(image_path):
    """检测图像中的人脸
    
    Args:
        image_path: 图像路径
    
    Returns:
        list: 人脸坐标列表
    """
    try:
        # 加载Haar级联分类器
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            return []
        
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 检测人脸
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        
        return faces.tolist()
    except Exception as e:
        logger.error(f"检测人脸失败: {e}")
        return []


def is_duplicate_image(image1, image2, hash_size=8, threshold=0):
    """判断两个图像是否重复
    
    Args:
        image1: 第一个图像路径或PIL对象
        image2: 第二个图像路径或PIL对象
        hash_size: 哈希大小
        threshold: 差异阈值（0表示完全相同）
    
    Returns:
        bool: 是否重复
    """
    # 加载图像
    if isinstance(image1, str):
        image1 = Image.open(image1)
    if isinstance(image2, str):
        image2 = Image.open(image2)
    
    # 计算哈希
    hash1 = calculate_hash(image1, hash_size)
    hash2 = calculate_hash(image2, hash_size)
    
    # 计算汉明距离
    hamming_distance = bin(int(hash1) ^ int(hash2)).count('1')
    
    return hamming_distance <= threshold


def get_file_hash(file_path):
    """计算文件的MD5哈希
    
    Args:
        file_path: 文件路径
    
    Returns:
        str: MD5哈希值
    """
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            md5_hash.update(byte_block)
    return md5_hash.hexdigest()


def load_json(file_path):
    """加载JSON文件
    
    Args:
        file_path: 文件路径
    
    Returns:
        dict: JSON数据
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"加载JSON文件失败: {e}")
        return {}


def save_json(data, file_path):
    """保存JSON文件
    
    Args:
        data: 数据
        file_path: 文件路径
    """
    try:
        output_dir = os.path.dirname(file_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"保存JSON文件到: {file_path}")
    except Exception as e:
        logger.error(f"保存JSON文件失败: {e}")


def create_directory(directory):
    """创建目录
    
    Args:
        directory: 目录路径
    """
    try:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"创建目录: {directory}")
    except Exception as e:
        logger.error(f"创建目录失败: {e}")


def list_files(directory, extensions=None):
    """列出目录中的文件
    
    Args:
        directory: 目录路径
        extensions: 文件扩展名列表
    
    Returns:
        list: 文件路径列表
    """
    files = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            if extensions and not any(filename.lower().endswith(ext) for ext in extensions):
                continue
            files.append(os.path.join(root, filename))
    return files


def get_class_names(data_dir):
    """获取类别名称
    
    Args:
        data_dir: 数据目录
    
    Returns:
        list: 类别名称列表
    """
    class_names = []
    for item in os.listdir(data_dir):
        item_path = os.path.join(data_dir, item)
        if os.path.isdir(item_path):
            class_names.append(item)
    return sorted(class_names)


def normalize_image(image, size=(224, 224)):
    """标准化图像
    
    Args:
        image: PIL图像对象
        size: 目标大小
    
    Returns:
        numpy.ndarray: 标准化后的图像
    """
    # 调整大小
    image = image.resize(size)
    # 转换为numpy数组
    image = np.array(image)
    # 归一化
    image = image / 255.0
    # 调整通道顺序 (H, W, C) -> (C, H, W)
    image = image.transpose(2, 0, 1)
    return image


def denormalize_image(image):
    """反标准化图像
    
    Args:
        image: 标准化后的图像
    
    Returns:
        PIL.Image: 反标准化后的图像
    """
    # 调整通道顺序 (C, H, W) -> (H, W, C)
    image = image.transpose(1, 2, 0)
    # 反归一化
    image = image * 255.0
    # 转换为PIL图像
    image = Image.fromarray(image.astype(np.uint8))
    return image


def calculate_mean_std(data_dir, extensions=('.jpg', '.jpeg', '.png', '.webp')):
    """计算数据集的均值和标准差
    
    Args:
        data_dir: 数据目录
        extensions: 文件扩展名
    
    Returns:
        tuple: (均值, 标准差)
    """
    import torch
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms
    
    class ImageDataset(Dataset):
        def __init__(self, data_dir, extensions):
            self.data_dir = data_dir
            self.extensions = extensions
            self.images = []
            
            for root, _, files in os.walk(data_dir):
                for file in files:
                    if file.lower().endswith(extensions):
                        self.images.append(os.path.join(root, file))
        
        def __len__(self):
            return len(self.images)
        
        def __getitem__(self, idx):
            image_path = self.images[idx]
            image = Image.open(image_path).convert('RGB')
            transform = transforms.ToTensor()
            return transform(image)
    
    dataset = ImageDataset(data_dir, extensions)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_images = 0
    
    for batch in loader:
        batch = batch.view(batch.size(0), batch.size(1), -1)
        mean += batch.mean(2).sum(0)
        std += batch.std(2).sum(0)
        total_images += batch.size(0)
    
    mean /= total_images
    std /= total_images
    
    return mean.numpy(), std.numpy()


def split_dataset(data_dir, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """分割数据集
    
    Args:
        data_dir: 数据目录
        output_dir: 输出目录
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
    """
    import shutil
    import random
    
    # 检查比例是否正确
    if train_ratio + val_ratio + test_ratio != 1.0:
        raise ValueError("比例之和必须为1.0")
    
    # 创建输出目录
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    test_dir = os.path.join(output_dir, 'test')
    
    create_directory(train_dir)
    create_directory(val_dir)
    create_directory(test_dir)
    
    # 遍历每个类别
    class_names = get_class_names(data_dir)
    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        
        # 获取图像列表
        images = list_files(class_dir, extensions=('.jpg', '.jpeg', '.png', '.webp'))
        random.shuffle(images)
        
        # 计算分割点
        total = len(images)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)
        
        # 分割图像
        train_images = images[:train_end]
        val_images = images[train_end:val_end]
        test_images = images[val_end:]
        
        # 创建类别目录
        create_directory(os.path.join(train_dir, class_name))
        create_directory(os.path.join(val_dir, class_name))
        create_directory(os.path.join(test_dir, class_name))
        
        # 复制图像
        for image in train_images:
            dest = os.path.join(train_dir, class_name, os.path.basename(image))
            shutil.copy(image, dest)
        
        for image in val_images:
            dest = os.path.join(val_dir, class_name, os.path.basename(image))
            shutil.copy(image, dest)
        
        for image in test_images:
            dest = os.path.join(test_dir, class_name, os.path.basename(image))
            shutil.copy(image, dest)
        
        logger.info(f"类别 {class_name} 分割完成: 训练集 {len(train_images)}, 验证集 {len(val_images)}, 测试集 {len(test_images)}")
    
    logger.info(f"数据集分割完成，保存到: {output_dir}")