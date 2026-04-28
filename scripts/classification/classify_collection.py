#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
脚本调用API实现对采集数据分类
每个文件夹下分为"是"和"否"两个子文件夹
"""

import os
import requests
import argparse
import logging
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('classify_collection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CollectionClassifier:
    def __init__(self, api_url, model_name, username='admin', password='admin123', max_workers=4):
        """
        初始化分类器
        
        Args:
            api_url: API地址
            model_name: 模型名称
            username: 用户名
            password: 密码
            max_workers: 最大线程数
        """
        self.api_url = api_url
        self.model_name = model_name
        self.username = username
        self.password = password
        self.max_workers = max_workers
        self.token = None
        self._get_token()
    
    def _get_token(self):
        """获取认证token"""
        try:
            login_url = self.api_url.replace('/classify', '/auth/login')
            data = {
                'username': self.username,
                'password': self.password
            }
            response = requests.post(login_url, data=data, timeout=30)
            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    self.token = result.get('data', {}).get('access_token')
                    logger.info("认证成功")
                    return True
        except Exception as e:
            logger.error(f"获取token失败: {e}")
        return False
    
    def classify_image(self, image_path):
        """
        调用API分类单张图片
        
        Args:
            image_path: 图片路径
        
        Returns:
            tuple: (是否是该角色, 角色名称, 置信度)
        """
        try:
            with open(image_path, 'rb') as f:
                files = {'file': (os.path.basename(image_path), f, 'image/jpeg')}
                data = {
                    'model_name': self.model_name,
                    'use_model': True,
                    'use_attributes': False,
                    'multi_role': False
                }
                headers = {}
                if self.token:
                    headers['Authorization'] = f'Bearer {self.token}'
                
                response = requests.post(self.api_url, files=files, data=data, headers=headers, timeout=30)
                
                if response.status_code == 200:
                    result = response.json()
                    if result.get('success'):
                        role = result.get('data', {}).get('role', 'unknown')
                        confidence = result.get('data', {}).get('confidence', 0.0)
                        return role, float(confidence)
                    else:
                        logger.warning(f"API返回失败: {result.get('message')}")
                else:
                    logger.warning(f"API请求失败: {response.status_code}")
        except Exception as e:
            logger.error(f"分类图片失败 {image_path}: {e}")
        
        return 'unknown', 0.0
    
    def classify_collection(self, collection_dir, threshold=0.5):
        """
        分类整个采集集
        
        Args:
            collection_dir: 采集集目录
            threshold: 置信度阈值
        """
        # 遍历每个角色文件夹
        for role_dir in os.listdir(collection_dir):
            role_path = os.path.join(collection_dir, role_dir)
            
            if not os.path.isdir(role_path):
                continue
            
            logger.info(f"开始分类角色: {role_dir}")
            
            # 创建"是"和"否"子文件夹
            yes_dir = os.path.join(role_path, "是")
            no_dir = os.path.join(role_path, "否")
            
            os.makedirs(yes_dir, exist_ok=True)
            os.makedirs(no_dir, exist_ok=True)
            
            # 获取所有图片文件
            image_files = []
            for file in os.listdir(role_path):
                file_path = os.path.join(role_path, file)
                if os.path.isfile(file_path) and file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    image_files.append(file_path)
            
            if not image_files:
                logger.info(f"角色 {role_dir} 没有图片")
                continue
            
            logger.info(f"找到 {len(image_files)} 张图片")
            
            # 使用线程池并行处理
            results = []
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_image = {
                    executor.submit(self.classify_image, image_path): image_path 
                    for image_path in image_files
                }
                
                for future in tqdm(as_completed(future_to_image), total=len(image_files)):
                    image_path = future_to_image[future]
                    try:
                        predicted_role, confidence = future.result()
                        results.append((image_path, predicted_role, confidence))
                    except Exception as e:
                        logger.error(f"处理图片失败 {image_path}: {e}")
            
            # 处理分类结果
            yes_count = 0
            no_count = 0
            
            for image_path, predicted_role, confidence in results:
                image_name = os.path.basename(image_path)
                
                # 检查预测的角色是否与文件夹名称匹配（支持拼音和汉字）
                is_match = False
                if predicted_role == role_dir:
                    is_match = True
                elif predicted_role in role_dir or role_dir in predicted_role:
                    # 处理部分匹配的情况
                    is_match = True
                
                if is_match and confidence >= threshold:
                    # 移动到"是"文件夹
                    dest_path = os.path.join(yes_dir, image_name)
                    os.rename(image_path, dest_path)
                    yes_count += 1
                else:
                    # 移动到"否"文件夹
                    dest_path = os.path.join(no_dir, image_name)
                    os.rename(image_path, dest_path)
                    no_count += 1
            
            logger.info(f"角色 {role_dir} 分类完成:")
            logger.info(f"  - 是: {yes_count} 张")
            logger.info(f"  - 否: {no_count} 张")
            logger.info(f"  - 总: {yes_count + no_count} 张")

def main():
    parser = argparse.ArgumentParser(description='对采集数据进行分类')
    parser.add_argument('--collection_dir', type=str, 
                        default='/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/role_images',
                        help='采集数据目录')
    parser.add_argument('--api_url', type=str, 
                        default='http://localhost:8000/api/classify',
                        help='API地址')
    parser.add_argument('--model_name', type=str, 
                        default='efficientnet_b3_loli_incremental_20260425_191252',
                        help='模型名称')
    parser.add_argument('--threshold', type=float, 
                        default=0.5,
                        help='置信度阈值')
    parser.add_argument('--max_workers', type=int, 
                        default=4,
                        help='最大线程数')
    
    args = parser.parse_args()
    
    logger.info(f"开始分类采集数据")
    logger.info(f"采集目录: {args.collection_dir}")
    logger.info(f"API地址: {args.api_url}")
    logger.info(f"模型名称: {args.model_name}")
    logger.info(f"置信度阈值: {args.threshold}")
    
    classifier = CollectionClassifier(
        api_url=args.api_url,
        model_name=args.model_name,
        max_workers=args.max_workers
    )
    
    classifier.classify_collection(args.collection_dir, args.threshold)
    
    logger.info("分类完成！")

if __name__ == '__main__':
    main()
