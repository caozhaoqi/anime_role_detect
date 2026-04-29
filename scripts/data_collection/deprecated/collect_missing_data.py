#!/usr/bin/env python3
"""
采集缺少数据的角色图片脚本
根据配置文件，对需要补充数据的角色进行图片采集
"""

import os
import json
import logging
import requests
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
from io import BytesIO

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='collect_missing_data.log',
    filemode='a'
)
logger = logging.getLogger(__name__)

# 全局配置
GLOBAL_CONFIG = {
    'download_dir': '../../data/role_images',
    'url_dir': '../../spider_image_system/data/img_url',
    'max_workers': 5,
    'timeout': 30,  # 增加超时时间到30秒
    'delay': 0.5,
    'min_resolution': [800, 800],
    'max_retries': 5,  # 增加最大重试次数到5次
    'proxy': None  # 代理设置，可根据需要配置
}

def ensure_directory(directory):
    """确保目录存在"""
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

def download_image(url, save_dir, role_name, timeout=30, min_resolution=(800, 800)):
    """下载单张图片"""
    retries = 0
    backoff_factor = 1  # 退避因子
    
    while retries < GLOBAL_CONFIG['max_retries']:
        try:
            headers = {
                'User-Agent': random.choice([
                    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/120.0.0.0',
                    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15',
                    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
                ]),
                'Referer': 'https://www.google.com/',
                'Accept': 'image/*',
                'Accept-Encoding': 'gzip, deflate, br',
                'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8'
            }
            
            # 配置代理
            proxies = None
            if GLOBAL_CONFIG['proxy']:
                proxies = {
                    'http': GLOBAL_CONFIG['proxy'],
                    'https': GLOBAL_CONFIG['proxy']
                }
            
            # 增加超时时间，根据重试次数动态调整
            current_timeout = timeout * (1 + retries * 0.5)
            
            response = requests.get(
                url, 
                headers=headers, 
                timeout=current_timeout,
                proxies=proxies,
                stream=True,  # 流式下载，减少内存使用
                allow_redirects=True  # 允许重定向
            )
            
            response.raise_for_status()  # 检查HTTP状态码
            
            # 检查内容类型是否为图片
            content_type = response.headers.get('Content-Type', '')
            if not content_type.startswith('image/'):
                return False, f'不是图片类型: {content_type}'
            
            # 检查图片大小
            content_length = int(response.headers.get('Content-Length', 0))
            if content_length < 1024:  # 小于1KB的图片可能是无效的
                return False, f'图片太小: {content_length} bytes'
            
            # 读取图片内容
            image_data = response.content
            
            # 检查图片分辨率
            img = Image.open(BytesIO(image_data))
            width, height = img.size
            
            if width < min_resolution[0] or height < min_resolution[1]:
                return False, f'分辨率不足 ({width}x{height})'
            
            # 检查图片是否损坏
            img.verify()
            
            # 生成唯一文件名
            file_extension = content_type.split('/')[-1]
            if file_extension == 'jpeg':
                file_extension = 'jpg'
            elif file_extension not in ['jpg', 'png', 'webp']:
                file_extension = 'jpg'  # 默认保存为jpg
            
            # 计算已下载的图片数量
            existing_files = [f for f in os.listdir(save_dir) if f.endswith(('.jpg', '.jpeg', '.png', '.webp'))]
            file_name = f"{role_name}_{len(existing_files) + 1}.{file_extension}"
            file_path = os.path.join(save_dir, file_name)
            
            # 保存图片
            with open(file_path, 'wb') as f:
                f.write(image_data)
            
            return True, f'成功下载: {file_name}'
            
        except requests.RequestException as e:
            retries += 1
            if retries >= GLOBAL_CONFIG['max_retries']:
                return False, f'下载失败: {str(e)}'
            # 指数退避
            wait_time = backoff_factor * (2 ** (retries - 1))
            logger.warning(f"下载失败，{wait_time:.2f}秒后重试: {url}")
            time.sleep(wait_time)
        except Exception as e:
            retries += 1
            if retries >= GLOBAL_CONFIG['max_retries']:
                return False, f'处理失败: {str(e)}'
            # 指数退避
            wait_time = backoff_factor * (2 ** (retries - 1))
            logger.warning(f"处理失败，{wait_time:.2f}秒后重试: {url}")
            time.sleep(wait_time)
    
    return False, '达到最大重试次数'

def process_role(role, base_save_dir, base_url_dir):
    """处理单个角色的图片下载"""
    role_name = role['name']
    target_count = role['target_count']
    
    # 确保保存目录存在
    save_dir = os.path.join(base_save_dir, role_name)
    ensure_directory(save_dir)
    
    # 计算当前已有的图片数量
    existing_files = [f for f in os.listdir(save_dir) if f.endswith(('.jpg', '.jpeg', '.png', '.webp'))]
    current_count = len(existing_files)
    
    # 计算需要下载的图片数量
    need_count = max(0, target_count - current_count)
    
    if need_count <= 0:
        logger.info(f"角色 {role_name} 已达到目标数量 {target_count} 张，无需下载")
        return role_name, 0, 0
    
    # 读取URL文件
    url_file = os.path.join(base_url_dir, f"{role_name}_img.txt")
    if not os.path.exists(url_file):
        logger.warning(f"角色 {role_name} 的URL文件不存在: {url_file}")
        return role_name, 0, 0
    
    # 读取URL列表
    with open(url_file, 'r', encoding='utf-8', errors='ignore') as f:
        urls = [line.strip() for line in f if line.strip()]
    
    if not urls:
        logger.warning(f"角色 {role_name} 的URL文件为空")
        return role_name, 0, 0
    
    # 随机打乱URL顺序
    random.shuffle(urls)
    
    # 限制URL数量，避免下载过多
    urls = urls[:need_count * 3]  # 多准备一些URL，因为可能有下载失败的
    
    logger.info(f"开始下载角色 {role_name} 的图片，当前已有 {current_count} 张，需要下载 {need_count} 张")
    
    # 下载图片
    success_count = 0
    failure_count = 0
    
    with ThreadPoolExecutor(max_workers=GLOBAL_CONFIG['max_workers']) as executor:
        futures = []
        
        for url in urls:
            # 检查是否已经达到目标数量
            if success_count >= need_count:
                break
            
            future = executor.submit(
                download_image,
                url,
                save_dir,
                role_name,
                GLOBAL_CONFIG['timeout'],
                tuple(GLOBAL_CONFIG['min_resolution'])
            )
            futures.append((future, url))
            
            # 避免请求过快
            time.sleep(GLOBAL_CONFIG['delay'])
        
        # 处理下载结果
        for future, url in futures:
            if success_count >= need_count:
                break
            
            try:
                success, message = future.result()
                if success:
                    success_count += 1
                    logger.info(f"{role_name}: {message}")
                else:
                    failure_count += 1
                    logger.warning(f"{role_name}: {message} - {url}")
            except Exception as e:
                failure_count += 1
                logger.error(f"{role_name}: 处理异常 - {str(e)} - {url}")
    
    logger.info(f"角色 {role_name} 下载完成，成功 {success_count} 张，失败 {failure_count} 张")
    return role_name, success_count, failure_count

def process_batch(batch, base_save_dir, base_url_dir):
    """处理一个批次的角色"""
    batch_id = batch['batch_id']
    batch_name = batch['name']
    batch_description = batch['description']
    delay = batch.get('delay', 2)
    roles = batch['roles']
    
    logger.info(f"\n开始处理批次 {batch_id}: {batch_name} - {batch_description}")
    logger.info(f"批次包含 {len(roles)} 个角色")
    
    total_success = 0
    total_failure = 0
    
    for role in roles:
        role_name, success, failure = process_role(role, base_save_dir, base_url_dir)
        total_success += success
        total_failure += failure
        
        # 角色之间的延迟
        time.sleep(delay)
    
    logger.info(f"批次 {batch_id}: {batch_name} 处理完成")
    logger.info(f"成功下载 {total_success} 张图片，失败 {total_failure} 张")
    logger.info(f"批次成功率: {total_success / (total_success + total_failure) * 100:.2f}%" if (total_success + total_failure) > 0 else "无下载")
    
    return total_success, total_failure

def main():
    """主函数"""
    logger.info("============================================================")
    logger.info("开始采集缺少数据的角色图片")
    logger.info("============================================================")
    
    # 脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 配置文件路径
    config_file = os.path.join(script_dir, 'missing_data_collection.json')
    
    # 检查配置文件是否存在
    if not os.path.exists(config_file):
        logger.error(f"配置文件不存在: {config_file}")
        return
    
    # 读取配置文件
    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 基础目录
    base_save_dir = os.path.join(script_dir, GLOBAL_CONFIG['download_dir'])
    base_url_dir = os.path.join(script_dir, GLOBAL_CONFIG['url_dir'])
    
    # 确保目录存在
    ensure_directory(base_save_dir)
    ensure_directory(base_url_dir)
    
    # 处理所有批次
    batch_plan = config.get('batch_plan', [])
    
    overall_success = 0
    overall_failure = 0
    
    for batch in batch_plan:
        success, failure = process_batch(batch, base_save_dir, base_url_dir)
        overall_success += success
        overall_failure += failure
    
    logger.info("\n============================================================")
    logger.info("采集缺少数据的角色图片完成")
    logger.info(f"总成功下载: {overall_success} 张")
    logger.info(f"总失败: {overall_failure} 张")
    if (overall_success + overall_failure) > 0:
        logger.info(f"总成功率: {overall_success / (overall_success + overall_failure) * 100:.2f}%")
    logger.info("============================================================")

if __name__ == "__main__":
    main()
