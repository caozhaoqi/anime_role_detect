#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量下载图片脚本
根据批次配置文件下载图片
"""

import os
import sys
import requests
from PIL import Image
import io
import time
import random
import logging
import json
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

try:
    from src.services.notification_service import (
        send_notification,
        send_training_progress_notification,
        send_training_complete_notification,
        send_training_error_notification
    )
    NOTIFICATION_AVAILABLE = True
except ImportError:
    NOTIFICATION_AVAILABLE = False
    def send_notification(*args, **kwargs): pass
    def send_training_progress_notification(*args, **kwargs): pass
    def send_training_complete_notification(*args, **kwargs): pass
    def send_training_error_notification(*args, **kwargs): pass

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('batch_download_images.log'),
        logging.StreamHandler()
    ]
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
    'proxy': None,  # 代理设置，可根据需要配置
    'notification_interval': 300,  # 通知间隔时间（秒），默认5分钟
    'last_notification_time': 0  # 上次通知时间
}

# 创建下载目录
os.makedirs(GLOBAL_CONFIG['download_dir'], exist_ok=True)


def check_and_send_scheduled_notification(role_stats: dict, total_stats: dict):
    """检查是否需要发送定时进度通知"""
    current_time = time.time()
    interval = GLOBAL_CONFIG.get('notification_interval', 300)

    if current_time - GLOBAL_CONFIG['last_notification_time'] >= interval:
        GLOBAL_CONFIG['last_notification_time'] = current_time

        current_time_str = datetime.now().strftime("%H:%M:%S")
        message = f"📥 数据采集中...\n时间: {current_time_str}\n\n"

        if role_stats:
            message += "📊 当前进度:\n"
            for role_name, stats in list(role_stats.items())[:5]:
                message += f"  • {role_name}: {stats.get('success', 0)} 张\n"
            if len(role_stats) > 5:
                message += f"  ... 还有 {len(role_stats) - 5} 个角色\n"

            total_success = total_stats.get('total_success', 0)
            total_fail = total_stats.get('total_fail', 0)
            message += f"\n总计: 成功 {total_success} 张, 失败 {total_fail} 张"

        logger.info(f"发送定时进度通知...")
        send_notification(message, level="info")
        logger.info(f"定时进度通知发送完成")
        return True
    return False

def load_config(config_path):
    """加载配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
    except Exception as e:
        logger.error(f"加载配置文件失败: {e}")
        return None

def is_valid_image(content, min_resolution=(800, 800)):
    """检查是否为有效图片"""
    try:
        img = Image.open(io.BytesIO(content))
        img.verify()
        
        # 检查分辨率
        img = Image.open(io.BytesIO(content))
        width, height = img.size
        if width < min_resolution[0] or height < min_resolution[1]:
            return False, f"分辨率不足 ({width}x{height})"
        
        return True, ""
    except Exception as e:
        return False, str(e)

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
            
            if response.status_code == 200:
                # 检查内容类型
                content_type = response.headers.get('Content-Type', '')
                if not content_type.startswith('image/'):
                    return False, f"不是图片类型: {content_type}"
                
                # 读取内容
                content = response.content
                
                is_valid, message = is_valid_image(content, min_resolution)
                if is_valid:
                    # 生成文件名
                    url_hash = abs(hash(url)) % 1000000
                    filename = f"{url_hash:06d}.jpg"
                    filepath = os.path.join(save_dir, filename)
                    
                    # 避免重复下载
                    if os.path.exists(filepath):
                        return False, "文件已存在"
                    
                    # 保存图片
                    with open(filepath, 'wb') as f:
                        f.write(content)
                    
                    return True, f"{filename}"
                else:
                    return False, f"无效图片: {message}"
            elif response.status_code in [429, 503, 504]:
                # 服务器繁忙，增加延迟后重试
                retries += 1
                delay = backoff_factor * (2 ** retries) + random.uniform(0, 1)
                logger.warning(f"服务器繁忙 (HTTP {response.status_code})，{delay:.2f}秒后重试...")
                time.sleep(delay)
                continue
            else:
                return False, f"HTTP {response.status_code}"
                
        except requests.exceptions.Timeout:
            retries += 1
            if retries >= GLOBAL_CONFIG['max_retries']:
                return False, "请求超时"
            delay = backoff_factor * (2 ** retries) + random.uniform(0, 1)
            logger.warning(f"请求超时，{delay:.2f}秒后重试...")
            time.sleep(delay)
        except requests.exceptions.ConnectionError:
            retries += 1
            if retries >= GLOBAL_CONFIG['max_retries']:
                return False, "连接错误"
            delay = backoff_factor * (2 ** retries) + random.uniform(0, 1)
            logger.warning(f"连接错误，{delay:.2f}秒后重试...")
            time.sleep(delay)
        except Exception as e:
            retries += 1
            if retries >= GLOBAL_CONFIG['max_retries']:
                return False, str(e)
            delay = backoff_factor * (2 ** retries) + random.uniform(0, 1)
            logger.warning(f"下载失败: {str(e)}，{delay:.2f}秒后重试...")
            time.sleep(delay)

def process_role(role_config, batch_config):
    """处理单个角色"""
    try:
        role_name = role_config['name']
        target_count = role_config['target_count']
        
        # 创建角色目录
        role_dir = os.path.join(GLOBAL_CONFIG['download_dir'], role_name)
        os.makedirs(role_dir, exist_ok=True)
        
        # 统计现有图片数量
        existing_images = len([f for f in os.listdir(role_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
        if existing_images >= target_count:
            logger.info(f"角色 {role_name} 已有 {existing_images} 张图片，达到目标数量，跳过下载")
            return role_name, 0, 0
        
        # 查找URL文件（支持中文和拼音格式）
        url_file = os.path.join(GLOBAL_CONFIG['url_dir'], f"{role_name}_img.txt")
        
        # 如果中文文件名不存在，尝试查找拼音格式的文件名
        if not os.path.exists(url_file):
            # 尝试查找所有可能的拼音格式文件
            import glob
            possible_files = glob.glob(os.path.join(GLOBAL_CONFIG['url_dir'], f"*_img.txt"))
            if possible_files:
                logger.info(f"角色 {role_name} 的URL文件不存在，尝试使用现有拼音格式文件")
                # 这里可以根据需要实现更智能的匹配逻辑
                # 暂时使用第一个找到的文件作为示例
                url_file = possible_files[0]
                logger.info(f"使用文件: {url_file}")
            else:
                logger.warning(f"角色 {role_name} 的URL文件不存在: {url_file}")
                return role_name, 0, 0
        
        # 读取URL文件
        with open(url_file, 'r', encoding='utf-8') as f:
            urls = [line.strip() for line in f if line.strip()]
        
        # 过滤无效URL
        valid_urls = []
        for url in urls:
            # 跳过SVG和图标文件
            if url.endswith('.svg') or 'icon' in url.lower():
                continue
            # 只保留图片文件
            if any(ext in url.lower() for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp']):
                valid_urls.append(url)
        
        # 调试信息
        logger.info(f"角色 {role_name}: 原始URL数量: {len(urls)}, 有效URL数量: {len(valid_urls)}")
        
        # 限制下载数量
        need_images = target_count - existing_images
        download_urls = valid_urls[:need_images]
        
        logger.info(f"角色 {role_name}: 需要下载 {need_images} 张图片，实际可下载 {len(download_urls)} 张")
        
        if not download_urls:
            logger.warning(f"角色 {role_name} 没有可下载的图片链接")
            return role_name, 0, 0
        
        logger.info(f"开始下载角色 {role_name} 的图片，共 {len(download_urls)} 个链接，目标 {target_count} 张")
        
        # 下载图片
        success_count = 0
        fail_count = 0
        
        # 获取质量要求
        min_resolution = batch_config.get('quality_requirements', {}).get('min_resolution', GLOBAL_CONFIG['min_resolution'])
        
        for url in download_urls:
            success, message = download_image(url, role_dir, role_name, GLOBAL_CONFIG['timeout'], min_resolution)
            if success:
                success_count += 1
                logger.info(f"角色 {role_name}: 下载成功 ({success_count}/{len(download_urls)}) - {message}")
            else:
                fail_count += 1
                logger.warning(f"角色 {role_name}: 下载失败 ({fail_count}/{len(download_urls)}) - {message}")
            
            # 延迟，避免请求过于频繁
            time.sleep(GLOBAL_CONFIG['delay'])
        
        # 检查最终数量
        final_count = existing_images + success_count
        logger.info(f"角色 {role_name}: 下载完成，成功 {success_count} 张，失败 {fail_count} 张，总计 {final_count}/{target_count} 张")
        return role_name, success_count, fail_count
        
    except Exception as e:
        logger.error(f"处理角色 {role_config['name']} 时出错: {e}")
        return role_config['name'], 0, 0

def process_batch(batch_config):
    """处理单个批次"""
    batch_id = batch_config['batch_id']
    batch_name = batch_config['name']
    start_time = time.time()

    logger.info("=" * 60)
    logger.info(f"开始处理批次 {batch_id}: {batch_name}")
    logger.info("=" * 60)

    logger.info(f"发送批次开始通知...")
    try:
        success = send_notification(
            f"📥 开始数据采集\n批次: {batch_id} - {batch_name}\n角色数: {len(batch_config['roles'])}",
            level="info"
        )
        logger.info(f"批次开始通知发送结果: {success}")
    except Exception as e:
        logger.error(f"发送批次开始通知失败: {e}")
    logger.info(f"批次开始通知发送完成")

    # 按优先级排序角色
    roles = sorted(batch_config['roles'], key=lambda x: 0 if x['priority'] == 'high' else 1)

    results = {}
    total_success = 0
    total_fail = 0

    for idx, role_config in enumerate(roles):
        role_name, success, fail = process_role(role_config, batch_config)
        if success > 0 or fail > 0:
            results[role_name] = {'success': success, 'fail': fail}
            total_success += success
            total_fail += fail

        check_and_send_scheduled_notification(results, {"total_success": total_success, "total_fail": total_fail})

        # 角色间延迟
        time.sleep(2)

    elapsed = time.time() - start_time
    elapsed_str = f"{elapsed / 60:.1f} 分钟" if elapsed >= 60 else f"{elapsed:.0f} 秒"

    logger.info("=" * 60)
    logger.info(f"批次 {batch_id}: {batch_name} 处理完成")
    logger.info(f"成功下载 {total_success} 张图片，失败 {total_fail} 张")
    logger.info("=" * 60)

    logger.info(f"发送批次完成通知...")
    send_notification(
        f"✅ 数据采集完成\n批次: {batch_id} - {batch_name}\n耗时: {elapsed_str}\n\n📊 结果:\n"
        f"  成功: {total_success} 张\n  失败: {total_fail} 张\n\n"
        f"📁 保存目录: {GLOBAL_CONFIG['download_dir']}",
        level="success"
    )
    logger.info(f"批次完成通知发送完成")

    return results, total_success, total_fail

def main():
    """主函数"""
    start_time = time.time()

    parser = argparse.ArgumentParser(description='批量下载图片脚本')
    parser.add_argument('--config', default='batch_config.json', help='配置文件路径')
    parser.add_argument('--batch', type=int, help='指定批次ID')
    parser.add_argument('--role', help='指定角色名称')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("批量下载图片脚本")
    print("=" * 60)
    
    try:
        # 加载配置
        config = load_config(args.config)
        if not config:
            send_training_error_notification(stage="数据采集", error_message="配置文件加载失败")
            return
        
        # 更新全局配置
        if 'global_settings' in config:
            GLOBAL_CONFIG.update(config['global_settings'])
        
        # 确保下载目录存在
        os.makedirs(GLOBAL_CONFIG['download_dir'], exist_ok=True)
        
        # 处理指定批次
        if args.batch:
            batch_config = None
            for batch in config['batch_plan']:
                if batch['batch_id'] == args.batch:
                    batch_config = batch
                    break
            
            if not batch_config:
                logger.error(f"批次 {args.batch} 不存在")
                send_training_error_notification(stage="数据采集", error_message=f"批次 {args.batch} 不存在")
                return
            
            process_batch(batch_config)
        
        # 处理指定角色
        elif args.role:
            found = False
            for batch in config['batch_plan']:
                for role in batch['roles']:
                    if role['name'] == args.role:
                        send_notification(
                            f"📥 开始下载角色: {args.role}",
                            level="info"
                        )
                        role_name, success, fail = process_role(role, batch)
                        send_notification(
                            f"✅ 角色 {role_name} 下载完成\n成功: {success} 张, 失败: {fail} 张",
                            level="success" if success > 0 else "info"
                        )
                        found = True
                        break
                if found:
                    break
            
            if not found:
                logger.error(f"角色 {args.role} 不存在于配置中")
                send_training_error_notification(stage="数据采集", error_message=f"角色 {args.role} 不存在")
        
        # 处理所有批次
        else:
            total_results = {}
            total_success_all = 0
            total_fail_all = 0
            
            send_notification(
                f"📥 开始处理所有批次\n总批次: {len(config['batch_plan'])}",
                level="info"
            )
            
            for batch_config in config['batch_plan']:
                results, success, fail = process_batch(batch_config)
                total_results.update(results)
                total_success_all += success
                total_fail_all += fail
            
            # 输出总结果
            print("\n" + "=" * 60)
            print("全部批次处理完成")
            print("=" * 60)
            print(f"成功处理 {len(total_results)} 个角色")
            print(f"共下载 {total_success_all} 张图片，失败 {total_fail_all} 张")
            
            if total_results:
                print("\n角色下载统计:")
                for role_name, stats in total_results.items():
                    print(f"  {role_name}: 成功 {stats['success']} 张, 失败 {stats['fail']} 张")
            
            print(f"\n图片已保存到: {GLOBAL_CONFIG['download_dir']}")
            
            elapsed = time.time() - start_time
            elapsed_str = f"{elapsed / 60:.1f} 分钟" if elapsed >= 60 else f"{elapsed:.0f} 秒"
            
            send_notification(
                f"✅ 所有批次处理完成\n耗时: {elapsed_str}\n\n📊 总结果:\n"
                f"  处理角色: {len(total_results)}\n"
                f"  成功: {total_success_all} 张\n"
                f"  失败: {total_fail_all} 张\n\n"
                f"📁 保存目录: {GLOBAL_CONFIG['download_dir']}",
                level="success"
            )
        
        print("=" * 60)
        
    except Exception as e:
        logger.error(f"数据采集失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        send_training_error_notification(stage="数据采集", error_message=str(e))
        print("=" * 60)

if __name__ == "__main__":
    main()
