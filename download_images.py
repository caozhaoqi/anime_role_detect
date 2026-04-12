#!/usr/bin/env python3
"""
脚本用于下载已采集的图片
"""

import os
import sys
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "spider_image_system" / "src"))

from loguru import logger
from spider_image_system.src.image.spider_img_save import download_images_from_file
from spider_image_system.src.run import constants

# 配置日志
logger.add(
    "download_images.log",
    rotation="10 MB",
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
)

def download_all_images():
    """下载所有已采集的图片"""
    # 确保停止标志为False
    constants.SpiderConfig.stop_download_image_flag = False
    
    try:
        # 查找所有图片URL文件
        project_root = Path(__file__).parent
        data_path = project_root / "spider_image_system" / "data"
        txt_files = []
        
        for root, _, files in os.walk(data_path):
            for f in files:
                if f.endswith("_img.txt"):
                    txt_files.append(os.path.join(root, f))
        
        if not txt_files:
            logger.warning("没有找到图片URL文件")
            return False
        
        logger.info(f"找到 {len(txt_files)} 个图片URL文件")
        
        # 处理每个文件
        for i, txt_path in enumerate(txt_files):
            logger.info(f"处理文件 {i+1}/{len(txt_files)}: {txt_path}")
            
            # 创建保存目录
            file_name = os.path.basename(txt_path).replace("_img.txt", "")
            save_dir = project_root / "downloaded_images" / file_name
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # 读取URL并下载
            with open(txt_path, 'r', encoding='utf-8') as f:
                urls = f.readlines()
            
            total_count = len(urls)
            logger.info(f"文件包含 {total_count} 个图片URL")
            
            for j, url in enumerate(urls):
                url = url.strip()
                if not url:
                    continue
                
                # 跳过SVG文件
                if url.endswith('.svg'):
                    logger.debug(f"跳过SVG文件: {url}")
                    continue
                
                # 提取文件名
                file_name = os.path.basename(url)
                if not file_name:
                    file_name = f"image_{j}.jpg"
                
                save_path = save_dir / file_name
                
                # 下载图片
                try:
                    import requests
                    response = requests.get(url, timeout=10)
                    if response.status_code == 200:
                        with open(save_path, 'wb') as f:
                            f.write(response.content)
                        logger.info(f"下载成功: {file_name} ({j+1}/{total_count})")
                    else:
                        logger.warning(f"下载失败: {url} (状态码: {response.status_code})")
                except Exception as e:
                    logger.warning(f"下载出错: {url} - {str(e)}")
            
        logger.success("所有图片下载完成！")
        return True
        
    except Exception as e:
        logger.error(f"下载过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # 确保停止标志为True
        constants.SpiderConfig.stop_download_image_flag = True

if __name__ == "__main__":
    logger.info("开始下载已采集的图片...")
    success = download_all_images()
    
    if success:
        logger.info("图片下载完成！")
    else:
        logger.warning("下载过程中出现问题")

