#!/usr/bin/env python3
"""
启动自动爬虫，根据关键词爬取角色图片
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.append(str(Path(__file__).parent))

# 先设置环境变量，确保 constants 模块加载时使用正确的路径
os.environ['SIS_DATA_PATH'] = str(Path(__file__).parent / "spider_image_system" / "data")
os.environ['SIS_BASIC_PATH'] = str(Path(__file__).parent / "spider_image_system")

# 添加 spider_image_system 到 Python 路径
sys.path.append(str(Path(__file__).parent / "spider_image_system"))

# 修改 constants 模块的初始化代码，使用环境变量
import spider_image_system.src.run.constants as constants
constants.data_path = os.environ.get('SIS_DATA_PATH', str(Path('./data').resolve()))
constants.basic_path = os.environ.get('SIS_BASIC_PATH', str(Path('.').resolve()))

from spider_image_system.src.ui_event.base_event import auto_spider_img_thread
from spider_image_system.src.file.file_process import get_image_keyword

class MockUI:
    """模拟 UI 对象，用于自动爬虫"""
    def sys_tips(self, msg):
        """显示系统提示"""
        print(f"系统提示: {msg}")

if __name__ == "__main__":
    # 初始化常量
    constants.SpiderConfig.stop_spider_url_flag = True
    
    # 打印路径信息
    print(f"data_path: {constants.data_path}")
    auto_spider_path = os.path.join(constants.data_path, "auto_spider_img")
    print(f"auto_spider_path: {auto_spider_path}")
    keyword_file = os.path.join(auto_spider_path, "spider_img_keyword.txt")
    print(f"keyword_file: {keyword_file}")
    
    # 检查文件是否存在
    if os.path.exists(keyword_file):
        print(f"文件存在: {keyword_file}")
        # 读取文件内容
        with open(keyword_file, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
            print(f"文件内容:\n{content}")
    else:
        print(f"文件不存在: {keyword_file}")
    
    # 测试 get_image_keyword 函数
    print("\n测试 get_image_keyword 函数:")
    keywords, txt_files = get_image_keyword()
    print(f"keywords: {keywords}")
    print(f"txt_files: {txt_files}")
    
    # 创建模拟 UI 对象
    mock_ui = MockUI()
    
    # 启动自动爬虫
    print("\n开始自动爬取角色图片...")
    result = auto_spider_img_thread(mock_ui)
    
    if result:
        print("自动爬取完成！")
    else:
        print("自动爬取失败！")
