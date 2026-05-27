#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动采集样本数不足的角色数据
根据数据集统计结果，对样本数少于目标数的角色进行URL采集和图片下载
"""

import os
import sys
import subprocess
import time

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 配置
DATASET_DIR = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/final_dataset'
TARGET_COUNT = 100  # 目标每个角色的图片数量
URL_DIR = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/img_url'  # 实际URL文件保存位置
OUTPUT_DIR = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/data/organized_images'

# 角色名称映射（中文到英文）
ROLE_MAPPING = {
    'Himesaka': '姬坂乃爱',
    'Koshenia': '科谢尼娅',
    'Clara': '克拉拉',
    'Hoshino': '小鸟游星野',
    'Mashiro': '真白',
    'Nazuna': '七草荠',
    'Shiro': '白',
    'Satori': '古明地觉',
    'Marisa': '雾雨魔理沙',
    'Reimu': '博丽灵梦',
    'Rem': '雷姆',
    'Ram': '拉姆',
    'Emilia': '爱蜜莉雅',
    'ZeroTwo': '零二',
    'Makima': '玛奇玛',
    'Power': '帕瓦',
    'Makoto': '诚哥',
    'Sayori': '纱世里',
    'Monika': '莫妮卡',
    'Natsuki': '夏树',
    'Yuri': '尤里',
}


def get_insufficient_roles(target_count=TARGET_COUNT):
    """获取样本数不足的角色列表"""
    insufficient = []
    
    if not os.path.exists(DATASET_DIR):
        print(f"❌ 数据集目录不存在: {DATASET_DIR}")
        return insufficient
    
    for role_name in sorted(os.listdir(DATASET_DIR)):
        role_dir = os.path.join(DATASET_DIR, role_name)
        if not os.path.isdir(role_dir) or role_name.startswith('.'):
            continue
        
        # 统计图片数量
        img_count = len([f for f in os.listdir(role_dir) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))])
        
        if img_count < target_count:
            needed = target_count - img_count
            insufficient.append({
                'name': role_name,
                'current': img_count,
                'needed': needed,
                'chinese_name': ROLE_MAPPING.get(role_name, role_name)
            })
    
    # 按需要的数量排序（需要最多的优先）
    insufficient.sort(key=lambda x: x['needed'], reverse=True)
    return insufficient


def check_url_file_exists(role_name):
    """检查角色是否已有URL文件"""
    url_file = os.path.join(URL_DIR, f"{role_name}_img.txt")
    return os.path.exists(url_file)


def count_urls_in_file(role_name):
    """统计URL文件中的URL数量"""
    url_file = os.path.join(URL_DIR, f"{role_name}_img.txt")
    if not os.path.exists(url_file):
        return 0
    
    with open(url_file, 'r', encoding='utf-8') as f:
        urls = [line.strip() for line in f if line.strip()]
    return len(urls)


def run_spider(role_name, chinese_name):
    """调用爬虫采集URL"""
    print(f"\n🔍 开始采集 {role_name} ({chinese_name}) 的URL...")
    
    # 使用新的爬虫脚本
    spider_script = os.path.join(os.path.dirname(__file__), 'spider_single_role.py')
    
    if os.path.exists(spider_script):
        try:
            cmd = ['python3', spider_script, '--role', role_name, '--chinese', chinese_name]
            print(f"执行: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                # 打印爬虫脚本的输出
                if result.stdout:
                    print(result.stdout)
                print(f"✅ {role_name} URL采集完成")
                return True
            else:
                print(f"⚠️ URL采集可能有问题")
                if result.stdout:
                    print(result.stdout)
                if result.stderr:
                    print(f"错误信息: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            print(f"⏰ {role_name} URL采集超时")
            return False
        except Exception as e:
            print(f"❌ {role_name} URL采集失败: {e}")
            return False
    else:
        print(f"❌ 爬虫脚本不存在: {spider_script}")
        return False


def download_images(role_name):
    """下载角色图片"""
    print(f"\n📥 开始下载 {role_name} 的图片...")
    
    url_file = os.path.join(URL_DIR, f"{role_name}_img.txt")
    if not os.path.exists(url_file):
        print(f"❌ 未找到 {role_name} 的URL文件")
        return False
    
    download_script = os.path.join(os.path.dirname(__file__), 'downloaders', 'smart_downloader.py')
    
    if os.path.exists(download_script):
        try:
            # 创建输出目录
            role_output_dir = os.path.join(OUTPUT_DIR, role_name)
            os.makedirs(role_output_dir, exist_ok=True)
            
            cmd = ['python3', download_script, '--role', role_name]
            print(f"执行: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0:
                print(f"✅ {role_name} 图片下载完成")
                return True
            else:
                print(f"⚠️ 图片下载可能有问题: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            print(f"⏰ {role_name} 图片下载超时")
            return False
        except Exception as e:
            print(f"❌ {role_name} 图片下载失败: {e}")
            return False
    else:
        print(f"❌ 下载脚本不存在: {download_script}")
        return False


def main():
    print("=" * 70)
    print("📊 自动采集样本数不足的角色数据")
    print("=" * 70)
    
    # 获取不足的角色
    insufficient_roles = get_insufficient_roles()
    
    if not insufficient_roles:
        print("🎉 所有角色样本数都已达到目标！")
        return
    
    print(f"\n发现 {len(insufficient_roles)} 个角色样本数不足:")
    print("-" * 70)
    print(f"{'角色':<15} {'中文':<12} {'当前数量':<10} {'需要补充':<10}")
    print("-" * 70)
    for role in insufficient_roles:
        print(f"{role['name']:<15} {role['chinese_name']:<12} {role['current']:<10} {role['needed']:<10}")
    print("-" * 70)
    
    # 开始采集
    for role in insufficient_roles[:5]:  # 先处理前5个最需要补充的角色
        print(f"\n{'='*70}")
        print(f"处理: {role['name']} ({role['chinese_name']})")
        print(f"当前: {role['current']} 张, 需要补充: {role['needed']} 张")
        print("="*70)
        
        # 检查URL文件
        has_url_file = check_url_file_exists(role['name'])
        url_count = count_urls_in_file(role['name'])
        
        if not has_url_file or url_count < role['needed']:
            print(f"\n1️⃣ 步骤一: 采集URL")
            success = run_spider(role['name'], role['chinese_name'])
            if not success:
                print(f"⚠️ URL采集失败，跳过该角色")
                continue
            time.sleep(2)  # 等待爬虫完成
        
        # 下载图片
        print(f"\n2️⃣ 步骤二: 下载图片")
        download_images(role['name'])
        time.sleep(1)
    
    print("\n" + "="*70)
    print("✅ 自动采集任务完成")
    print("="*70)


if __name__ == '__main__':
    main()
