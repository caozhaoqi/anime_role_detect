#!/usr/bin/env python3
import os
import sys
from pathlib import Path

# 添加爬虫系统路径
sys.path.append(str(Path(__file__).parent.parent.parent / 'spider_image_system/src'))

from run.constants import get_pinyin, PINYIN_MAPPING

def main():
    print("=" * 60)
    print("🔍 检查剩余角色的URL资源")
    print("=" * 60)
    
    # 需要补充URL的角色（中文名）
    remaining_roles_cn = [
        '月千夜',    # yue4qian1ye4 - 需要3张
        '爱丽儿',    # ai4li4er3 - 需要9张
        '小闪',      # xiao3shan3 - 需要21张
        '釉壶',      # you4hu2 - 需要21张
        '克萝萝',    # ke4luo2luo2 - 需要30张
        '芙丽希娅',  # fu2li4xi1ya4 - 需要39张
    ]
    
    url_dir = Path('spider_image_system/data/img_url')
    
    print("\n当前各角色URL情况:")
    print("-" * 40)
    
    for role_cn in remaining_roles_cn:
        pinyin = get_pinyin(role_cn)
        url_file = url_dir / f'{pinyin}_img.txt'
        
        if url_file.exists():
            with open(url_file, 'r') as f:
                urls = [line.strip() for line in f if line.strip()]
            print(f"{role_cn} ({pinyin}): {len(urls)} 个URL")
        else:
            print(f"{role_cn} ({pinyin}): 无URL文件")
    
    print("\n" + "=" * 60)
    print("⚠️ 注意：要获取更多URL，需要使用完整的爬虫系统")
    print("请运行 spider_image_system 的UI界面或API来重新爬取这些角色")
    print("=" * 60)
    
    print("\n推荐操作：")
    print("1. 启动 spider_image_system 服务")
    print("2. 通过UI或API为以下角色重新爬取URL：")
    for role_cn in remaining_roles_cn:
        print(f"   - {role_cn}")

if __name__ == '__main__':
    main()
