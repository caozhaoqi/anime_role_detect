#!/usr/bin/env python3
"""
启动优先级角色采集任务
优先采集图片数量少于100张的角色
"""

import subprocess
import sys
from pathlib import Path

BASE_DIR = Path("/Users/caozhaoqi/PycharmProjects/anime_role_detect")
SPIDER_RUN_DIR = BASE_DIR / "spider_image_system/src/run"

def get_priority_roles(limit=10):
    """获取最需要采集的角色列表"""
    role_list = [
        # 第一优先级 - 急需补充
        "Alice",
        "Zao Wu",
        "Kou Erfu",
        "March 7th",
        "Vepley",
        "Ren Ye Ren",
        "Shakri",
        "Xiao Mei Yan",
        "Aris wei4lan2dang4an4",
        "Zao You",
        "Xia Keli",
        "Qing Que",
        "yi1se4lin2",
        "Tibao",
        "Lam",
        "Spark",
        "Yin Lang",
        "Kaelu",
        "Xing",
        "Collei",
        "Columbina",
        # 第二优先级 - 需要补充
        "Luo Qian",
        "luo4ke3ke3",
        "Nezuko",
        "Yue Qian Ye",
        "Elysia",
        "Diona",
        "Nagan",
        "Rosci",
    ]
    return role_list[:limit]

def start_spider(keyword):
    """启动爬虫采集单个角色"""
    print(f"🚀 开始采集: {keyword}")
    
    try:
        # 构建爬虫命令
        cmd = [
            "python3", "sis_main_process.py",
            "--keyword", keyword,
            "--pages", "5"  # 每个角色采集5页
        ]
        
        result = subprocess.run(
            cmd,
            cwd=SPIDER_RUN_DIR,
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode == 0:
            print(f"✅ 采集完成: {keyword}")
            return True
        else:
            print(f"❌ 采集失败: {keyword}")
            print(f"错误信息: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ 采集超时: {keyword}")
        return False
    except Exception as e:
        print(f"❌ 采集异常: {keyword} - {e}")
        return False

def main():
    """主函数"""
    print("=== 启动优先级角色采集任务 ===")
    print()
    
    # 获取优先级角色列表
    priority_roles = get_priority_roles(20)
    print(f"待采集角色数量: {len(priority_roles)}")
    print()
    
    # 依次采集每个角色
    success_count = 0
    fail_count = 0
    
    for i, role in enumerate(priority_roles, 1):
        print(f"\n[{i}/{len(priority_roles)}]")
        if start_spider(role):
            success_count += 1
        else:
            fail_count += 1
            
    print("\n=== 采集任务完成 ===")
    print(f"成功: {success_count} 个角色")
    print(f"失败: {fail_count} 个角色")
    
    return success_count

if __name__ == "__main__":
    main()
