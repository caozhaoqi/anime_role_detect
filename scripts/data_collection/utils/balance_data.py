#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
平衡数据，确保每个角色的图片数量相对均衡
"""

import os
import json
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('balance_data.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 全局配置
GLOBAL_CONFIG = {
    'download_dir': '../../data/role_images',
    'url_dir': '../../spider_image_system/data/img_url',
    'target_count': 60  # 目标图片数量
}

def get_role_image_count(role_name):
    """获取角色的图片数量"""
    role_dir = os.path.join(GLOBAL_CONFIG['download_dir'], role_name)
    if not os.path.exists(role_dir):
        return 0
    
    image_files = []
    for file in os.listdir(role_dir):
        if file.endswith(('.jpg', '.jpeg', '.png', '.gif', '.webp')):
            image_files.append(file)
    
    return len(image_files)

def get_all_roles():
    """获取所有角色"""
    roles = []
    if not os.path.exists(GLOBAL_CONFIG['download_dir']):
        return roles
    
    for dir_name in os.listdir(GLOBAL_CONFIG['download_dir']):
        role_dir = os.path.join(GLOBAL_CONFIG['download_dir'], dir_name)
        if os.path.isdir(role_dir):
            roles.append(dir_name)
    
    return roles

def get_url_files():
    """获取所有URL文件"""
    url_files = []
    if not os.path.exists(GLOBAL_CONFIG['url_dir']):
        return url_files
    
    for file_name in os.listdir(GLOBAL_CONFIG['url_dir']):
        if file_name.endswith('_img.txt'):
            role_name = file_name.replace('_img.txt', '')
            url_files.append(role_name)
    
    return url_files

def create_balance_plan():
    """创建数据平衡计划"""
    roles = get_all_roles()
    url_files = get_url_files()
    
    # 分析每个角色的图片数量
    role_stats = []
    for role in roles:
        count = get_role_image_count(role)
        has_url_file = role in url_files
        role_stats.append({
            'name': role,
            'count': count,
            'has_url_file': has_url_file,
            'need': max(0, GLOBAL_CONFIG['target_count'] - count)
        })
    
    # 按需要的图片数量排序
    role_stats.sort(key=lambda x: x['need'], reverse=True)
    
    # 分析无图片的角色
    url_roles_without_images = [role for role in url_files if role not in roles]
    
    logger.info(f"现有角色数: {len(roles)}")
    logger.info(f"有URL文件但无图片的角色数: {len(url_roles_without_images)}")
    logger.info(f"\n需要补充图片的角色（按需求排序）:")
    
    for role in role_stats:
        if role['need'] > 0:
            logger.info(f"  {role['name']}: 当前 {role['count']} 张，需要 {role['need']} 张，{'有URL文件' if role['has_url_file'] else '无URL文件'}")
    
    # 为无图片的角色创建记录
    for role in url_roles_without_images:
        logger.info(f"  {role}: 当前 0 张，需要 {GLOBAL_CONFIG['target_count']} 张，有URL文件")
    
    # 创建平衡配置文件
    balance_config = {
        'target_count': GLOBAL_CONFIG['target_count'],
        'roles': []
    }
    
    # 添加需要补充的角色
    for role in role_stats:
        if role['need'] > 0 and role['has_url_file']:
            balance_config['roles'].append({
                'name': role['name'],
                'current_count': role['count'],
                'target_count': GLOBAL_CONFIG['target_count'],
                'need': role['need'],
                'priority': 'high' if role['need'] > 30 else 'medium'
            })
    
    # 添加无图片但有URL文件的角色
    for role in url_roles_without_images:
        balance_config['roles'].append({
            'name': role,
            'current_count': 0,
            'target_count': GLOBAL_CONFIG['target_count'],
            'need': GLOBAL_CONFIG['target_count'],
            'priority': 'high'
        })
    
    # 保存平衡配置文件
    with open('balance_config.json', 'w', encoding='utf-8') as f:
        json.dump(balance_config, f, ensure_ascii=False, indent=2)
    
    logger.info(f"\n平衡配置文件已保存到 balance_config.json")
    logger.info(f"共 {len(balance_config['roles'])} 个角色需要补充图片")
    
    return balance_config

def main():
    """主函数"""
    print("=" * 60)
    print("平衡数据计划")
    print("=" * 60)
    
    balance_config = create_balance_plan()
    
    print("\n" + "=" * 60)
    print("平衡计划创建完成")
    print("=" * 60)
    print(f"目标图片数量: {balance_config['target_count']}")
    print(f"需要补充的角色数: {len(balance_config['roles'])}")
    print("\n优先级分布:")
    
    high_priority = [r for r in balance_config['roles'] if r['priority'] == 'high']
    medium_priority = [r for r in balance_config['roles'] if r['priority'] == 'medium']
    
    print(f"  高优先级: {len(high_priority)} 个角色")
    print(f"  中优先级: {len(medium_priority)} 个角色")
    
    print("\n下一步建议:")
    print("1. 运行批量下载脚本，使用 balance_config.json 作为配置")
    print("2. 定期检查数据平衡情况")
    print("3. 为无URL文件的角色创建URL文件")
    print("=" * 60)

if __name__ == "__main__":
    main()
