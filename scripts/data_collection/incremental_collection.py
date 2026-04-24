#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增量采集机制，定期更新和补充新的角色图片
"""

import os
import json
import logging
import datetime
import argparse
from batch_download_images import load_config, process_role

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('incremental_collection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 全局配置
GLOBAL_CONFIG = {
    'download_dir': '../../data/role_images',
    'url_dir': '../../spider_image_system/data/img_url',
    'config_file': 'batch_config.json',
    'last_run_file': 'last_run.json',
    'incremental_count': 10  # 每次增量采集的图片数量
}

def load_last_run():
    """加载上次运行记录"""
    if os.path.exists(GLOBAL_CONFIG['last_run_file']):
        try:
            with open(GLOBAL_CONFIG['last_run_file'], 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"加载上次运行记录失败: {e}")
            return {}
    return {}

def save_last_run(last_run):
    """保存上次运行记录"""
    try:
        with open(GLOBAL_CONFIG['last_run_file'], 'w', encoding='utf-8') as f:
            json.dump(last_run, f, ensure_ascii=False, indent=2)
        logger.info("上次运行记录已保存")
    except Exception as e:
        logger.error(f"保存上次运行记录失败: {e}")

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

def run_incremental_collection():
    """执行增量采集"""
    # 加载配置
    config = load_config(GLOBAL_CONFIG['config_file'])
    if not config:
        logger.error("加载配置文件失败")
        return
    
    # 加载上次运行记录
    last_run = load_last_run()
    current_time = datetime.datetime.now().isoformat()
    
    # 处理每个批次
    total_success = 0
    total_fail = 0
    
    for batch_config in config['batch_plan']:
        batch_id = batch_config['batch_id']
        batch_name = batch_config['name']
        
        logger.info("=" * 60)
        logger.info(f"开始处理批次 {batch_id}: {batch_name}")
        logger.info("=" * 60)
        
        # 处理每个角色
        for role_config in batch_config['roles']:
            role_name = role_config['name']
            
            # 获取当前图片数量
            current_count = get_role_image_count(role_name)
            
            # 计算需要下载的数量
            need_count = min(GLOBAL_CONFIG['incremental_count'], role_config['target_count'] - current_count)
            
            if need_count <= 0:
                logger.info(f"角色 {role_name} 已达到目标数量 ({current_count}/{role_config['target_count']})，跳过")
                continue
            
            # 临时修改目标数量，只下载增量部分
            original_target = role_config['target_count']
            role_config['target_count'] = current_count + need_count
            
            logger.info(f"开始增量采集角色 {role_name}，当前 {current_count} 张，目标增加 {need_count} 张")
            
            # 执行下载
            role_name, success, fail = process_role(role_config, batch_config)
            total_success += success
            total_fail += fail
            
            # 恢复原始目标数量
            role_config['target_count'] = original_target
            
            # 更新上次运行记录
            last_run[role_name] = {
                'last_run': current_time,
                'current_count': current_count + success,
                'success': success,
                'fail': fail
            }
        
        # 批次间延迟
        import time
        time.sleep(5)
    
    # 保存上次运行记录
    save_last_run(last_run)
    
    # 输出总结果
    logger.info("=" * 60)
    logger.info("增量采集完成")
    logger.info("=" * 60)
    logger.info(f"成功下载 {total_success} 张图片，失败 {total_fail} 张")
    logger.info("=" * 60)

def create_cron_job():
    """创建cron作业，定期执行增量采集"""
    cron_script = f"""
#!/bin/bash
# 增量采集定时任务
cd /Users/caozhaoqi/PycharmProjects/anime_role_detect/scripts/data_collection
python3 incremental_collection.py
"""
    
    # 保存cron脚本
    with open('run_incremental.sh', 'w') as f:
        f.write(cron_script)
    
    # 设置执行权限
    os.chmod('run_incremental.sh', 0o755)
    
    logger.info("cron脚本已创建: run_incremental.sh")
    logger.info("建议添加以下cron作业（每天凌晨2点执行）:")
    logger.info("0 2 * * * /Users/caozhaoqi/PycharmProjects/anime_role_detect/scripts/data_collection/run_incremental.sh")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='增量采集脚本')
    parser.add_argument('--run', action='store_true', help='执行增量采集')
    parser.add_argument('--create-cron', action='store_true', help='创建cron作业')
    parser.add_argument('--count', type=int, default=10, help='每次增量采集的图片数量')
    
    args = parser.parse_args()
    
    # 更新配置
    if args.count:
        GLOBAL_CONFIG['incremental_count'] = args.count
    
    print("=" * 60)
    print("增量采集机制")
    print("=" * 60)
    
    if args.create_cron:
        create_cron_job()
    elif args.run:
        run_incremental_collection()
    else:
        print("使用 --run 选项执行增量采集")
        print("使用 --create-cron 选项创建cron作业")
        print("使用 --count 选项指定每次增量采集的图片数量")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
