#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据采集状态跟踪脚本
记录和管理采集进度
"""

import os
import json
import logging
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('collection_tracker.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 状态文件路径
STATUS_FILE = './collection_status.json'
DATA_DIR = './data/role_images'


def load_status():
    """加载采集状态"""
    if os.path.exists(STATUS_FILE):
        try:
            with open(STATUS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"加载状态文件失败: {e}")
            return {'batches': {}, 'roles': {}, 'last_updated': None}
    else:
        return {'batches': {}, 'roles': {}, 'last_updated': None}


def save_status(status):
    """保存采集状态"""
    try:
        status['last_updated'] = datetime.now().isoformat()
        with open(STATUS_FILE, 'w', encoding='utf-8') as f:
            json.dump(status, f, ensure_ascii=False, indent=2)
        logger.info(f"状态已保存到: {STATUS_FILE}")
    except Exception as e:
        logger.error(f"保存状态文件失败: {e}")


def update_role_status(role_name, batch_id, success_count, fail_count):
    """更新角色采集状态"""
    status = load_status()
    
    # 确保角色状态存在
    if role_name not in status['roles']:
        status['roles'][role_name] = {
            'total_success': 0,
            'total_fail': 0,
            'batches': {},
            'current_count': 0
        }
    
    # 更新角色状态
    status['roles'][role_name]['total_success'] += success_count
    status['roles'][role_name]['total_fail'] += fail_count
    status['roles'][role_name]['batches'][str(batch_id)] = {
        'success': success_count,
        'fail': fail_count,
        'timestamp': datetime.now().isoformat()
    }
    
    # 计算当前图片数量
    role_dir = os.path.join(DATA_DIR, role_name)
    if os.path.exists(role_dir):
        current_count = len([f for f in os.listdir(role_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
        status['roles'][role_name]['current_count'] = current_count
    
    # 保存状态
    save_status(status)
    return status


def update_batch_status(batch_id, batch_name, success_count, fail_count, roles):
    """更新批次采集状态"""
    status = load_status()
    
    # 确保批次状态存在
    if str(batch_id) not in status['batches']:
        status['batches'][str(batch_id)] = {
            'name': batch_name,
            'total_success': 0,
            'total_fail': 0,
            'roles': {},
            'status': 'in_progress',
            'start_time': datetime.now().isoformat(),
            'end_time': None
        }
    
    # 更新批次状态
    status['batches'][str(batch_id)]['total_success'] += success_count
    status['batches'][str(batch_id)]['total_fail'] += fail_count
    
    # 更新批次角色状态
    for role_name, role_stats in roles.items():
        status['batches'][str(batch_id)]['roles'][role_name] = role_stats
    
    # 保存状态
    save_status(status)
    return status


def mark_batch_completed(batch_id):
    """标记批次完成"""
    status = load_status()
    
    if str(batch_id) in status['batches']:
        status['batches'][str(batch_id)]['status'] = 'completed'
        status['batches'][str(batch_id)]['end_time'] = datetime.now().isoformat()
        save_status(status)
        logger.info(f"批次 {batch_id} 已标记为完成")
    else:
        logger.error(f"批次 {batch_id} 不存在")


def get_collection_stats():
    """获取采集统计信息"""
    status = load_status()
    
    stats = {
        'total_batches': len(status['batches']),
        'completed_batches': len([b for b in status['batches'].values() if b.get('status') == 'completed']),
        'total_roles': len(status['roles']),
        'total_success': sum(r['total_success'] for r in status['roles'].values()),
        'total_fail': sum(r['total_fail'] for r in status['roles'].values()),
        'total_images': sum(r.get('current_count', 0) for r in status['roles'].values())
    }
    
    return stats


def generate_report():
    """生成采集报告"""
    status = load_status()
    stats = get_collection_stats()
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'statistics': stats,
        'batch_status': status['batches'],
        'role_status': status['roles']
    }
    
    # 保存报告
    report_path = f'collection_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    logger.info(f"采集报告已生成: {report_path}")
    return report


def display_status():
    """显示当前采集状态"""
    status = load_status()
    stats = get_collection_stats()
    
    print("=" * 60)
    print("数据采集状态")
    print("=" * 60)
    print(f"总批次: {stats['total_batches']}")
    print(f"已完成批次: {stats['completed_batches']}")
    print(f"总角色: {stats['total_roles']}")
    print(f"成功下载: {stats['total_success']} 张")
    print(f"失败下载: {stats['total_fail']} 张")
    print(f"总图片数: {stats['total_images']} 张")
    print()
    
    # 显示批次状态
    if status['batches']:
        print("批次状态:")
        for batch_id, batch_info in status['batches'].items():
            status_str = batch_info.get('status', 'unknown')
            success = batch_info.get('total_success', 0)
            fail = batch_info.get('total_fail', 0)
            print(f"  批次 {batch_id}: {batch_info['name']} - {status_str}")
            print(f"    成功: {success}, 失败: {fail}")
    
    # 显示角色状态
    if status['roles']:
        print("\n角色状态:")
        for role_name, role_info in status['roles'].items():
            current = role_info.get('current_count', 0)
            success = role_info['total_success']
            fail = role_info['total_fail']
            print(f"  {role_name}: {current} 张图片")
            print(f"    成功: {success}, 失败: {fail}")
    
    print("=" * 60)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='数据采集状态跟踪脚本')
    parser.add_argument('command', choices=['status', 'report', 'update', 'complete'], help='执行的命令')
    parser.add_argument('--batch', type=int, help='批次ID')
    parser.add_argument('--role', help='角色名称')
    parser.add_argument('--success', type=int, default=0, help='成功数量')
    parser.add_argument('--fail', type=int, default=0, help='失败数量')
    
    args = parser.parse_args()
    
    if args.command == 'status':
        display_status()
    elif args.command == 'report':
        generate_report()
    elif args.command == 'update':
        if args.role and args.batch:
            update_role_status(args.role, args.batch, args.success, args.fail)
        elif args.batch:
            # 需要提供角色信息
            logger.error("更新批次状态需要提供角色信息")
        else:
            logger.error("更新状态需要提供角色和批次信息")
    elif args.command == 'complete':
        if args.batch:
            mark_batch_completed(args.batch)
        else:
            logger.error("标记批次完成需要提供批次ID")

if __name__ == '__main__':
    main()
