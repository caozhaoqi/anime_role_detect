#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动化训练流程脚本
集成数据采集、模型训练、评估和部署
"""

import os
import sys
import time
import subprocess
import json
from datetime import datetime

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

try:
    from src.core.logging.global_logger import get_logger
    logger = get_logger("automated_training")
except ModuleNotFoundError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("automated_training")

# 配置
CONFIG = {
    'data_dir': './data/role_images',
    'model_dir': './models',
    'scripts_dir': './scripts',
    'base_model': './models/resnet18_loli8/model_best.pth',
    'max_images_per_role': 50,
    'train_epochs': 15,
    'eval_metrics': ['accuracy', 'f1_score'],
    'notification_email': None
}


def run_command(command, cwd=None):
    """运行命令并返回结果"""
    logger.info(f"执行命令: {command}")
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            cwd=cwd,
            capture_output=True, 
            text=True
        )
        if result.returncode == 0:
            logger.info("命令执行成功")
            return True, result.stdout
        else:
            logger.error(f"命令执行失败: {result.stderr}")
            return False, result.stderr
    except Exception as e:
        logger.error(f"执行命令时出错: {e}")
        return False, str(e)


def collect_data():
    """采集数据"""
    logger.info("=" * 60)
    logger.info("开始数据采集")
    logger.info("=" * 60)
    
    # 运行批量下载脚本（处理批次1）
    download_script = os.path.join(CONFIG['scripts_dir'], 'data_collection', 'batch_download_images.py')
    config_file = os.path.join(CONFIG['scripts_dir'], 'data_collection', 'batch_config.json')
    
    # 先处理批次1（原神核心角色）
    success, output = run_command(f'python3 {download_script} --config {config_file} --batch 1')
    
    if success:
        # 运行状态跟踪脚本
        tracker_script = os.path.join(CONFIG['scripts_dir'], 'data_collection', 'collection_tracker.py')
        run_command(f'python3 {tracker_script} status')
        run_command(f'python3 {tracker_script} report')
        
        logger.info("数据采集完成")
        return True
    else:
        logger.error("数据采集失败")
        return False


def train_model():
    """训练模型"""
    logger.info("=" * 60)
    logger.info("开始模型训练")
    logger.info("=" * 60)
    
    # 检查基础模型是否存在
    if not os.path.exists(CONFIG['base_model']):
        logger.error(f"基础模型不存在: {CONFIG['base_model']}")
        return False
    
    # 运行增量训练脚本
    train_script = os.path.join(CONFIG['scripts_dir'], 'model_training', 'incremental_train.py')
    
    # 生成输出目录
    output_dir = os.path.join(CONFIG['model_dir'], 'incremental')
    os.makedirs(output_dir, exist_ok=True)
    
    # 构建命令
    command = f'python3 {train_script} '
    command += f'--base_model {CONFIG["base_model"]} '
    command += f'--new_data {CONFIG["data_dir"]} '
    command += f'--output_dir {output_dir} '
    command += f'--test_data {CONFIG["data_dir"]}'
    
    success, output = run_command(command)
    
    if success:
        logger.info("模型训练完成")
        return True
    else:
        logger.error("模型训练失败")
        return False


def evaluate_model():
    """评估模型"""
    logger.info("=" * 60)
    logger.info("开始模型评估")
    logger.info("=" * 60)
    
    # 运行模型管理脚本生成摘要
    management_script = os.path.join(CONFIG['scripts_dir'], 'model_training', 'model_management.py')
    success, output = run_command(f'python3 {management_script} summary')
    
    if success:
        logger.info("模型评估完成")
        return True
    else:
        logger.error("模型评估失败")
        return False


def deploy_model():
    """部署模型"""
    logger.info("=" * 60)
    logger.info("开始模型部署")
    logger.info("=" * 60)
    
    # 检查最佳模型链接
    best_model_link = os.path.join(CONFIG['model_dir'], 'best_incremental_model.txt')
    if not os.path.exists(best_model_link):
        logger.error("最佳模型链接不存在")
        return False
    
    # 读取最佳模型路径
    with open(best_model_link, 'r') as f:
        best_model_path = f.read().strip()
    
    logger.info(f"部署最佳模型: {best_model_path}")
    
    # 这里可以添加部署逻辑，比如复制模型到服务目录
    # 例如: cp -r {best_model_path} /path/to/service/models/
    
    logger.info("模型部署完成")
    return True


def generate_report():
    """生成训练报告"""
    logger.info("=" * 60)
    logger.info("生成训练报告")
    logger.info("=" * 60)
    
    # 读取模型摘要
    summary_path = os.path.join(CONFIG['model_dir'], 'model_summary.json')
    if not os.path.exists(summary_path):
        logger.error("模型摘要不存在")
        return False
    
    with open(summary_path, 'r', encoding='utf-8') as f:
        model_summary = json.load(f)
    
    # 生成报告
    report = {
        'timestamp': datetime.now().isoformat(),
        'models': model_summary,
        'data_info': {
            'data_dir': CONFIG['data_dir'],
            'total_roles': len([d for d in os.listdir(CONFIG['data_dir']) if os.path.isdir(os.path.join(CONFIG['data_dir'], d))])
        },
        'config': CONFIG
    }
    
    # 保存报告
    report_path = os.path.join(CONFIG['model_dir'], f'training_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    logger.info(f"训练报告已生成: {report_path}")
    return True


def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("自动化训练流程开始")
    logger.info("=" * 60)
    
    start_time = time.time()
    
    # 步骤1: 采集数据
    if not collect_data():
        logger.error("数据采集失败，流程终止")
        return 1
    
    # 步骤2: 训练模型
    if not train_model():
        logger.error("模型训练失败，流程终止")
        return 1
    
    # 步骤3: 评估模型
    if not evaluate_model():
        logger.error("模型评估失败，流程终止")
        return 1
    
    # 步骤4: 部署模型
    if not deploy_model():
        logger.error("模型部署失败，流程终止")
        return 1
    
    # 步骤5: 生成报告
    if not generate_report():
        logger.error("生成报告失败，流程终止")
        return 1
    
    total_time = time.time() - start_time
    logger.info("=" * 60)
    logger.info("自动化训练流程完成")
    logger.info(f"总耗时: {total_time:.2f}秒")
    logger.info("=" * 60)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
