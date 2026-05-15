#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
清理不再使用或无法使用的训练脚本
"""
import os
import sys

MODEL_TRAINING_DIR = '/Users/caozhaoqi/PycharmProjects/anime_role_detect/scripts/model_training'

# 已知有用的脚本（保留）
KEEP_SCRIPTS = [
    '__init__.py',
    'CharacterClassifier.py',
    'CharacterClassifier_v2.py',
    'train_loli_full_data.py',
    'train_loli_optimized.py',
    'advanced_training.py',
    'model_management.py'
]

# 不再使用的脚本（删除）
DELETE_SCRIPTS = [
    'train_loli_models.py',      # 旧的8角色训练脚本
    'train_loli_20images.py',    # 仅用20张图片训练的脚本
    'train_loli_anti_overfit.py',# 旧的防过拟合脚本
    'train_loli_reorganized.py', # 旧的重组数据训练脚本
    'incremental_train.py',      # 增量训练（可能已过时）
    'scheduled_training.py',     # 定时训练（可能已过时）
    'automated_training.py',     # 自动化训练（可能已过时）
    'train_three_models.py',     # 同时训练三个模型（旧版）
    'priority_collect.py',       # 优先级采集（非训练脚本）
    'start_priority_collect.py', # 启动优先级采集（非训练脚本）
    'convert_to_coreml.py',      # CoreML转换（可能不再使用）
    'convert_wd_tagger_to_coreml.py', # CoreML转换（可能不再使用）
    'create_end_to_end_coreml_model.py', # CoreML模型创建（可能不再使用）
    'create_nsfw_model.py',      # NSFW模型创建（可能不再使用）
    'export_model_for_serving.py', # 服务导出（可能不再使用）
    'benchmark_full_data.py',    # 基准测试（可能不再使用）
    'confusion_matrix_analysis.py', # 混淆矩阵分析（可能不再使用）
    'balance_dataset.py'         # 数据集平衡（可能不再使用）
]

def clean_scripts():
    """清理不再使用的训练脚本"""
    print("🔍 清理不再使用的训练脚本")
    print("=" * 60)
    
    deleted_count = 0
    kept_count = 0
    
    for filename in os.listdir(MODEL_TRAINING_DIR):
        filepath = os.path.join(MODEL_TRAINING_DIR, filename)
        
        if os.path.isfile(filepath):
            if filename in DELETE_SCRIPTS:
                try:
                    os.remove(filepath)
                    print(f"🗑️ 删除: {filename}")
                    deleted_count += 1
                except Exception as e:
                    print(f"❌ 删除失败 {filename}: {e}")
            elif filename in KEEP_SCRIPTS:
                print(f"✅ 保留: {filename}")
                kept_count += 1
            else:
                print(f"⚠️ 未分类: {filename}")
    
    print("-" * 60)
    print(f"已删除: {deleted_count} 个脚本")
    print(f"保留: {kept_count} 个脚本")
    print("✅ 清理完成")

if __name__ == '__main__':
    clean_scripts()