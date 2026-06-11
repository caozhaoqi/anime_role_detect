#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据清洗Worker脚本
用于在子进程中运行数据清洗任务，避免PyTorch多线程死锁
"""

import sys
import os
import json
import time
from pathlib import Path

# 添加项目根目录
_current_file = Path(__file__).resolve()
project_root = _current_file.parent.parent
sys.path.insert(0, str(project_root))

from src.data_pipeline.cleaning_pipeline import CleaningPipeline, CleaningConfig


def update_task_status(task_file: str, status: str, **kwargs):
    """更新任务状态文件"""
    try:
        with open(task_file, "r", encoding="utf-8") as f:
            task_config = json.load(f)
        
        task_config["status"] = status
        for key, value in kwargs.items():
            task_config[key] = value
        
        with open(task_file, "w", encoding="utf-8") as f:
            json.dump(task_config, f)
        
        return True
    except Exception as e:
        print(f"更新任务状态失败: {e}", file=sys.stderr)
        return False


def update_db_record(task_id: str, status: str, **kwargs):
    """更新数据库记录"""
    try:
        from src.services.support.database_service import CleaningRecordDB, get_db_service
        from datetime import datetime
        
        db = get_db_service()
        CleaningRecordDB.update_status(db, task_id, status, **kwargs)
        return True
    except Exception as e:
        print(f"更新数据库记录失败: {e}", file=sys.stderr)
        return False


def main():
    if len(sys.argv) < 2:
        print("Usage: python _run_cleaning_worker.py <task_file>", file=sys.stderr)
        sys.exit(1)
    
    task_file = sys.argv[1]
    
    try:
        # 读取任务配置
        with open(task_file, "r", encoding="utf-8") as f:
            task_config = json.load(f)
        
        task_id = task_config["task_id"]
        input_dir = task_config["input_dir"]
        output_dir = task_config["output_dir"]
        
        print(f"开始执行清洗任务: {task_id}", flush=True)
        
        # 构建配置
        config = CleaningConfig(
            enable_deduplication=task_config.get("enable_deduplication", True),
            enable_consistency_filter=task_config.get("enable_consistency_filter", True),
            enable_cluster_filter=task_config.get("enable_cluster_filter", True),
            enable_mislabeled_detector=task_config.get("enable_mislabeled_detector", True),
            enable_danbooru_enrichment=task_config.get("enable_danbooru_enrichment", False),
            similarity_threshold=task_config.get("similarity_threshold", 0.95),
            consistency_threshold=task_config.get("consistency_threshold", 0.25),
            outlier_threshold=task_config.get("outlier_threshold", 0.7),
            text_threshold=task_config.get("text_threshold", 0.2),
            confusion_gap=task_config.get("confusion_gap", 0.08),
            dedup_dry_run=task_config.get("dry_run", False),
            consistency_dry_run=task_config.get("dry_run", False),
            cluster_dry_run=task_config.get("dry_run", False),
            min_images_per_character=task_config.get("min_images_per_character", 5),
        )
        
        # 创建流水线
        pipeline = CleaningPipeline(input_dir, output_dir, config)
        
        # 运行流水线
        report = pipeline.run()
        
        end_time = time.time()
        duration_seconds = end_time - task_config.get("start_time", end_time)
        
        # 更新任务状态文件
        update_task_status(
            task_file,
            "completed",
            end_time=end_time,
            duration_seconds=duration_seconds,
            result={
                "duration_seconds": report.duration_seconds,
                "total_characters": report.total_characters,
                "total_original_images": report.total_original_images,
                "total_cleaned_images": report.total_cleaned_images,
                "total_removed_images": report.total_removed_images,
                "overall_keep_rate": report.overall_keep_rate,
                "dedup_removed": report.dedup_removed,
                "consistency_removed": report.consistency_removed,
                "cluster_removed": report.cluster_removed,
                "mislabeled_removed": report.mislabeled_removed,
                "character_results": report.character_results,
                "report_path": f"{output_dir}/cleaning_report.json",
            },
        )
        
        # 更新数据库记录
        update_db_record(
            task_id,
            "completed",
            completed_at=datetime.now(),
            total_files=report.total_original_images,
            processed_files=report.total_original_images,
            valid_files=report.total_cleaned_images,
            rejected_files=report.total_removed_images,
            duplicate_files=report.dedup_removed,
            report_path=f"{output_dir}/cleaning_report.json",
            duration_seconds=int(report.duration_seconds),
        )
        
        print(f"清洗任务完成: {task_id}", flush=True)
        
    except Exception as e:
        import traceback
        error_msg = str(e)
        print(f"清洗任务失败: {error_msg}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        
        # 更新任务状态文件
        update_task_status(task_file, "failed", error=error_msg, end_time=time.time())
        
        # 更新数据库记录
        try:
            from datetime import datetime
            update_db_record(
                task_id,
                "failed",
                completed_at=datetime.now(),
                error_message=error_msg,
            )
        except:
            pass
        
        sys.exit(1)


if __name__ == "__main__":
    main()