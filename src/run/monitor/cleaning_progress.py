#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据清理进度监控模块
提供数据清理任务的实时进度统计
"""

import os
import sys
import json
from datetime import datetime
from typing import Dict, Any

# 添加项目路径
_current_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.dirname(os.path.dirname(_current_dir))

# 进度数据文件
PROGRESS_FILE = os.path.join(_src_dir, "data", "cleaning_progress.json")


class CleaningProgressTracker:
    """数据清理进度追踪器"""
    
    def __init__(self):
        self.progress_data = self._load_progress()
    
    def _load_progress(self) -> Dict[str, Any]:
        """加载进度数据"""
        if os.path.exists(PROGRESS_FILE):
            try:
                with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass
        return self._create_default_progress()
    
    def _create_default_progress(self) -> Dict[str, Any]:
        """创建默认进度数据"""
        return {
            "last_updated": datetime.now().isoformat(),
            "total_samples": 0,
            "tasks": {
                "annotation": {
                    "name": "自动标注",
                    "status": "pending",
                    "total": 0,
                    "completed": 0,
                    "failed": 0,
                    "progress": 0.0,
                    "start_time": None,
                    "end_time": None,
                    "message": ""
                },
                "deduplication": {
                    "name": "去重处理",
                    "status": "pending",
                    "total": 0,
                    "completed": 0,
                    "failed": 0,
                    "progress": 0.0,
                    "start_time": None,
                    "end_time": None,
                    "message": ""
                },
                "quality_filter": {
                    "name": "质量过滤",
                    "status": "pending",
                    "total": 0,
                    "completed": 0,
                    "failed": 0,
                    "progress": 0.0,
                    "start_time": None,
                    "end_time": None,
                    "message": ""
                },
                "character_matching": {
                    "name": "角色匹配",
                    "status": "pending",
                    "total": 0,
                    "completed": 0,
                    "failed": 0,
                    "progress": 0.0,
                    "start_time": None,
                    "end_time": None,
                    "message": ""
                },
                "data_export": {
                    "name": "数据导出",
                    "status": "pending",
                    "total": 0,
                    "completed": 0,
                    "failed": 0,
                    "progress": 0.0,
                    "start_time": None,
                    "end_time": None,
                    "message": ""
                }
            },
            "summary": {
                "total_processed": 0,
                "total_valid": 0,
                "total_rejected": 0,
                "total_duplicates": 0,
                "avg_confidence": 0.0,
                "avg_quality_score": 0.0
            }
        }
    
    def _save_progress(self):
        """保存进度数据"""
        os.makedirs(os.path.dirname(PROGRESS_FILE), exist_ok=True)
        with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.progress_data, f, ensure_ascii=False, indent=2)
    
    def update_task_progress(self, task_id: str, completed: int, total: int, 
                            status: str = "running", message: str = ""):
        """
        更新任务进度
        
        Args:
            task_id: 任务ID
            completed: 已完成数量
            total: 总数量
            status: 状态 (pending, running, completed, failed)
            message: 状态消息
        """
        if task_id not in self.progress_data["tasks"]:
            return
        
        task = self.progress_data["tasks"][task_id]
        task["completed"] = completed
        task["total"] = total
        task["status"] = status
        task["message"] = message
        
        if total > 0:
            task["progress"] = min(100.0, (completed / total) * 100)
        else:
            task["progress"] = 0.0
        
        if status == "running" and not task["start_time"]:
            task["start_time"] = datetime.now().isoformat()
        
        if status in ["completed", "failed"]:
            task["end_time"] = datetime.now().isoformat()
        
        self.progress_data["last_updated"] = datetime.now().isoformat()
        self._save_progress()
    
    def start_task(self, task_id: str, total: int = 0):
        """开始任务"""
        self.update_task_progress(task_id, 0, total, "running", "任务开始")
    
    def complete_task(self, task_id: str, completed: int, message: str = "任务完成"):
        """完成任务"""
        if task_id in self.progress_data["tasks"]:
            total = self.progress_data["tasks"][task_id]["total"] or completed
            self.update_task_progress(task_id, completed, total, "completed", message)
    
    def fail_task(self, task_id: str, message: str = "任务失败"):
        """标记任务失败"""
        if task_id in self.progress_data["tasks"]:
            self.progress_data["tasks"][task_id]["status"] = "failed"
            self.progress_data["tasks"][task_id]["message"] = message
            self.progress_data["tasks"][task_id]["end_time"] = datetime.now().isoformat()
            self.progress_data["last_updated"] = datetime.now().isoformat()
            self._save_progress()
    
    def update_summary(self, summary_data: Dict[str, Any]):
        """更新汇总统计"""
        self.progress_data["summary"].update(summary_data)
        self.progress_data["last_updated"] = datetime.now().isoformat()
        self._save_progress()
    
    def set_total_samples(self, total: int):
        """设置总样本数"""
        self.progress_data["total_samples"] = total
        self._save_progress()
    
    def get_progress(self) -> Dict[str, Any]:
        """获取当前进度数据"""
        return self._load_progress()
    
    def reset_progress(self):
        """重置所有进度"""
        self.progress_data = self._create_default_progress()
        self._save_progress()


def get_cleaning_progress() -> Dict[str, Any]:
    """获取数据清理进度（便捷函数）"""
    tracker = CleaningProgressTracker()
    return tracker.get_progress()


def generate_cleaning_progress_html() -> str:
    """生成数据清理进度的HTML面板"""
    progress = get_cleaning_progress()
    
    # 状态颜色映射
    status_colors = {
        "pending": "#9E9E9E",
        "running": "#2196F3",
        "completed": "#4CAF50",
        "failed": "#f44336"
    }
    
    status_labels = {
        "pending": "等待中",
        "running": "运行中",
        "completed": "已完成",
        "failed": "失败"
    }
    
    tasks_html = ""
    for task_id, task in progress["tasks"].items():
        status_color = status_colors.get(task["status"], "#9E9E9E")
        status_label = status_labels.get(task["status"], task["status"])
        
        tasks_html += f"""
        <div class="task-card">
            <div class="task-header">
                <span class="task-name">{task['name']}</span>
                <span class="task-status" style="background: {status_color};">{status_label}</span>
            </div>
            <div class="progress-bar-container">
                <div class="progress-bar" style="width: {task['progress']}%; background: {status_color};"></div>
            </div>
            <div class="task-info">
                <span class="progress-text">{task['completed']}/{task['total']}</span>
                <span class="progress-percent">{task['progress']:.1f}%</span>
            </div>
            {f'<div class="task-message">{task["message"]}</div>' if task["message"] else ''}
        </div>
        """
    
    summary = progress["summary"]
    total_processed = summary["total_processed"]
    total_valid = summary["total_valid"]
    total_rejected = summary["total_rejected"]
    total_duplicates = summary["total_duplicates"]
    
    summary_html = f"""
    <div class="summary-grid">
        <div class="summary-card">
            <div class="summary-icon">📊</div>
            <div class="summary-value">{total_processed}</div>
            <div class="summary-label">总处理数</div>
        </div>
        <div class="summary-card success">
            <div class="summary-icon">✅</div>
            <div class="summary-value">{total_valid}</div>
            <div class="summary-label">有效样本</div>
        </div>
        <div class="summary-card warning">
            <div class="summary-icon">❌</div>
            <div class="summary-value">{total_rejected}</div>
            <div class="summary-label">已过滤</div>
        </div>
        <div class="summary-card info">
            <div class="summary-icon">🔄</div>
            <div class="summary-value">{total_duplicates}</div>
            <div class="summary-label">重复数</div>
        </div>
        <div class="summary-card">
            <div class="summary-icon">🎯</div>
            <div class="summary-value">{summary['avg_confidence']:.2f}</div>
            <div class="summary-label">平均置信度</div>
        </div>
        <div class="summary-card">
            <div class="summary-icon">⭐</div>
            <div class="summary-value">{summary['avg_quality_score']:.2f}</div>
            <div class="summary-label">平均质量分</div>
        </div>
    </div>
    """
    
    html = f"""
    <div class="cleaning-panel">
        <div class="panel-header">
            <h2>🧹 数据清理进度</h2>
            <span class="last-update">更新时间: {progress['last_updated']}</span>
        </div>
        
        {summary_html}
        
        <div class="tasks-section">
            <h3>📋 清理任务列表</h3>
            <div class="tasks-grid">
                {tasks_html}
            </div>
        </div>
    </div>
    
    <style>
        .cleaning-panel {{
            background: #16213e;
            border-radius: 10px;
            border: 1px solid #333;
            padding: 20px;
        }}
        
        .panel-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 1px solid #333;
        }}
        
        .panel-header h2 {{
            color: #667eea;
            margin: 0;
        }}
        
        .last-update {{
            color: #888;
            font-size: 0.9em;
        }}
        
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
            gap: 15px;
            margin-bottom: 25px;
        }}
        
        .summary-card {{
            background: rgba(255, 255, 255, 0.05);
            border-radius: 8px;
            padding: 15px;
            text-align: center;
            border: 1px solid #333;
        }}
        
        .summary-card.success {{ border-color: #4CAF50; }}
        .summary-card.warning {{ border-color: #ff9800; }}
        .summary-card.info {{ border-color: #2196F3; }}
        
        .summary-icon {{
            font-size: 1.8em;
            margin-bottom: 8px;
        }}
        
        .summary-value {{
            font-size: 1.5em;
            font-weight: bold;
            color: #fff;
            margin-bottom: 4px;
        }}
        
        .summary-label {{
            font-size: 0.85em;
            color: #888;
        }}
        
        .tasks-section h3 {{
            color: #fff;
            margin-bottom: 15px;
            font-size: 1.1em;
        }}
        
        .tasks-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 15px;
        }}
        
        .task-card {{
            background: rgba(255, 255, 255, 0.05);
            border-radius: 8px;
            padding: 15px;
            border: 1px solid #333;
        }}
        
        .task-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }}
        
        .task-name {{
            font-weight: bold;
            color: #fff;
        }}
        
        .task-status {{
            padding: 4px 10px;
            border-radius: 15px;
            font-size: 0.8em;
            color: white;
        }}
        
        .progress-bar-container {{
            height: 8px;
            background: #333;
            border-radius: 4px;
            overflow: hidden;
            margin-bottom: 8px;
        }}
        
        .progress-bar {{
            height: 100%;
            transition: width 0.3s ease;
            border-radius: 4px;
        }}
        
        .task-info {{
            display: flex;
            justify-content: space-between;
            font-size: 0.9em;
        }}
        
        .progress-text {{
            color: #888;
        }}
        
        .progress-percent {{
            color: #667eea;
            font-weight: bold;
        }}
        
        .task-message {{
            margin-top: 8px;
            padding-top: 8px;
            border-top: 1px solid #333;
            font-size: 0.85em;
            color: #888;
            font-style: italic;
        }}
    </style>
    """
    
    return html


# 示例用法
if __name__ == "__main__":
    tracker = CleaningProgressTracker()
    
    # 模拟更新进度
    tracker.set_total_samples(1000)
    
    tracker.start_task("annotation", 1000)
    tracker.update_task_progress("annotation", 350, 1000, "running", "正在标注...")
    
    tracker.start_task("deduplication", 1000)
    tracker.update_task_progress("deduplication", 200, 1000, "running", "正在去重...")
    
    tracker.update_summary({
        "total_processed": 550,
        "total_valid": 480,
        "total_rejected": 70,
        "total_duplicates": 45,
        "avg_confidence": 0.85,
        "avg_quality_score": 0.78
    })
    
    print("✅ 进度数据已更新")
    print(json.dumps(tracker.get_progress(), ensure_ascii=False, indent=2))
