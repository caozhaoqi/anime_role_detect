#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
推理Worker
从Redis队列中获取任务并执行推理
"""

import os
import sys
import time
import json
import signal
import threading
from pathlib import Path
from typing import Optional, Dict, Any
import argparse

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent

from src.core.logging.global_logger import get_logger
from src.services.inference_queue.queue_manager import InferenceQueueManager, get_queue_manager

logger = get_logger("inference_worker")


class InferenceWorker:
    """
    推理Worker
    
    持续从Redis队列获取任务并执行推理
    """
    
    def __init__(
        self,
        worker_id: Optional[str] = None,
        model_name: str = "ViT-B/32",
        max_tasks: int = -1,
        device: Optional[str] = None,
    ):
        """
        初始化Worker
        
        Args:
            worker_id: Worker ID，None则自动生成
            model_name: 使用的模型名称
            max_tasks: 最大处理任务数，-1表示无限
            device: 运行设备
        """
        self.worker_id = worker_id or f"worker_{os.getpid()}"
        self.model_name = model_name
        self.max_tasks = max_tasks
        self.device = device
        
        self.queue_manager = get_queue_manager()
        self.embedder = None
        self.feature_store = None
        
        self.running = False
        self.tasks_processed = 0
        self.tasks_failed = 0
        
        # 注册信号处理
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        
        logger.info(f"Worker {self.worker_id} 初始化完成")
    
    def _signal_handler(self, signum, frame):
        """处理终止信号"""
        logger.info(f"Worker {self.worker_id} 收到信号 {signum}，正在停止...")
        self.running = False
    
    def _init_models(self):
        """初始化模型"""
        if self.embedder is None:
            from src.core.recognition.clip_embedder_cached import CLIPEmbedderCached
            
            logger.info(f"加载CLIP模型: {self.model_name}")
            self.embedder = CLIPEmbedderCached(
                model_name=self.model_name,
                device=self.device,
                cache_dir="./clip_cache",
            )
        
        if self.feature_store is None:
            from src.core.recognition.feature_store import FeatureStore
            
            logger.info("加载特征库...")
            self.feature_store = FeatureStore(dimension=self.embedder.embedding_dim)
            
            # 尝试加载已有的特征库
            feature_store_path = "data/feature_store"
            if os.path.exists(feature_store_path + ".faiss"):
                self.feature_store.load(feature_store_path)
                logger.info("特征库加载完成")
    
    def _process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理单个任务
        
        Args:
            task: 任务字典
            
        Returns:
            结果字典
        """
        task_id = task["id"]
        model_name = task.get("model_name", self.model_name)
        top_k = task.get("top_k", 5)
        use_cache = task.get("use_cache", True)
        
        # 解码图片数据
        image_data = bytes.fromhex(task["image_data"])
        
        # 保存为临时文件
        temp_path = f"/tmp/inference_{task_id}.jpg"
        with open(temp_path, "wb") as f:
            f.write(image_data)
        
        try:
            # 提取特征
            feature = self.embedder.embed_image(temp_path)
            
            if feature is None:
                raise ValueError("特征提取失败")
            
            # 搜索特征库
            results = self.feature_store.search(feature, top_k=top_k)
            
            return {
                "task_id": task_id,
                "success": True,
                "results": [
                    {
                        "character": char,
                        "similarity": float(sim),
                    }
                    for char, sim in results
                ],
                "model_name": model_name,
            }
            
        finally:
            # 清理临时文件
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def start(self):
        """启动Worker"""
        logger.info(f"Worker {self.worker_id} 启动")
        
        self.running = True
        
        # 初始化模型
        self._init_models()
        
        while self.running:
            try:
                # 获取任务
                task = self.queue_manager.get_task()
                
                if task is None:
                    # 队列为空，短暂休息
                    time.sleep(0.5)
                    continue
                
                task_id = task["id"]
                logger.info(f"处理任务: {task_id}")
                
                try:
                    # 处理任务
                    result = self._process_task(task)
                    
                    # 完成任务
                    self.queue_manager.complete_task(task_id, result)
                    self.tasks_processed += 1
                    
                    logger.info(f"任务完成: {task_id} (已处理: {self.tasks_processed})")
                    
                except Exception as e:
                    logger.error(f"任务处理失败: {task_id}, 错误: {e}")
                    self.queue_manager.fail_task(task_id, str(e))
                    self.tasks_failed += 1
                
                # 检查是否达到最大任务数
                if self.max_tasks > 0 and self.tasks_processed >= self.max_tasks:
                    logger.info(f"达到最大任务数 {self.max_tasks}，Worker停止")
                    self.running = False
                    
            except Exception as e:
                logger.error(f"Worker错误: {e}")
                time.sleep(1)
        
        logger.info(f"Worker {self.worker_id} 停止")
        logger.info(f"统计: 成功 {self.tasks_processed}, 失败 {self.tasks_failed}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取Worker统计信息"""
        return {
            "worker_id": self.worker_id,
            "tasks_processed": self.tasks_processed,
            "tasks_failed": self.tasks_failed,
            "running": self.running,
        }


class WorkerPool:
    """
    Worker池
    管理多个Worker进程
    """
    
    def __init__(
        self,
        num_workers: int = 4,
        model_name: str = "ViT-B/32",
    ):
        """
        初始化Worker池
        
        Args:
            num_workers: Worker数量
            model_name: 模型名称
        """
        self.num_workers = num_workers
        self.model_name = model_name
        self.workers = []
        self.threads = []
    
    def start(self):
        """启动所有Worker"""
        logger.info(f"启动Worker池: {self.num_workers}个Worker")
        
        for i in range(self.num_workers):
            worker = InferenceWorker(
                worker_id=f"worker_{i}",
                model_name=self.model_name,
            )
            
            thread = threading.Thread(target=worker.start, daemon=True)
            thread.start()
            
            self.workers.append(worker)
            self.threads.append(thread)
        
        logger.info("Worker池启动完成")
    
    def stop(self):
        """停止所有Worker"""
        logger.info("停止Worker池...")
        
        for worker in self.workers:
            worker.running = False
        
        for thread in self.threads:
            thread.join(timeout=5)
        
        logger.info("Worker池已停止")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取Worker池统计"""
        total_processed = sum(w.tasks_processed for w in self.workers)
        total_failed = sum(w.tasks_failed for w in self.workers)
        
        return {
            "num_workers": self.num_workers,
            "total_processed": total_processed,
            "total_failed": total_failed,
            "workers": [w.get_stats() for w in self.workers],
        }


def main():
    parser = argparse.ArgumentParser(description="推理Worker")
    parser.add_argument("--worker-id", type=str, default=None, help="Worker ID")
    parser.add_argument("--model", type=str, default="ViT-B/32", help="模型名称")
    parser.add_argument("--max-tasks", type=int, default=-1, help="最大任务数")
    parser.add_argument("--device", type=str, default=None, help="运行设备")
    parser.add_argument("--num-workers", type=int, default=1, help="Worker数量")
    
    args = parser.parse_args()
    
    if args.num_workers > 1:
        # 启动Worker池
        pool = WorkerPool(
            num_workers=args.num_workers,
            model_name=args.model,
        )
        pool.start()
        
        try:
            while True:
                time.sleep(10)
                stats = pool.get_stats()
                logger.info(f"Worker池统计: {stats}")
        except KeyboardInterrupt:
            pool.stop()
    else:
        # 启动单个Worker
        worker = InferenceWorker(
            worker_id=args.worker_id,
            model_name=args.model,
            max_tasks=args.max_tasks,
            device=args.device,
        )
        worker.start()


if __name__ == "__main__":
    main()
