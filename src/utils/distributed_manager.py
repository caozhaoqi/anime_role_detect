#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分布式任务管理器

支持分布式采集和处理，实现任务的分配、执行和监控
"""

import os
import time
import logging
import pickle
import threading
import queue
import json
from datetime import datetime
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class Task(ABC):
    """
    任务抽象基类
    """
    
    def __init__(self, task_id, task_type, **kwargs):
        """
        初始化任务
        
        Args:
            task_id: 任务ID
            task_type: 任务类型
            **kwargs: 任务参数
        """
        self.task_id = task_id
        self.task_type = task_type
        self.status = 'pending'  # pending, running, completed, failed
        self.priority = kwargs.get('priority', 0)
        self.created_time = datetime.now()
        self.start_time = None
        self.end_time = None
        self.result = None
        self.error = None
        self.retry_count = 0
        self.max_retries = kwargs.get('max_retries', 3)
        
        # 任务参数
        self.params = kwargs
    
    @abstractmethod
    def execute(self):
        """
        执行任务
        
        Returns:
            任务执行结果
        """
        pass
    
    def to_dict(self):
        """
        转换为字典
        
        Returns:
            任务字典
        """
        return {
            'task_id': self.task_id,
            'task_type': self.task_type,
            'status': self.status,
            'priority': self.priority,
            'created_time': self.created_time.isoformat(),
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'result': self.result,
            'error': str(self.error) if self.error else None,
            'retry_count': self.retry_count,
            'max_retries': self.max_retries,
            'params': self.params
        }
    
    @classmethod
    def from_dict(cls, task_dict):
        """
        从字典创建任务
        
        Args:
            task_dict: 任务字典
            
        Returns:
            任务实例
        """
        task = cls(
            task_id=task_dict['task_id'],
            task_type=task_dict['task_type'],
            **task_dict['params']
        )
        task.status = task_dict['status']
        task.priority = task_dict['priority']
        task.created_time = datetime.fromisoformat(task_dict['created_time'])
        task.start_time = datetime.fromisoformat(task_dict['start_time']) if task_dict['start_time'] else None
        task.end_time = datetime.fromisoformat(task_dict['end_time']) if task_dict['end_time'] else None
        task.result = task_dict['result']
        task.error = task_dict['error']
        task.retry_count = task_dict['retry_count']
        task.max_retries = task_dict['max_retries']
        return task


class ImageCollectionTask(Task):
    """
    图像采集任务
    """
    
    def __init__(self, task_id, series, character, max_images=50, **kwargs):
        """
        初始化图像采集任务
        
        Args:
            task_id: 任务ID
            series: 系列名称
            character: 角色名称
            max_images: 最大图像数
            **kwargs: 其他参数
        """
        super().__init__(task_id, 'image_collection', series=series, character=character, max_images=max_images, **kwargs)
        self.series = series
        self.character = character
        self.max_images = max_images
    
    def execute(self):
        """
        执行图像采集任务
        
        Returns:
            采集结果
        """
        try:
            from src.data_collection.keyword_based_collector import KeywordBasedDataCollector
            
            # 创建采集器实例
            collector = KeywordBasedDataCollector(
                output_dir=self.params.get('output_dir', 'data/train')
            )
            
            # 执行采集
            result = collector._process_character(
                series=self.series,
                character_name=self.character,
                max_images=self.max_images
            )
            
            return {
                'character': f"{self.series}_{self.character}",
                'image_count': result,
                'status': 'success'
            }
        except Exception as e:
            logger.error(f"执行图像采集任务失败: {e}")
            raise


class TaskQueue:
    """
    任务队列
    """
    
    def __init__(self, queue_path='./task_queue'):
        """
        初始化任务队列
        
        Args:
            queue_path: 队列存储路径
        """
        self.queue_path = queue_path
        self.task_queue = queue.PriorityQueue()
        self.processed_tasks = {}
        self.lock = threading.Lock()
        
        # 创建队列目录
        os.makedirs(queue_path, exist_ok=True)
        
        # 加载持久化的任务
        self._load_tasks()
    
    def add_task(self, task):
        """
        添加任务
        
        Args:
            task: 任务实例
        """
        with self.lock:
            # 将任务添加到优先级队列
            self.task_queue.put((-task.priority, task.task_id, task))
            
            # 持久化任务
            self._save_task(task)
            
            logger.info(f"添加任务: {task.task_id}, 类型: {task.task_type}, 优先级: {task.priority}")
    
    def get_task(self):
        """
        获取任务
        
        Returns:
            任务实例或None
        """
        try:
            with self.lock:
                if not self.task_queue.empty():
                    _, task_id, task = self.task_queue.get(block=False)
                    task.status = 'running'
                    task.start_time = datetime.now()
                    
                    # 更新任务状态
                    self._save_task(task)
                    
                    logger.info(f"获取任务: {task_id}, 类型: {task.task_type}")
                    return task
                return None
        except queue.Empty:
            return None
    
    def complete_task(self, task, result):
        """
        完成任务
        
        Args:
            task: 任务实例
            result: 任务结果
        """
        with self.lock:
            task.status = 'completed'
            task.end_time = datetime.now()
            task.result = result
            
            # 保存任务结果
            self.processed_tasks[task.task_id] = task
            self._save_task(task)
            
            logger.info(f"完成任务: {task.task_id}, 结果: {result}")
    
    def fail_task(self, task, error):
        """
        任务失败
        
        Args:
            task: 任务实例
            error: 错误信息
        """
        with self.lock:
            task.status = 'failed'
            task.end_time = datetime.now()
            task.error = error
            task.retry_count += 1
            
            # 如果重试次数未达到上限，重新添加任务
            if task.retry_count < task.max_retries:
                task.status = 'pending'
                task.start_time = None
                task.end_time = None
                self.task_queue.put((-task.priority, task.task_id, task))
                logger.info(f"任务失败，重新添加: {task.task_id}, 重试次数: {task.retry_count}")
            else:
                # 保存失败任务
                self.processed_tasks[task.task_id] = task
                logger.error(f"任务最终失败: {task.task_id}, 错误: {error}")
            
            # 更新任务状态
            self._save_task(task)
    
    def _save_task(self, task):
        """
        保存任务
        
        Args:
            task: 任务实例
        """
        task_file = os.path.join(self.queue_path, f"task_{task.task_id}.json")
        with open(task_file, 'w', encoding='utf-8') as f:
            json.dump(task.to_dict(), f, ensure_ascii=False, indent=2)
    
    def _load_tasks(self):
        """
        加载任务
        """
        try:
            for filename in os.listdir(self.queue_path):
                if filename.startswith('task_') and filename.endswith('.json'):
                    task_file = os.path.join(self.queue_path, filename)
                    try:
                        with open(task_file, 'r', encoding='utf-8') as f:
                            task_dict = json.load(f)
                            
                        # 重建任务实例
                        if task_dict['task_type'] == 'image_collection':
                            task = ImageCollectionTask.from_dict(task_dict)
                            
                            # 只加载未完成的任务
                            if task.status in ['pending', 'running']:
                                self.task_queue.put((-task.priority, task.task_id, task))
                                logger.info(f"加载任务: {task.task_id}, 状态: {task.status}")
                            else:
                                self.processed_tasks[task.task_id] = task
                    except Exception as e:
                        logger.error(f"加载任务失败 {filename}: {e}")
        except Exception as e:
            logger.error(f"加载任务队列失败: {e}")
    
    def get_queue_size(self):
        """
        获取队列大小
        
        Returns:
            队列大小
        """
        return self.task_queue.qsize()
    
    def get_stats(self):
        """
        获取队列统计信息
        
        Returns:
            统计信息字典
        """
        stats = {
            'queue_size': self.get_queue_size(),
            'processed_tasks': len(self.processed_tasks),
            'pending_tasks': 0,
            'running_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0
        }
        
        # 统计任务状态
        for task in self.processed_tasks.values():
            if task.status == 'completed':
                stats['completed_tasks'] += 1
            elif task.status == 'failed':
                stats['failed_tasks'] += 1
        
        # 统计队列中的任务
        temp_queue = []
        while not self.task_queue.empty():
            _, task_id, task = self.task_queue.get()
            temp_queue.append((-task.priority, task_id, task))
            if task.status == 'pending':
                stats['pending_tasks'] += 1
            elif task.status == 'running':
                stats['running_tasks'] += 1
        
        # 将任务放回队列
        for item in temp_queue:
            self.task_queue.put(item)
        
        return stats


class Worker:
    """
    工作节点
    """
    
    def __init__(self, worker_id, task_queue, max_workers=5):
        """
        初始化工作节点
        
        Args:
            worker_id: 工作节点ID
            task_queue: 任务队列
            max_workers: 最大工作线程数
        """
        self.worker_id = worker_id
        self.task_queue = task_queue
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.running = False
        self.thread = None
    
    def start(self):
        """
        启动工作节点
        """
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        logger.info(f"启动工作节点: {self.worker_id}, 最大线程数: {self.max_workers}")
    
    def stop(self):
        """
        停止工作节点
        """
        self.running = False
        if self.thread:
            self.thread.join(timeout=10)
        self.executor.shutdown(wait=False)
        logger.info(f"停止工作节点: {self.worker_id}")
    
    def _run(self):
        """
        运行工作节点
        """
        while self.running:
            try:
                # 获取任务
                task = self.task_queue.get_task()
                if task:
                    # 提交任务到线程池
                    future = self.executor.submit(self._process_task, task)
                    future.add_done_callback(lambda f, t=task: self._task_done(f, t))
                else:
                    # 队列为空，等待一段时间
                    time.sleep(1)
            except Exception as e:
                logger.error(f"工作节点错误: {e}")
                time.sleep(1)
    
    def _process_task(self, task):
        """
        处理任务
        
        Args:
            task: 任务实例
            
        Returns:
            任务结果
        """
        try:
            logger.info(f"处理任务: {task.task_id}, 类型: {task.task_type}")
            result = task.execute()
            logger.info(f"任务处理完成: {task.task_id}")
            return result
        except Exception as e:
            logger.error(f"任务执行失败: {task.task_id}, 错误: {e}")
            raise
    
    def _task_done(self, future, task):
        """
        任务完成回调
        
        Args:
            future: 任务未来对象
            task: 任务实例
        """
        try:
            result = future.result()
            self.task_queue.complete_task(task, result)
        except Exception as e:
            self.task_queue.fail_task(task, e)


class DistributedManager:
    """
    分布式管理器
    """
    
    def __init__(self, manager_id='manager_001', queue_path='./task_queue', max_workers=5):
        """
        初始化分布式管理器
        
        Args:
            manager_id: 管理器ID
            queue_path: 队列存储路径
            max_workers: 最大工作线程数
        """
        self.manager_id = manager_id
        self.task_queue = TaskQueue(queue_path)
        self.worker = Worker(f"worker_{manager_id}", self.task_queue, max_workers)
        self.stats = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'start_time': datetime.now()
        }
    
    def start(self):
        """
        启动分布式管理器
        """
        self.worker.start()
        logger.info(f"启动分布式管理器: {self.manager_id}")
    
    def stop(self):
        """
        停止分布式管理器
        """
        self.worker.stop()
        logger.info(f"停止分布式管理器: {self.manager_id}")
    
    def add_task(self, task_type, **kwargs):
        """
        添加任务
        
        Args:
            task_type: 任务类型
            **kwargs: 任务参数
            
        Returns:
            任务ID
        """
        import uuid
        
        task_id = str(uuid.uuid4())
        
        if task_type == 'image_collection':
            task = ImageCollectionTask(
                task_id=task_id,
                task_type=task_type,
                **kwargs
            )
        else:
            raise ValueError(f"未知的任务类型: {task_type}")
        
        self.task_queue.add_task(task)
        self.stats['total_tasks'] += 1
        
        return task_id
    
    def add_image_collection_tasks(self, series, characters, max_images=50):
        """
        添加图像采集任务
        
        Args:
            series: 系列名称
            characters: 角色列表
            max_images: 每个角色的最大图像数
            
        Returns:
            任务ID列表
        """
        task_ids = []
        
        for character in characters:
            task_id = self.add_task(
                'image_collection',
                series=series,
                character=character,
                max_images=max_images
            )
            task_ids.append(task_id)
        
        logger.info(f"添加了 {len(task_ids)} 个图像采集任务")
        return task_ids
    
    def get_stats(self):
        """
        获取统计信息
        
        Returns:
            统计信息字典
        """
        queue_stats = self.task_queue.get_stats()
        
        # 更新统计信息
        self.stats['completed_tasks'] = queue_stats['completed_tasks']
        self.stats['failed_tasks'] = queue_stats['failed_tasks']
        self.stats['queue_stats'] = queue_stats
        
        return self.stats
    
    def monitor(self, interval=10):
        """
        监控任务执行状态
        
        Args:
            interval: 监控间隔（秒）
        """
        while True:
            try:
                stats = self.get_stats()
                logger.info(f"监控统计: {json.dumps(stats, indent=2)}")
                time.sleep(interval)
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"监控错误: {e}")
                time.sleep(interval)
