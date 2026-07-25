#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
推理队列管理器
管理异步推理任务的提交和结果获取
支持Redis和内存队列两种模式，Redis不可用时自动回退到内存队列
"""

import json
import uuid
import time
from typing import Optional, Dict, Any
from datetime import datetime
from enum import Enum
from collections import deque

import redis
from redis.exceptions import ConnectionError as RedisConnectionError
from src.core.logging import get_enhanced_logger as get_logger

logger = get_logger("inference_queue")


class TaskStatus(Enum):
    """任务状态"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


class MemoryQueueManager:
    """
    内存队列管理器（Redis回退方案）
    
    基于内存实现任务队列，功能与Redis版本一致
    """
    
    def __init__(self, task_timeout: int = 600, result_ttl: int = 3600):
        self.task_timeout = task_timeout
        self.result_ttl = result_ttl
        self.task_queue = deque()
        self.processing_queue = {}
        self.results = {}
        self.status = {}
    
    def lpush(self, key: str, value: str):
        self.task_queue.append(value)
    
    def brpop(self, key: str, timeout: int = 1):
        if self.task_queue:
            return (key, self.task_queue.popleft())
        time.sleep(timeout)
        return None
    
    def hset(self, key: str, field: str, value: str):
        self.processing_queue[field] = value
    
    def hgetall(self, key: str):
        return self.processing_queue.copy()
    
    def hdel(self, key: str, field: str):
        if field in self.processing_queue:
            del self.processing_queue[field]
    
    def llen(self, key: str):
        return len(self.task_queue)
    
    def hlen(self, key: str):
        return len(self.processing_queue)
    
    def get(self, key: str):
        return self.status.get(key)
    
    def setex(self, key: str, ttl: int, value: str):
        self.status[key] = value
    
    def delete(self, key: str):
        if key in self.status:
            del self.status[key]
    
    def set(self, key: str, value: str):
        self.status[key] = value


class InferenceQueueManager:
    """
    推理队列管理器
    
    基于Redis实现任务队列，支持：
    - 任务提交
    - 状态查询
    - 结果获取
    - 超时处理
    - Redis不可用时自动回退到内存队列
    """
    
    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        task_timeout: int = 600,
        result_ttl: int = 3600,
    ):
        """
        初始化队列管理器
        
        Args:
            redis_host: Redis主机
            redis_port: Redis端口
            redis_db: Redis数据库
            task_timeout: 任务超时时间（秒），默认10分钟
            result_ttl: 结果保留时间（秒）
        """
        self.task_timeout = task_timeout
        self.result_ttl = result_ttl
        self.use_redis = True
        
        # 队列名称
        self.task_queue = "inference:tasks"
        self.processing_queue = "inference:processing"
        self.result_prefix = "inference:result:"
        self.status_prefix = "inference:status:"
        
        try:
            self.redis_client = redis.Redis(
                host=redis_host,
                port=redis_port,
                db=redis_db,
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2
            )
            self.redis_client.ping()
            logger.info(f"推理队列管理器初始化完成: {redis_host}:{redis_port}")
        except (RedisConnectionError, Exception) as e:
            logger.warning(f"Redis连接失败，回退到内存队列: {e}")
            self.use_redis = False
            self.redis_client = MemoryQueueManager(task_timeout, result_ttl)
    
    def is_connected(self) -> bool:
        """检查队列管理器是否连接正常"""
        if not self.use_redis:
            return True
        try:
            self.redis_client.ping()
            return True
        except Exception:
            return False
    
    def submit_task(
        self,
        image_data: bytes,
        model_name: str = "ViT-B/32",
        top_k: int = 5,
        use_cache: bool = True,
    ) -> str:
        """
        提交推理任务
        
        Args:
            image_data: 图片二进制数据
            model_name: 模型名称
            top_k: 返回前K个结果
            use_cache: 是否使用缓存
            
        Returns:
            任务ID
        """
        task_id = str(uuid.uuid4())
        
        task = {
            "id": task_id,
            "type": "image_classification",
            "model_name": model_name,
            "top_k": top_k,
            "use_cache": use_cache,
            "image_data": image_data.hex(),  # 转为hex字符串
            "created_at": datetime.now().isoformat(),
            "timeout": self.task_timeout,
        }
        
        # 设置初始状态
        self._set_status(task_id, TaskStatus.PENDING)
        
        # 添加到任务队列
        self.redis_client.lpush(self.task_queue, json.dumps(task))
        
        logger.info(f"任务提交成功: {task_id}")
        return task_id
    
    def get_task(self) -> Optional[Dict[str, Any]]:
        """
        获取一个待处理任务（Worker调用）
        
        Returns:
            任务字典或None
        """
        # 从队列中弹出任务（阻塞，超时1秒）
        result = self.redis_client.brpop(self.task_queue, timeout=1)
        
        if result:
            _, task_json = result
            task = json.loads(task_json)
            
            # 更新状态为处理中
            self._set_status(task["id"], TaskStatus.PROCESSING)
            
            # 添加到处理中队列
            self.redis_client.hset(
                self.processing_queue,
                task["id"],
                json.dumps({"start_time": time.time(), "task": task}),
            )
            
            return task
        
        return None
    
    def complete_task(self, task_id: str, result: Dict[str, Any]):
        """
        完成任务（Worker调用）
        
        Args:
            task_id: 任务ID
            result: 结果字典
        """
        # 更新状态
        self._set_status(task_id, TaskStatus.COMPLETED)
        
        # 保存结果
        result_key = f"{self.result_prefix}{task_id}"
        self.redis_client.setex(
            result_key,
            self.result_ttl,
            json.dumps({
                "status": "completed",
                "result": result,
                "completed_at": datetime.now().isoformat(),
            }),
        )
        
        # 从处理中队列移除
        self.redis_client.hdel(self.processing_queue, task_id)
        
        logger.info(f"任务完成: {task_id}")
    
    def fail_task(self, task_id: str, error: str):
        """
        标记任务失败（Worker调用）
        
        Args:
            task_id: 任务ID
            error: 错误信息
        """
        self._set_status(task_id, TaskStatus.FAILED)
        
        result_key = f"{self.result_prefix}{task_id}"
        self.redis_client.setex(
            result_key,
            self.result_ttl,
            json.dumps({
                "status": "failed",
                "error": error,
                "failed_at": datetime.now().isoformat(),
            }),
        )
        
        self.redis_client.hdel(self.processing_queue, task_id)
        
        logger.error(f"任务失败: {task_id}, 错误: {error}")
    
    def get_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        获取任务状态
        
        Args:
            task_id: 任务ID
            
        Returns:
            状态字典或None
        """
        status_key = f"{self.status_prefix}{task_id}"
        status = self.redis_client.get(status_key)
        
        if status:
            return json.loads(status)
        
        return None
    
    def get_result(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        获取任务结果
        
        Args:
            task_id: 任务ID
            
        Returns:
            结果字典或None
        """
        result_key = f"{self.result_prefix}{task_id}"
        result = self.redis_client.get(result_key)
        
        if result:
            return json.loads(result)
        
        return None
    
    def _set_status(self, task_id: str, status: TaskStatus):
        """设置任务状态"""
        status_key = f"{self.status_prefix}{task_id}"
        self.redis_client.setex(
            status_key,
            self.task_timeout + self.result_ttl,
            json.dumps({
                "status": status.value,
                "updated_at": datetime.now().isoformat(),
            }),
        )
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """
        获取队列统计信息
        
        Returns:
            统计字典
        """
        pending = self.redis_client.llen(self.task_queue)
        processing = self.redis_client.hlen(self.processing_queue)
        
        return {
            "pending_tasks": pending,
            "processing_tasks": processing,
            "total_active": pending + processing,
        }
    
    def cleanup_expired(self):
        """清理过期任务"""
        # 检查处理中超时的任务
        processing_tasks = self.redis_client.hgetall(self.processing_queue)
        current_time = time.time()
        
        for task_id, task_info_json in processing_tasks.items():
            task_info = json.loads(task_info_json)
            start_time = task_info.get("start_time", 0)
            
            if current_time - start_time > self.task_timeout:
                # 标记为超时
                self.fail_task(task_id, "Task timeout")
                logger.warning(f"任务超时: {task_id}")


# 全局队列管理器实例
_queue_manager = None


def get_queue_manager() -> InferenceQueueManager:
    """获取队列管理器单例"""
    global _queue_manager
    if _queue_manager is None:
        import os
        _queue_manager = InferenceQueueManager(
            redis_host=os.environ.get("REDIS_HOST", "localhost"),
            redis_port=int(os.environ.get("REDIS_PORT", 6379)),
        )
    return _queue_manager
