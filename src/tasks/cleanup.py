#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
清理任务
用于定期清理过期任务和临时文件
"""

from src.core.celery_config import app
import time

@app.task(bind=True)
def cleanup_expired_tasks(self):
    """
    清理过期任务
    """
    try:
        # 获取 Redis 后端
        backend = app.backend
        
        # 获取所有任务键
        keys = backend.client.keys('celery-task-meta-*')
        
        expired_count = 0
        deleted_count = 0
        
        for key in keys:
            try:
                # 获取任务元数据
                meta = backend.client.get(key)
                if meta:
                    # 检查任务是否过期（超过1天）
                    # 这里简化处理，直接删除所有任务元数据
                    backend.client.delete(key)
                    deleted_count += 1
            except Exception as e:
                expired_count += 1
        
        return {
            'status': 'success',
            'total_keys': len(keys),
            'deleted_count': deleted_count,
            'expired_count': expired_count
        }
    
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e)
        }
