"""
链路存储服务 - 支持内存存储和Redis存储

用于持久化存储追踪数据，支持查询和聚合分析
"""

import json
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from abc import ABC, abstractmethod

from src.core.logging.global_logger import get_logger
from src.utils.monitoring.tracing import Trace, Span

logger = get_logger("trace_storage_service")


class TraceStorage(ABC):
    """
    链路存储抽象基类
    """

    @abstractmethod
    def store_trace(self, trace: Trace):
        """存储Trace"""
        pass

    @abstractmethod
    def get_trace(self, trace_id: str) -> Optional[dict]:
        """获取Trace"""
        pass

    @abstractmethod
    def get_all_traces(self) -> List[dict]:
        """获取所有Trace"""
        pass

    @abstractmethod
    def get_traces_by_time_range(
        self, start_time: float, end_time: float
    ) -> List[dict]:
        """按时间范围获取Trace"""
        pass

    @abstractmethod
    def delete_trace(self, trace_id: str):
        """删除Trace"""
        pass

    @abstractmethod
    def clear_expired_traces(self, max_age_seconds: int):
        """清理过期的Trace"""
        pass

    @abstractmethod
    def get_trace_count(self) -> int:
        """获取Trace数量"""
        pass


class MemoryTraceStorage(TraceStorage):
    """
    内存存储实现
    """

    def __init__(self, max_traces: int = 1000):
        self.traces: Dict[str, dict] = {}
        self.max_traces = max_traces

    def store_trace(self, trace: Trace):
        """存储Trace"""
        # 如果超过最大数量，删除最旧的
        if len(self.traces) >= self.max_traces:
            oldest_key = min(self.traces.keys(), key=lambda k: self.traces[k]["start_time"])
            del self.traces[oldest_key]
        
        trace_data = trace.to_dict()
        self.traces[trace.trace_id] = trace_data
        logger.debug(f"已存储Trace: {trace.trace_id[:8]}...")

    def get_trace(self, trace_id: str) -> Optional[dict]:
        """获取Trace"""
        return self.traces.get(trace_id)

    def get_all_traces(self) -> List[dict]:
        """获取所有Trace"""
        return list(self.traces.values())

    def get_traces_by_time_range(
        self, start_time: float, end_time: float
    ) -> List[dict]:
        """按时间范围获取Trace"""
        return [
            trace for trace in self.traces.values()
            if start_time <= trace.get("start_time", 0) <= end_time
        ]

    def delete_trace(self, trace_id: str):
        """删除Trace"""
        if trace_id in self.traces:
            del self.traces[trace_id]
            logger.debug(f"已删除Trace: {trace_id[:8]}...")

    def clear_expired_traces(self, max_age_seconds: int):
        """清理过期的Trace"""
        now = time.time()
        expired_ids = [
            trace_id for trace_id, trace in self.traces.items()
            if now - trace.get("start_time", 0) > max_age_seconds
        ]
        for trace_id in expired_ids:
            del self.traces[trace_id]
        logger.debug(f"清理了 {len(expired_ids)} 个过期Trace")

    def get_trace_count(self) -> int:
        """获取Trace数量"""
        return len(self.traces)


class RedisTraceStorage(TraceStorage):
    """
    Redis存储实现
    """

    def __init__(self, redis_client):
        self.redis_client = redis_client
        self.key_prefix = "trace:"
        self.index_key = "trace:index"

    def _get_key(self, trace_id: str) -> str:
        """获取Redis键"""
        return f"{self.key_prefix}{trace_id}"

    def store_trace(self, trace: Trace):
        """存储Trace"""
        trace_data = trace.to_dict()
        key = self._get_key(trace.trace_id)
        
        # 存储Trace数据
        self.redis_client.set(key, json.dumps(trace_data))
        
        # 添加到索引（存储trace_id和时间戳）
        self.redis_client.zadd(self.index_key, {trace.trace_id: trace_data["start_time"]})
        
        logger.debug(f"已存储Trace到Redis: {trace.trace_id[:8]}...")

    def get_trace(self, trace_id: str) -> Optional[dict]:
        """获取Trace"""
        key = self._get_key(trace_id)
        data = self.redis_client.get(key)
        if data:
            return json.loads(data)
        return None

    def get_all_traces(self) -> List[dict]:
        """获取所有Trace"""
        # 获取所有trace_id
        trace_ids = self.redis_client.zrange(self.index_key, 0, -1)
        
        traces = []
        for trace_id in trace_ids:
            # decode_responses=True时，返回的已经是字符串
            trace_id_str = trace_id if isinstance(trace_id, str) else trace_id.decode()
            key = self._get_key(trace_id_str)
            data = self.redis_client.get(key)
            if data:
                traces.append(json.loads(data))
        
        return traces

    def get_traces_by_time_range(
        self, start_time: float, end_time: float
    ) -> List[dict]:
        """按时间范围获取Trace"""
        # 使用ZRANGEBYSCORE获取时间范围内的trace_id
        trace_ids = self.redis_client.zrangebyscore(
            self.index_key, start_time, end_time
        )
        
        traces = []
        for trace_id in trace_ids:
            # decode_responses=True时，返回的已经是字符串
            trace_id_str = trace_id if isinstance(trace_id, str) else trace_id.decode()
            key = self._get_key(trace_id_str)
            data = self.redis_client.get(key)
            if data:
                traces.append(json.loads(data))
        
        return traces

    def delete_trace(self, trace_id: str):
        """删除Trace"""
        key = self._get_key(trace_id)
        self.redis_client.delete(key)
        self.redis_client.zrem(self.index_key, trace_id)
        logger.debug(f"已从Redis删除Trace: {trace_id[:8]}...")

    def clear_expired_traces(self, max_age_seconds: int):
        """清理过期的Trace"""
        cutoff_time = time.time() - max_age_seconds
        
        # 获取过期的trace_id
        expired_ids = self.redis_client.zrangebyscore(
            self.index_key, 0, cutoff_time
        )
        
        # 删除过期的Trace
        for trace_id in expired_ids:
            # decode_responses=True时，返回的已经是字符串
            trace_id_str = trace_id if isinstance(trace_id, str) else trace_id.decode()
            key = self._get_key(trace_id_str)
            self.redis_client.delete(key)
            self.redis_client.zrem(self.index_key, trace_id_str)
        
        logger.debug(f"从Redis清理了 {len(expired_ids)} 个过期Trace")

    def get_trace_count(self) -> int:
        """获取Trace数量"""
        return self.redis_client.zcard(self.index_key)


class TraceStorageService:
    """
    链路存储服务 - 统一管理存储实现
    """

    _instance: Optional["TraceStorageService"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "initialized") and self.initialized:
            return
        
        # 优先使用Redis存储，支持跨进程共享追踪数据
        try:
            import redis
            redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
            # 测试Redis连接
            redis_client.ping()
            self.storage: TraceStorage = RedisTraceStorage(redis_client)
            logger.info("链路存储服务初始化完成（使用Redis存储）")
        except Exception as e:
            # Redis不可用时使用内存存储
            logger.warning(f"Redis连接失败，使用内存存储: {e}")
            self.storage: TraceStorage = MemoryTraceStorage()
            logger.info("链路存储服务初始化完成（使用内存存储）")
        
        self.initialized = True

    def set_storage(self, storage: TraceStorage):
        """设置存储实现"""
        self.storage = storage

    def store_trace(self, trace: Trace):
        """存储Trace"""
        self.storage.store_trace(trace)

    def get_trace(self, trace_id: str) -> Optional[dict]:
        """获取Trace"""
        return self.storage.get_trace(trace_id)

    def get_all_traces(self) -> List[dict]:
        """获取所有Trace"""
        return self.storage.get_all_traces()

    def get_traces_by_time_range(
        self, start_time: float, end_time: float
    ) -> List[dict]:
        """按时间范围获取Trace"""
        return self.storage.get_traces_by_time_range(start_time, end_time)

    def get_recent_traces(self, limit: int = 50) -> List[dict]:
        """获取最近的Trace"""
        all_traces = self.storage.get_all_traces()
        all_traces.sort(key=lambda t: t.get("start_time", 0), reverse=True)
        return all_traces[:limit]

    def delete_trace(self, trace_id: str):
        """删除Trace"""
        self.storage.delete_trace(trace_id)

    def clear_expired_traces(self, max_age_hours: int = 24):
        """清理过期的Trace"""
        self.storage.clear_expired_traces(max_age_hours * 3600)

    def get_trace_count(self) -> int:
        """获取Trace数量"""
        return self.storage.get_trace_count()

    def get_aggregated_stats(self, hours: int = 24) -> dict:
        """获取聚合统计信息"""
        end_time = time.time()
        start_time = end_time - hours * 3600
        
        traces = self.storage.get_traces_by_time_range(start_time, end_time)
        
        if not traces:
            return {
                "total_traces": 0,
                "avg_duration_ms": 0,
                "min_duration_ms": 0,
                "max_duration_ms": 0,
                "error_count": 0,
                "success_count": 0,
                "status_distribution": {},
                "endpoint_distribution": {},
            }
        
        durations = [t.get("duration_ms", 0) for t in traces]
        error_count = sum(1 for t in traces if t.get("status") == "ERROR")
        success_count = len(traces) - error_count
        
        # 统计状态分布
        status_dist = {}
        for t in traces:
            status = t.get("status", "UNKNOWN")
            status_dist[status] = status_dist.get(status, 0) + 1
        
        # 统计端点分布（从Span中提取）
        endpoint_dist = {}
        for t in traces:
            spans = t.get("spans", [])
            for span in spans:
                endpoint = span.get("attributes", {}).get("http.path", "unknown")
                endpoint_dist[endpoint] = endpoint_dist.get(endpoint, 0) + 1
        
        return {
            "total_traces": len(traces),
            "avg_duration_ms": round(sum(durations) / len(durations), 2),
            "min_duration_ms": min(durations),
            "max_duration_ms": max(durations),
            "error_count": error_count,
            "success_count": success_count,
            "error_rate": round(error_count / len(traces) * 100, 2),
            "status_distribution": status_dist,
            "endpoint_distribution": endpoint_dist,
            "time_range": {
                "start": datetime.fromtimestamp(start_time).isoformat(),
                "end": datetime.fromtimestamp(end_time).isoformat(),
            },
        }

    def search_traces(
        self,
        endpoint: Optional[str] = None,
        status: Optional[str] = None,
        min_duration_ms: Optional[float] = None,
        max_duration_ms: Optional[float] = None,
        limit: int = 50,
    ) -> List[dict]:
        """搜索Trace"""
        all_traces = self.storage.get_all_traces()
        
        filtered = []
        for trace in all_traces:
            # 按端点过滤
            if endpoint:
                has_matching_span = False
                for span in trace.get("spans", []):
                    span_endpoint = span.get("attributes", {}).get("http.path", "")
                    if endpoint in span_endpoint:
                        has_matching_span = True
                        break
                if not has_matching_span:
                    continue
            
            # 按状态过滤
            if status and trace.get("status") != status:
                continue
            
            # 按最小持续时间过滤
            duration = trace.get("duration_ms", 0)
            if min_duration_ms and duration < min_duration_ms:
                continue
            
            # 按最大持续时间过滤
            if max_duration_ms and duration > max_duration_ms:
                continue
            
            filtered.append(trace)
        
        # 按时间排序
        filtered.sort(key=lambda t: t.get("start_time", 0), reverse=True)
        
        return filtered[:limit]


# 全局服务实例
_trace_storage_service: Optional[TraceStorageService] = None


def get_trace_storage_service() -> TraceStorageService:
    """获取链路存储服务实例"""
    global _trace_storage_service
    if _trace_storage_service is None:
        _trace_storage_service = TraceStorageService()
    return _trace_storage_service


def init_trace_storage_service(use_redis: bool = False, redis_client=None):
    """初始化链路存储服务"""
    global _trace_storage_service
    _trace_storage_service = TraceStorageService()
    
    if use_redis and redis_client:
        _trace_storage_service.set_storage(RedisTraceStorage(redis_client))
        logger.info("链路存储服务已切换到Redis存储")
    else:
        logger.info("链路存储服务使用内存存储")
    
    return _trace_storage_service
