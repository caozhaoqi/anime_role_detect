#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""轻量 metrics 层：在现有 JSONL 日志之上，补一层可量化的运行指标。

- 仅用标准库（threading/json/os），主 ``.venv`` 与 ``t2i-mac`` venv 均可 import。
- 三类原语：``record_latency``(耗时分布) / ``set_gauge``(瞬时值) / ``inc_counter``(计数)。
- 每条指标同时追加写入 ``logs/metrics.jsonl``（与业务日志分离），并保留内存窗口供
  ``summary()`` 即时聚合（P50/P95/max/avg）。
- 典型用途：量化「MPS 内存尖峰」「生成耗时」「idle 卸载/重启次数」，回应资源治理需求。
"""
from __future__ import annotations

import json
import os
import threading
import time
from collections import defaultdict, deque

_LOG_DIR = "logs"
_LOG_PATH = os.path.join(_LOG_DIR, "metrics.jsonl")
_WINDOW = 200  # 每个 latency 指标保留的最近样本数


class MetricsCollector:
    def __init__(self, window: int = _WINDOW, log_path: str = _LOG_PATH) -> None:
        self._lock = threading.Lock()
        self._latencies: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window))
        self._gauges: Dict[str, float] = {}
        self._counters: Dict[str, int] = defaultdict(int)
        self._log_path = log_path
        try:
            os.makedirs(os.path.dirname(self._log_path), exist_ok=True)
        except OSError:
            pass

    # ---- 原语 ----
    def record_latency(self, name: str, seconds: float) -> None:
        with self._lock:
            self._latencies[name].append(float(seconds))
        self._emit(name, "latency", seconds)

    def set_gauge(self, name: str, value: float) -> None:
        with self._lock:
            self._gauges[name] = float(value)
        self._emit(name, "gauge", value)

    def inc_counter(self, name: str, by: int = 1) -> None:
        with self._lock:
            self._counters[name] += by
        self._emit(name, "counter", self._counters[name])

    # ---- 聚合 ----
    def summary(self) -> dict:
        with self._lock:
            out: Dict[str, object] = {}
            for name, vals in self._latencies.items():
                if not vals:
                    continue
                s = sorted(vals)
                n = len(s)
                p50 = s[int(n * 0.5)]
                p95 = s[min(n - 1, int(n * 0.95))]
                out[name] = {
                    "count": n,
                    "p50": round(p50, 3),
                    "p95": round(p95, 3),
                    "max": round(s[-1], 3),
                    "avg": round(sum(s) / n, 3),
                }
            out["_gauges"] = dict(self._gauges)
            out["_counters"] = dict(self._counters)
        return out

    # ---- 落盘（与业务 JSONL 分离，便于独立采集）----
    def _emit(self, name: str, kind: str, value) -> None:
        try:
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(
                    json.dumps({"t": round(time.time(), 3), "name": name, "kind": kind, "value": value})
                    + "\n"
                )
        except OSError:
            pass


# 进程级单例
metrics = MetricsCollector()
