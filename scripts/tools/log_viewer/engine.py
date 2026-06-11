# -*- coding: utf-8 -*-
"""
日志引擎 - 解析、过滤、统计
"""

import json
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

# ============ 配置 ============
LOG_DIR = Path("/Users/caozhaoqi/PycharmProjects/anime_role_detect/logs")
# ==============================


# ───────── 服务名提取 ─────────

def extract_service(entry):
    """从日志条目提取服务名（按包路径第二段归类）"""
    record = entry.get("record", {})
    name = record.get("name", "") or ""
    if not name:
        return record.get("module", "other")
    parts = name.split(".")
    if len(parts) >= 3:
        return parts[1]  # src.core.xxx -> core
    return parts[0]  # app -> app


# ───────── 日志解析 ─────────

def parse_logs():
    """解析所有JSONL日志文件"""
    logs = []
    if not LOG_DIR.exists():
        return logs

    for f in sorted(LOG_DIR.glob("*.jsonl")):
        try:
            with open(f, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        entry["_service"] = extract_service(entry)
                        logs.append(entry)
                    except json.JSONDecodeError:
                        logs.append({
                            "text": line,
                            "_service": "raw",
                            "record": {
                                "level": {"name": "RAW", "no": 0},
                                "time": {"repr": str(datetime.fromtimestamp(f.stat().st_mtime))},
                                "message": line[:200],
                            }
                        })
        except Exception:
            continue
    return logs


# ───────── 过滤搜索 ─────────

def filter_logs(logs, keyword="", level="", service="",
                date_from="", date_to="", sort_order="desc",
                offset=0, limit=100):
    """过滤和搜索日志（模糊匹配）"""
    filtered = []

    def _get_time_key(entry):
        return entry.get("record", {}).get("time", {}).get("repr", "")

    def _fuzzy_match(text, kw_parts):
        """模糊匹配：拆分关键词为多个词，任一匹配即通过"""
        text_lower = text.lower()
        return any(part in text_lower for part in kw_parts)

    kw_parts = [w.strip() for w in keyword.lower().split() if w.strip()] if keyword else []

    for entry in logs:
        record = entry.get("record", {})
        text = entry.get("text", "")
        msg = record.get("message", "")
        level_name = record.get("level", {}).get("name", "")
        svc = entry.get("_service", "")
        time_str = record.get("time", {}).get("repr", "")

        # 模糊关键词搜索（匹配 text / message / 文件名 / 函数名）
        if kw_parts:
            search_text = " ".join([
                text, msg,
                record.get("file", {}).get("name", ""),
                record.get("function", ""),
            ])
            if not _fuzzy_match(search_text, kw_parts):
                continue

        # 级别过滤
        if level and level.upper() != level_name.upper():
            continue

        # 服务名过滤
        if service and service != svc:
            continue

        # 时间范围过滤
        if date_from and time_str < date_from:
            continue
        if date_to and time_str > date_to:
            continue

        filtered.append(entry)

    total = len(filtered)

    # 排序
    filtered.sort(key=_get_time_key, reverse=(sort_order != "asc"))

    # 分页
    paginated = filtered[offset:offset + limit] if limit > 0 else filtered

    return paginated, total


# ───────── 统计 ─────────

def get_stats(logs):
    """获取日志统计信息"""
    stats = {
        "total": len(logs),
        "by_level": {},
        "by_service": {},
        "by_hour": {},
        "errors": [],
        "time_range": {"start": None, "end": None},
    }

    for entry in logs:
        record = entry.get("record", {})
        level = record.get("level", {}).get("name", "UNKNOWN")
        svc = entry.get("_service", "other")
        time_str = record.get("time", {}).get("repr", "")
        msg = record.get("message", "")
        exc = record.get("exception")

        stats["by_level"][level] = stats["by_level"].get(level, 0) + 1
        stats["by_service"][svc] = stats["by_service"].get(svc, 0) + 1

        if time_str:
            try:
                hour = time_str[11:13] if len(time_str) > 13 else "00"
                stats["by_hour"][hour] = stats["by_hour"].get(hour, 0) + 1
            except Exception:
                pass

        if time_str:
            if stats["time_range"]["start"] is None or time_str < stats["time_range"]["start"]:
                stats["time_range"]["start"] = time_str
            if stats["time_range"]["end"] is None or time_str > stats["time_range"]["end"]:
                stats["time_range"]["end"] = time_str

        if level in ("ERROR", "CRITICAL") or exc:
            stats["errors"].append({
                "time": time_str,
                "service": svc,
                "message": msg[:200],
                "level": level,
            })

    stats["by_service"] = dict(sorted(stats["by_service"].items(), key=lambda x: -x[1]))
    stats["errors"] = stats["errors"][-50:]
    return stats


# ───────── 缓存 ─────────

_log_cache = []
_cache_time = 0
_cache_lock = threading.Lock()


def get_cached_logs(force_refresh=False):
    """获取缓存的日志，每30秒刷新一次"""
    global _log_cache, _cache_time
    now = time.time()
    with _cache_lock:
        if force_refresh or not _log_cache or now - _cache_time > 30:
            _log_cache = parse_logs()
            _cache_time = now
        return _log_cache


# ───────── 实时Tail ─────────

def tail_logs(lines=50):
    """tail日志文件的最新行"""
    if not LOG_DIR.exists():
        return []

    results = []
    for f in sorted(LOG_DIR.glob("*.jsonl")):
        try:
            cmd = ["tail", "-n", str(lines), str(f)]
            output = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode("utf-8")
            for line in output.strip().split("\n"):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    entry["_service"] = extract_service(entry)
                    results.append(entry)
                except json.JSONDecodeError:
                    results.append({
                        "text": line,
                        "_service": "raw",
                        "record": {"level": {"name": "RAW", "no": 0}, "message": line[:300]},
                    })
        except Exception:
            continue

    return results[-lines:]