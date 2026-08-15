#!/usr/bin/env python3
"""
统一日志配置（OPTIMIZATION_PLAN 2.5 结构化日志）

将 loguru 的输出统一为 JSON 行，便于 ES / Kibana 检索。
字段：timestamp / level / service / trace_id / message / module / line / function

用法：
    from src.core.logging_setup import setup_logging
    setup_logging("api-service")              # 在入口模块加载处调用一次
    setup_logging("model-service", "DEBUG")   # 可指定级别

注入 trace_id（结构化字段，便于链路追踪）：
    logger.bind(trace_id="abc-123").info("处理请求")
    # 或借助现有 enhanced_logger：
    #   get_enhanced_logger("x").bind(trace_id="abc-123").info("...")
"""
import sys
from pathlib import Path

from loguru import logger

# 幂等标志：每个进程（对应一个服务）只配置一次，重复调用不会重复添加 handler
_initialized = False


def setup_logging(service_name: str, level: str = "INFO") -> None:
    """配置 loguru 输出结构化 JSON 到 stdout。

    Args:
        service_name: 服务名，如 "api-service" / "model-service" / "api-gateway"
        level: 日志级别（默认 "INFO"）
    """
    global _initialized
    if _initialized:
        return

    # 清空任何已有 handler（含默认 handler 及历史配置），保证"统一出口格式"
    logger.remove()

    def _json_format(record: dict) -> str:
        """返回 loguru 的动态格式模板。

        注意：本环境 loguru（0.7.3）对 callable format 的处理是——
        先调用 format(record) 得到「模板字符串」，再对模板做 .format_map(record)。
        因此这里返回的是带 {字段} 占位符的模板，而非最终 JSON 文本。

        为了对任意 message 内容（含引号/花括号/换行）都能产出合法 JSON，
        将 trace_id / message / service 以「同时做了 JSON 转义与模板转义」的字面量拼入模板，
        避免被 .format_map 二次解析；仅保留安全字段（time/level/module/line/function）作占位符。
        """
        extra = dict(record.get("extra") or {})
        trace_id = extra.get("trace_id", "")

        # 同时做 JSON 转义与 loguru 模板转义（{ } 必须转成 {{ }} 否则被 format_map 当字段）
        def _escape(s: str) -> str:
            return (
                s.replace("\\", "\\\\")
                .replace('"', '\\"')
                .replace("{", "{{")
                .replace("}", "}}")
                .replace("\n", "\\n")
                .replace("\r", "\\r")
                .replace("\t", "\\t")
            )

        # 字面量部分直接拼接，不参与 .format_map；结尾补 \n（loguru 不在 sink 层加换行）
        return (
            '{{"timestamp": "{time:%Y-%m-%dT%H:%M:%S.%f%z}", '
            '"level": "{level.name}", '
            '"service": "' + _escape(service_name) + '", '
            '"trace_id": "' + _escape(trace_id) + '", '
            '"message": "' + _escape(record["message"]) + '", '
            '"module": "{module}", '
            '"line": {line}, '
            '"function": "{function}"}}'
            "\n"
        )

    logger.add(
        sys.stdout,
        level=level,
        format=_json_format,
        serialize=False,
    )

    # 同时写入中央统一日志，使业务应用日志进入 logs/anime_role_detect_unified.log
    # 与 logs/anime_role_detect_structured_{date}.jsonl，供 /logs 查看器与可观测面板消费。
    # 此前 setup_logging 只写 stdout，导致中央日志文件里只有监控脚本的日志（"全是监控"）。
    # 这里用与 enhanced_logger 一致的固定文件名，确保日志查看 API 能读到应用日志。
    _log_dir = Path("logs")
    _log_dir.mkdir(parents=True, exist_ok=True)

    _plain_format = (
        "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
        "{name}:{function}:{line} | {message}"
    )

    # 固定文件名（与 enhanced_logger 一致，日志查看 API 默认读 anime_role_detect_unified.log），
    # 不按 service_name 拼接，否则会变成 api-gateway_unified.log 而脱离统一视图。
    logger.add(
        str(_log_dir / "anime_role_detect_unified.log"),
        rotation="100 MB",
        retention="3 days",
        compression="zip",
        level="INFO",
        format=_plain_format,
        colorize=False,
        enqueue=True,
    )

    logger.add(
        str(_log_dir / "anime_role_detect_structured_{time:YYYY-MM-DD}.jsonl"),
        rotation="00:00",
        retention="7 days",
        compression="zip",
        level="INFO",
        serialize=True,
        enqueue=True,
    )

    _initialized = True
