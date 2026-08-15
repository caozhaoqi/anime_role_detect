#!/usr/bin/env python3
"""
全局日志系统模块
统一管理系统日志，包括系统运行状态、模型推理结果、模型训练结果和错误日志
支持标准格式与 JSON 结构化日志输出（适配 ELK/Loki）
"""
import os
import sys
import json
from datetime import datetime
from pathlib import Path
from loguru import logger


class GlobalLogger:
    """
    全局日志系统类
    统一管理所有类型的日志
    """

    def __init__(self, log_dir: str = "logs"):
        """
        初始化全局日志系统

        Args:
            log_dir: 日志根目录
        """
        self.log_dir = Path(log_dir)
        self._setup_directories()
        self._configure_logger()

    def _setup_directories(self):
        """
        创建日志目录结构
        """
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.system_log_dir = self.log_dir / "system"
        self.inference_log_dir = self.log_dir / "inference"
        self.training_log_dir = self.log_dir / "training"
        self.error_log_dir = self.log_dir / "error"

        for dir_path in [
            self.system_log_dir,
            self.inference_log_dir,
            self.training_log_dir,
            self.error_log_dir,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def _configure_logger(self):
        """
        配置loguru日志系统
        """
        logger.remove()

        log_format = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | <level>{message}</level>"
        # 纯文本格式（无颜色标签）：用于文件/标准输出 sink。
        # 关键修复：loguru 默认 colorize=True 会对已渲染的日志内容二次解析，
        # 当 message 含 <module>（模块级调用）或 {...}（字典参数，如 {'http_request': 10.0}）时
        # 会抛 ValueError/KeyError 并静默丢弃日志。改用纯文本 + colorize=False 彻底规避。
        log_format_plain = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}"

        # JSON 格式日志文件（供 ELK Filebeat / Loki 采集）
        # 按日期轮转，每天生成新文件，保留14天
        json_log_file = str(self.log_dir / "structured_{time:YYYY-MM-DD}.jsonl")
        logger.add(
            json_log_file,
            rotation="00:00",
            retention="14 days",
            compression="zip",
            level="INFO",
            serialize=True,
            enqueue=True,
        )

        unified_log_file = str(self.log_dir / "unified.log")
        logger.add(
            unified_log_file,
            rotation="100 MB",
            retention="7 days",
            compression="zip",
            # INFO：避免 DEBUG 级资源采样刷屏统一日志（log_viewer.py 仍读 unified.log）
            level="INFO",
            format=log_format_plain, colorize=False,
            enqueue=True,
        )

        system_log_file = str(self.system_log_dir / "system_{time:YYYY-MM-DD}.log")
        logger.add(
            system_log_file,
            rotation="100 MB",
            retention="7 days",
            compression="zip",
            level="INFO",
            format=log_format_plain, colorize=False,
        )

        # 推理日志配置
        inference_log_file = str(self.inference_log_dir / "inference_{time:YYYY-MM-DD}.log")
        logger.add(
            inference_log_file,
            rotation="100 MB",
            retention="14 days",
            compression="zip",
            level="INFO",
            format=log_format_plain, colorize=False,
        )

        # 训练日志配置
        training_log_file = str(self.training_log_dir / "training_{time:YYYY-MM-DD}.log")
        logger.add(
            training_log_file,
            rotation="200 MB",
            retention="30 days",
            compression="zip",
            level="INFO",
            format=log_format_plain, colorize=False,
        )

        # 错误日志配置
        error_log_file = str(self.error_log_dir / "error_{time:YYYY-MM-DD}.log")
        logger.add(
            error_log_file,
            rotation="50 MB",
            retention="30 days",
            compression="zip",
            level="ERROR",
            format=log_format_plain, colorize=False,
        )

        # 控制台输出配置
        logger.add(sys.stdout, level="INFO", format=log_format_plain, colorize=False)

        # 抑制第三方库的 INFO/DEBUG 噪音（标准 logging 体系）。
        # transformers / diffusers 等加载模型时会刷大量 INFO；抬高到 WARNING 可显著减少噪音，
        # 且不影响项目自身的业务/错误日志（业务走 loguru，与标准 logging 互相独立）。
        import logging as _std_logging

        for _lib in (
            "transformers", "diffusers", "modelscope", "accelerate",
            "matplotlib", "PIL", "urllib3", "httpx",
            "uvicorn", "uvicorn.access", "uvicorn.error",
        ):
            _std_logging.getLogger(_lib).setLevel(_std_logging.WARNING)

    def get_logger(self, name: str = "global"):
        """
        获取日志记录器

        Args:
            name: 记录器名称

        Returns:
            loguru.Logger: 日志记录器
        """
        return logger.bind(name=name)

    def log_system(self, message: str, level: str = "info", **kwargs):
        """
        记录系统日志

        Args:
            message: 日志消息
            level: 日志级别
            **kwargs: 额外参数
        """
        log_method = getattr(logger.bind(system=True), level.lower())
        log_method(message, **kwargs)

    def log_inference(self, message: str, level: str = "info", **kwargs):
        """
        记录推理日志

        Args:
            message: 日志消息
            level: 日志级别
            **kwargs: 额外参数
        """
        log_method = getattr(logger.bind(inference=True), level.lower())
        log_method(message, **kwargs)

    def log_training(self, message: str, level: str = "info", **kwargs):
        """
        记录训练日志

        Args:
            message: 日志消息
            level: 日志级别
            **kwargs: 额外参数
        """
        log_method = getattr(logger.bind(training=True), level.lower())
        log_method(message, **kwargs)

    def log_error(self, message: str, level: str = "error", **kwargs):
        """
        记录错误日志

        Args:
            message: 日志消息
            level: 日志级别
            **kwargs: 额外参数
        """
        log_method = getattr(logger.bind(error=True), level.lower())
        log_method(message, **kwargs)


# 创建全局日志实例
global_logger = GlobalLogger()


# 便捷函数
def get_logger(name: str = "global"):
    """
    获取日志记录器

    Args:
        name: 记录器名称

    Returns:
        loguru.Logger: 日志记录器
    """
    return global_logger.get_logger(name)


def log_system(message: str, level: str = "info", **kwargs):
    """
    记录系统日志

    Args:
        message: 日志消息
        level: 日志级别
        **kwargs: 额外参数
    """
    global_logger.log_system(message, level, **kwargs)


def log_inference(message: str, level: str = "info", **kwargs):
    """
    记录推理日志

    Args:
        message: 日志消息
        level: 日志级别
        **kwargs: 额外参数
    """
    global_logger.log_inference(message, level, **kwargs)


def log_training(message: str, level: str = "info", **kwargs):
    """
    记录训练日志

    Args:
        message: 日志消息
        level: 日志级别
        **kwargs: 额外参数
    """
    global_logger.log_training(message, level, **kwargs)


def log_error(message: str, level: str = "error", **kwargs):
    """
    记录错误日志

    Args:
        message: 日志消息
        level: 日志级别
        **kwargs: 额外参数
    """
    global_logger.log_error(message, level, **kwargs)


def get_unified_log(tail: int = 100) -> str:
    """
    获取统一日志的最后N行

    Args:
        tail: 返回的最后行数

    Returns:
        str: 日志内容
    """
    unified_log_file = global_logger.log_dir / "unified.log"
    if unified_log_file.exists():
        try:
            lines = unified_log_file.read_text(encoding="utf-8").split("\n")
            return "\n".join(lines[-tail:])
        except Exception as e:
            return f"读取日志失败: {e}"
    return "暂无日志"


def tail_unified_log(lines: int = 50) -> str:
    """
    获取统一日志的最后N行（简洁别名）

    Args:
        lines: 返回的最后行数

    Returns:
        str: 日志内容
    """
    return get_unified_log(tail=lines)


def get_log_info() -> dict:
    """
    获取日志系统信息

    Returns:
        dict: 日志目录和文件信息
    """
    log_dir = global_logger.log_dir
    info = {"log_dir": str(log_dir), "unified_log": str(log_dir / "unified.log"), "files": {}}
    for subdir in ["system", "inference", "training", "error"]:
        path = log_dir / subdir
        if path.exists():
            info["files"][subdir] = [str(f) for f in path.glob("*.log*")]
    return info
