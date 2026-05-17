import logging
import sys
from pathlib import Path
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler


class DailyLogHandler(TimedRotatingFileHandler):
    """按天轮转的日志处理器"""

    def __init__(self, log_dir, base_filename="app.log"):
        log_dir = Path(log_dir)
        log_dir.mkdir(exist_ok=True)

        log_file = log_dir / base_filename
        super().__init__(
            str(log_file),
            when='midnight',
            interval=1,
            backupCount=30,
            encoding='utf-8'
        )

        self.suffix = "%Y%m%d.log"
        self.extMatch = r"^\d{8}\.log$"


class ColoredFormatter(logging.Formatter):
    """带颜色的日志格式化器"""

    COLORS = {
        'DEBUG': '\033[36m',
        'INFO': '\033[32m',
        'WARNING': '\033[33m',
        'ERROR': '\033[31m',
        'CRITICAL': '\033[35m',
    }
    RESET = '\033[0m'

    def format(self, record):
        if hasattr(sys.stderr, 'isatty') and sys.stderr.isatty():
            record.levelname = f"{self.COLORS.get(record.levelname, '')}{record.levelname}{self.RESET}"
        return super().format(record)


_loggers = {}


def setup_logger(
    name="AnnotationTool",
    log_dir=None,
    level=logging.INFO,
    console=True
):
    """设置日志系统

    Args:
        name: 日志记录器名称
        log_dir: 日志目录，默认在项目下创建logs目录
        level: 日志级别，默认INFO
        console: 是否输出到控制台，默认True
    """
    if log_dir is None:
        log_dir = Path(__file__).parent.parent / "logs"

    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    file_handler = DailyLogHandler(log_dir, f"{name.lower()}.log")
    file_handler.setLevel(level)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_formatter = ColoredFormatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    _loggers[name] = logger
    return logger


def get_logger(name="AnnotationTool"):
    """获取已设置的日志记录器"""
    if name not in _loggers:
        return setup_logger(name)
    return _loggers[name]


def log_operation(msg, level='info'):
    """快速记录操作的辅助函数"""
    logger = get_logger("Operation")
    getattr(logger, level.lower())(msg)


def log_system(msg, level='info'):
    """快速记录系统事件的辅助函数"""
    logger = get_logger("System")
    getattr(logger, level.lower())(msg)
