import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_DIR = Path("./anime_role_detect_logs")
LOG_DIR.mkdir(exist_ok=True)

LOG_FILES = {
    "api_gateway": LOG_DIR / "api_gateway.log",
    "core_api": LOG_DIR / "core_api.log",
    "model_service": LOG_DIR / "model_service.log",
    "unified": LOG_DIR / "unified.log",
}


def setup_logger(name: str, log_file: str = None, level=logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        file_handler = RotatingFileHandler(
            log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_unified_logger(name: str) -> logging.Logger:
    logger = setup_logger(name, str(LOG_FILES["unified"]))
    return logger


def get_logger(service_name: str) -> logging.Logger:
    if service_name not in LOG_FILES:
        service_name = "unified"
    return setup_logger(service_name, str(LOG_FILES[service_name]))


def get_all_logs(tail: int = 100) -> str:
    unified_log = LOG_FILES["unified"]
    if unified_log.exists():
        lines = unified_log.read_text(encoding="utf-8").split("\n")
        return "\n".join(lines[-tail:])
    return "暂无日志"


def tail_log(service: str = "unified", lines: int = 50) -> str:
    log_file = LOG_FILES.get(service, LOG_FILES["unified"])
    if log_file.exists():
        all_lines = log_file.read_text(encoding="utf-8").split("\n")
        return "\n".join(all_lines[-lines:])
    return f"日志文件不存在: {log_file}"


def list_services() -> dict:
    return {service: str(path) for service, path in LOG_FILES.items()}


if __name__ == "__main__":
    print("=== 动漫角色识别系统 - 统一日志 ===")
    print(f"\n日志目录: {LOG_DIR}")
    print("\n服务日志文件:")
    for service, path in LOG_FILES.items():
        exists = "✓" if path.exists() else "✗"
        print(f"  [{exists}] {service}: {path}")

    print(f"\n统一日志内容 (最后50行):")
    print("-" * 60)
    print(get_all_logs(50))
