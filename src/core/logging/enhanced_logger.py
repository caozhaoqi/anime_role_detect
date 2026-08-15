import os
import sys
import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union
from loguru import logger
from loguru._logger import Logger

try:
    from opentelemetry import trace
    from opentelemetry.trace import Span, Tracer
    HAS_OPENTELEMETRY = True
except ImportError:
    HAS_OPENTELEMETRY = False
    # Fallback: define dummy types so type annotations don't crash at import time
    # when opentelemetry is not installed
    Span = Any
    Tracer = Any

from .request_context import RequestContext


class EnhancedLogger:
    COMPONENTS = {
        'api': 'api',
        'db': 'sql',
        'redis': 'redis',
        'model': 'model',
        'service': 'service',
    }

    def __init__(self, log_dir: str = "logs", service_name: str = "anime_role_detect"):
        self.log_dir = Path(log_dir)
        self.service_name = service_name
        self._setup_directories()
        self._configure_logger()
        self._tracer = None
        if HAS_OPENTELEMETRY:
            self._tracer = trace.get_tracer(service_name)

    def _setup_directories(self):
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.system_log_dir = self.log_dir / "system"
        self.inference_log_dir = self.log_dir / "inference"
        self.training_log_dir = self.log_dir / "training"
        self.error_log_dir = self.log_dir / "error"
        self.access_log_dir = self.log_dir / "access"
        self.operation_log_dir = self.log_dir / "operation"

        for dir_path in [
            self.system_log_dir,
            self.inference_log_dir,
            self.training_log_dir,
            self.error_log_dir,
            self.access_log_dir,
            self.operation_log_dir,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def _format_log_record(self, record):
        extra = record.get("extra", {})
        return "{time:YYYY-MM-DD HH:mm:ss.SSS} | {trace_id} | {request_id} | {level: <8} | {component} | {name}:{function}:{line} | {message}".format(
            time=record["time"],
            trace_id=extra.get("trace_id", "-"),
            request_id=extra.get("request_id", "-"),
            level=record["level"].name,
            component=extra.get("component", "service"),
            name=record["name"],
            function=record["function"],
            line=record["line"],
            message=record["message"],
        )

    def _configure_logger(self):
        logger.remove()

        log_format = self._format_log_record

        json_log_file = str(self.log_dir / f"{self.service_name}_structured_{{time:YYYY-MM-DD}}.jsonl")
        logger.add(
            json_log_file,
            rotation="00:00",
            retention="7 days",
            compression="zip",
            level="INFO",
            serialize=True,
            enqueue=True,
        )

        unified_log_file = str(self.log_dir / f"{self.service_name}_unified.log")
        logger.add(
            unified_log_file,
            rotation="100 MB",
            retention="3 days",
            compression="zip",
            level="DEBUG",
            format=log_format,
            colorize=False,
            enqueue=True,
        )

        system_log_file = str(self.system_log_dir / f"system_{{time:YYYY-MM-DD}}.log")
        logger.add(
            system_log_file,
            rotation="100 MB",
            retention="3 days",
            compression="zip",
            level="INFO",
            format=log_format,
            colorize=False,
            filter=lambda record: record["extra"].get("log_type") == "system" or "log_type" not in record["extra"],
        )

        inference_log_file = str(self.inference_log_dir / f"inference_{{time:YYYY-MM-DD}}.log")
        logger.add(
            inference_log_file,
            rotation="100 MB",
            retention="7 days",
            compression="zip",
            level="INFO",
            format=log_format,
            colorize=False,
            filter=lambda record: record["extra"].get("log_type") == "inference",
        )

        training_log_file = str(self.training_log_dir / f"training_{{time:YYYY-MM-DD}}.log")
        logger.add(
            training_log_file,
            rotation="200 MB",
            retention="10 days",
            compression="zip",
            level="INFO",
            format=log_format,
            colorize=False,
            filter=lambda record: record["extra"].get("log_type") == "training",
        )

        error_log_file = str(self.error_log_dir / f"error_{{time:YYYY-MM-DD}}.log")
        logger.add(
            error_log_file,
            rotation="50 MB",
            retention="10 days",
            compression="zip",
            level="ERROR",
            format=log_format,
            colorize=False,
            filter=lambda record: record["extra"].get("log_type") == "error" or record["level"].name == "ERROR",
        )

        access_log_file = str(self.access_log_dir / f"access_{{time:YYYY-MM-DD}}.log")
        logger.add(
            access_log_file,
            rotation="100 MB",
            retention="7 days",
            compression="zip",
            level="INFO",
            format=log_format,
            colorize=False,
            filter=lambda record: record["extra"].get("log_type") == "access",
        )

        operation_log_file = str(self.operation_log_dir / f"operation_{{time:YYYY-MM-DD}}.log")
        logger.add(
            operation_log_file,
            rotation="100 MB",
            retention="10 days",
            compression="zip",
            level="INFO",
            format=log_format,
            colorize=False,
            filter=lambda record: record["extra"].get("log_type") == "operation",
        )

        logger.add(sys.stdout, level="INFO", format=log_format, colorize=False)

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

    def _get_base_extra(self, component: str = "service", log_type: str = "system") -> Dict[str, Any]:
        return {
            'trace_id': RequestContext.get_trace_id(),
            'request_id': RequestContext.get_request_id(),
            'span_id': RequestContext.get_span_id() or "",
            'user_id': RequestContext.get_user_id() or "",
            'component': component,
            'log_type': log_type,
            'service': self.service_name,
        }

    def _get_logger(self, **extra) -> Logger:
        base_extra = self._get_base_extra()
        base_extra.update(extra)
        return logger.bind(**base_extra)

    def info(self, message: str, **kwargs):
        self._get_logger(**kwargs).info(message)

    def debug(self, message: str, **kwargs):
        self._get_logger(**kwargs).debug(message)

    def warning(self, message: str, **kwargs):
        self._get_logger(**kwargs).warning(message)

    def error(self, message: str, **kwargs):
        self._get_logger(**kwargs).error(message)

    def critical(self, message: str, **kwargs):
        self._get_logger(**kwargs).critical(message)

    def log_system(self, message: str, level: str = "info", **kwargs):
        extra = self._get_base_extra(component="system", log_type="system")
        extra.update(kwargs)
        log_method = getattr(logger.bind(**extra), level.lower())
        log_method(message)

    def log_inference(self, message: str, level: str = "info", **kwargs):
        extra = self._get_base_extra(component="model", log_type="inference")
        extra.update(kwargs)
        log_method = getattr(logger.bind(**extra), level.lower())
        log_method(message)

    def log_training(self, message: str, level: str = "info", **kwargs):
        extra = self._get_base_extra(component="model", log_type="training")
        extra.update(kwargs)
        log_method = getattr(logger.bind(**extra), level.lower())
        log_method(message)

    def log_error(self, message: str, error: Exception = None, level: str = "error", **kwargs):
        extra = self._get_base_extra(component="service", log_type="error")
        extra.update(kwargs)

        if error is not None:
            extra.update({
                'error_type': error.__class__.__name__,
                'error_message': str(error),
                'error_stack': ''.join(traceback.format_exception(type(error), error, error.__traceback__)),
            })

        log_method = getattr(logger.bind(**extra), level.lower())
        log_method(message)

    def log_access(self, method: str, url: str, status_code: int, duration: float = 0, **kwargs):
        extra = self._get_base_extra(component="api", log_type="access")
        extra.update({
            'http_method': method,
            'http_url': url,
            'http_status_code': status_code,
            'duration_ms': duration,
        })
        extra.update(kwargs)
        message = f"{method} {url} {status_code} ({duration:.2f}ms)"
        logger.bind(**extra).info(message)

    def log_operation(self, operation: str, operator: str = "", target: str = "", result: str = "success", **kwargs):
        extra = self._get_base_extra(component="service", log_type="operation")
        extra.update({
            'operation': operation,
            'operator': operator or RequestContext.get_user_id() or "",
            'target': target,
            'result': result,
        })
        extra.update(kwargs)
        message = f"[{operator}] {operation} {target} -> {result}"
        logger.bind(**extra).info(message)

    def log_db_operation(self, operation: str, table: str, record_id: str = "", **kwargs):
        extra = self._get_base_extra(component="db", log_type="system")
        extra.update({
            'db_operation': operation,
            'db_table': table,
            'db_record_id': record_id,
        })
        extra.update(kwargs)
        message = f"DB {operation} {table} {record_id}"
        logger.bind(**extra).info(message)

    def log_redis_operation(self, operation: str, key: str = "", **kwargs):
        extra = self._get_base_extra(component="redis", log_type="system")
        extra.update({
            'redis_operation': operation,
            'redis_key': key,
        })
        extra.update(kwargs)
        message = f"Redis {operation} {key}"
        logger.bind(**extra).info(message)

    def log_with_span(self, message: str, span: Optional[Span] = None, level: str = "info", **kwargs):
        if span is not None:
            span_id = format(span.get_span_context().span_id, '016x')
            trace_id = format(span.get_span_context().trace_id, '032x')
            RequestContext.set_trace_context(trace_id, span_id)
            span.add_event(message, attributes=kwargs)

        extra = self._get_base_extra(**kwargs)
        log_method = getattr(logger.bind(**extra), level.lower())
        log_method(message)

    def start_span(self, name: str, component: str = "service", **kwargs) -> Optional[Span]:
        if not HAS_OPENTELEMETRY or self._tracer is None:
            return None

        span = self._tracer.start_span(name)
        span.set_attribute('component', component)
        span.set_attribute('service', self.service_name)

        for key, value in kwargs.items():
            span.set_attribute(key, value)

        span_id = format(span.get_span_context().span_id, '016x')
        trace_id = format(span.get_span_context().trace_id, '032x')
        RequestContext.set_trace_context(trace_id, span_id)

        return span

    def end_span(self, span: Span, error: Exception = None, status_code: int = 200):
        if span is None or not HAS_OPENTELEMETRY:
            return

        if error is not None:
            span.set_status(trace.Status(trace.StatusCode.ERROR, str(error)))
            span.set_attribute('error.type', error.__class__.__name__)
            span.set_attribute('error.message', str(error))
            span.set_attribute('error.stack', ''.join(traceback.format_exception(type(error), error, error.__traceback__)))

        span.set_attribute('http.status_code', status_code)
        span.end()

    def get_logger(self, name: str = "global") -> Logger:
        extra = self._get_base_extra()
        return logger.bind(name=name, **extra)

    def get_log_info(self) -> dict:
        log_dir = self.log_dir
        info = {"log_dir": str(log_dir), "service": self.service_name, "files": {}}
        for subdir in ["system", "inference", "training", "error", "access", "operation"]:
            path = log_dir / subdir
            if path.exists():
                info["files"][subdir] = [str(f) for f in path.glob("*.log*")]
        return info


enhanced_logger = EnhancedLogger()


def get_enhanced_logger(name: str = "global") -> Logger:
    return enhanced_logger.get_logger(name)


def log_with_context(message: str, level: str = "info", **kwargs):
    log_method = getattr(enhanced_logger, level.lower())
    log_method(message, **kwargs)


def log_access(method: str, url: str, status_code: int, duration: float = 0, **kwargs):
    enhanced_logger.log_access(method, url, status_code, duration, **kwargs)


def log_operation(operation: str, operator: str = "", target: str = "", result: str = "success", **kwargs):
    enhanced_logger.log_operation(operation, operator, target, result, **kwargs)


def log_db(operation: str, table: str, record_id: str = "", **kwargs):
    enhanced_logger.log_db_operation(operation, table, record_id, **kwargs)


def log_redis(operation: str, key: str = "", **kwargs):
    enhanced_logger.log_redis_operation(operation, key, **kwargs)


def log_error_with_stack(message: str, error: Exception = None, **kwargs):
    enhanced_logger.log_error(message, error, **kwargs)


def start_trace_span(name: str, component: str = "service", **kwargs):
    return enhanced_logger.start_span(name, component, **kwargs)


def end_trace_span(span, error: Exception = None, status_code: int = 200):
    enhanced_logger.end_span(span, error, status_code)