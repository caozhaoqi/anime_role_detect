from .global_logger import (
    get_logger,
    log_system,
    log_inference,
    log_training,
    log_error,
    get_unified_log,
    tail_unified_log,
    get_log_info,
    GlobalLogger
)

from .unified_logger import (
    setup_logger,
    get_unified_logger,
    get_all_logs,
    tail_log,
    list_services,
    LOG_DIR,
    LOG_FILES
)

__all__ = [
    'get_logger',
    'log_system',
    'log_inference',
    'log_training',
    'log_error',
    'get_unified_log',
    'tail_unified_log',
    'get_log_info',
    'GlobalLogger',
    'setup_logger',
    'get_unified_logger',
    'get_all_logs',
    'tail_log',
    'list_services',
    'LOG_DIR',
    'LOG_FILES'
]
