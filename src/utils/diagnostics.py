import torch
import psutil
import platform
import os
import gc
from typing import Dict, Any
from loguru import logger


class CrossPlatformDiagnostics:

    @staticmethod
    def get_device_info() -> str:
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @classmethod
    def dump_memory_snapshot(cls) -> Dict[str, Any]:
        device = cls.get_device_info()
        process = psutil.Process(os.getpid())

        diag_data = {
            "platform": platform.system(),
            "platform_release": platform.release(),
            "platform_version": platform.version(),
            "cpu_percent": psutil.cpu_percent(interval=1),
            "ram_used_gb": process.memory_info().rss / (1024**3),
            "ram_available_gb": psutil.virtual_memory().available / (1024**3),
            "ram_percent": psutil.virtual_memory().percent,
            "process_id": os.getpid(),
            "process_name": process.name(),
            "process_threads": process.num_threads(),
        }

        if device == "cuda":
            try:
                diag_data["gpu_device_count"] = torch.cuda.device_count()
                diag_data["gpu_current_device"] = torch.cuda.current_device()
                diag_data["gpu_device_name"] = torch.cuda.get_device_name(0)
                diag_data["gpu_allocated_gb"] = torch.cuda.memory_allocated(0) / (1024**3)
                diag_data["gpu_reserved_gb"] = torch.cuda.memory_reserved(0) / (1024**3)
                diag_data["gpu_max_allocated_gb"] = torch.cuda.max_memory_allocated(0) / (1024**3)
                diag_data["gpu_memory_summary"] = torch.cuda.memory_summary(abbreviated=True)
            except Exception as e:
                logger.error(f"获取CUDA信息失败: {e}")
                diag_data["gpu_error"] = str(e)

        elif device == "mps":
            try:
                diag_data["mps_ready"] = torch.backends.mps.is_available()
                diag_data["mps_built"] = torch.backends.mps.is_built()
                diag_data["sys_mem_free_gb"] = psutil.virtual_memory().available / (1024**3)
                diag_data["sys_mem_total_gb"] = psutil.virtual_memory().total / (1024**3)
            except Exception as e:
                logger.error(f"获取MPS信息失败: {e}")
                diag_data["mps_error"] = str(e)

        logger.error(f"--- 崩溃前设备状态快照 ---\n{diag_data}")
        return diag_data

    @staticmethod
    def clear_cache():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            logger.info("已清理CUDA缓存")
        elif torch.backends.mps.is_available():
            gc.collect()
            logger.info("已触发MPS内存回收")
        else:
            gc.collect()
            logger.info("已清理CPU内存")

    @staticmethod
    def get_memory_usage() -> Dict[str, float]:
        process = psutil.Process(os.getpid())
        device = CrossPlatformDiagnostics.get_device_info()

        memory_info = {
            "ram_used_gb": process.memory_info().rss / (1024**3),
            "ram_percent": psutil.virtual_memory().percent,
            "ram_available_gb": psutil.virtual_memory().available / (1024**3),
        }

        if device == "cuda":
            memory_info["gpu_allocated_gb"] = torch.cuda.memory_allocated(0) / (1024**3)
            memory_info["gpu_reserved_gb"] = torch.cuda.memory_reserved(0) / (1024**3)
            memory_info["gpu_max_allocated_gb"] = torch.cuda.max_memory_allocated(0) / (1024**3)

        return memory_info

    @staticmethod
    def check_memory_threshold(threshold_percent: float = 85.0) -> bool:
        process = psutil.Process(os.getpid())
        ram_percent = psutil.virtual_memory().percent

        if ram_percent > threshold_percent:
            logger.warning(f"内存使用率过高: {ram_percent:.2f}%")
            return True

        device = CrossPlatformDiagnostics.get_device_info()
        if device == "cuda":
            gpu_allocated = torch.cuda.memory_allocated(0)
            gpu_total = torch.cuda.get_device_properties(0).total_memory
            gpu_percent = (gpu_allocated / gpu_total) * 100

            if gpu_percent > threshold_percent:
                logger.warning(f"GPU显存使用率过高: {gpu_percent:.2f}%")
                return True

        return False

    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        return {
            "platform": platform.system(),
            "platform_release": platform.release(),
            "platform_version": platform.version(),
            "architecture": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": psutil.cpu_count(logical=True),
            "cpu_count_physical": psutil.cpu_count(logical=False),
            "ram_total_gb": psutil.virtual_memory().total / (1024**3),
            "disk_total_gb": psutil.disk_usage("/").total / (1024**3),
            "disk_used_gb": psutil.disk_usage("/").used / (1024**3),
            "disk_free_gb": psutil.disk_usage("/").free / (1024**3),
        }

    @staticmethod
    def diagnose_oom_error(error: RuntimeError) -> Dict[str, Any]:
        err_msg = str(error).lower()
        is_oom = (
            "out of memory" in err_msg
            or "allotted memory" in err_msg
            or "cuda out of memory" in err_msg
        )

        diagnosis = {
            "is_oom": is_oom,
            "error_message": str(error),
            "error_type": type(error).__name__,
            "device": CrossPlatformDiagnostics.get_device_info(),
            "memory_snapshot": CrossPlatformDiagnostics.get_memory_usage(),
        }

        if is_oom:
            logger.critical(f"🚨 检测到 OOM 异常！平台: {diagnosis['device']}")
            CrossPlatformDiagnostics.dump_memory_snapshot()
            CrossPlatformDiagnostics.clear_cache()

        return diagnosis
