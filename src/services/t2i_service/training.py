#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LoRA 训练作业管理：把 train_lora_sd15.py 作为子进程异步跑（Mac CPU 训练较慢）。

- start_training(role, params) 生成该角色的 metadata.csv（基于 data/final_dataset/<role> 参考图），
  然后以 t2i-mac venv 的 python 启动训练脚本，返回 job_id。
- get_job(job_id) 轮询作业状态/日志。
"""
from __future__ import annotations

import csv
import os
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.services.t2i_service import config
from src.core.metrics import metrics
import logging
import sys

# t2i 专用 logger（与 generator.py 同名，logging 缓存复用，guard 防止重复加 handler）
_t2i_logger = logging.getLogger("t2i")
if not _t2i_logger.handlers:
    _t2i_handler = logging.StreamHandler(sys.stdout)
    _t2i_handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
    )
    _t2i_logger.addHandler(_t2i_handler)
    _t2i_logger.setLevel(logging.INFO)
    _t2i_logger.propagate = False


@dataclass
class TrainJob:
    job_id: str
    role: str
    status: str = "queued"          # queued | running | succeeded | failed
    log: list[str] = field(default_factory=list)
    progress: str = ""
    created_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    output_dir: str = ""

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "role": self.role,
            "status": self.status,
            "progress": self.progress,
            "log_tail": self.log[-50:],
            "log_lines": len(self.log),
            "created_at": self.created_at,
            "finished_at": self.finished_at,
            "output_dir": self.output_dir,
        }


@dataclass
class GenerateJob:
    """异步图像生成作业：提交即返回 job_id，后台线程执行推理，前端轮询进度。

    与 TrainJob 共用 _JOBS_LOCK 与 get_job/list_jobs，避免长连接超时与 UI 卡死。
    """
    job_id: str
    role: str
    status: str = "queued"          # queued | running | succeeded | failed
    progress: str = ""
    created_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    result: Optional[dict] = None   # 成功后写入 generate_sync 的完整返回
    error: Optional[str] = None
    params: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {
            "job_id": self.job_id,
            "type": "generate",
            "role": self.role,
            "status": self.status,
            "progress": self.progress,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
            "params": self.params,
        }
        if self.result is not None:
            d["result"] = self.result
        if self.error:
            d["error"] = self.error
        return d


_JOBS: dict[str, TrainJob] = {}
_GEN_JOBS: dict[str, GenerateJob] = {}
_JOBS_LOCK = threading.Lock()


def start_generation(
    role: str,
    method: str = "ip_adapter",
    prompt: Optional[str] = None,
    negative: Optional[str] = None,
    scale: float = config.DEFAULT_SCALE,
    steps: int = config.DEFAULT_STEPS,
    cfg: float = config.DEFAULT_CFG,
    num: int = config.DEFAULT_NUM,
    num_ref: int = config.DEFAULT_NUM_REF,
    seed: int = 42,
    device: Optional[str] = None,
) -> GenerateJob:
    """异步提交一次图像生成作业，立即返回 job_id；后台线程执行推理。

    与 start_training 同样采用"提交即返回"模式，规避长连接超时与前端卡死。
    首次生成加载 SD1.5 + IP-Adapter 需数十秒，期间连接已释放，UI 可正常轮询。
    """
    params = dict(
        method=method, prompt=prompt, negative=negative, scale=scale,
        steps=steps, cfg=cfg, num=num, num_ref=num_ref, seed=seed, device=device,
    )
    job = GenerateJob(
        job_id=uuid.uuid4().hex[:12],
        role=role,
        status="queued",
        progress="排队中…",
        params=params,
    )
    with _JOBS_LOCK:
        _GEN_JOBS[job.job_id] = job

    # 延迟导入，避免模块加载期出现循环依赖
    from src.services.t2i_service.generator import T2IGenerator

    def _run():
        gen = T2IGenerator.get_instance()
        try:
            with _JOBS_LOCK:
                j = _GEN_JOBS.get(job.job_id)
                if j:
                    j.status = "running"
                    j.progress = "加载模型并生成中…（首次约 30–60s）"
            result = gen.generate_sync(
                role=role, prompt=prompt, negative=negative, scale=scale,
                steps=steps, cfg=cfg, num=num, method=method, num_ref=num_ref,
                seed=seed, device=device,
            )
            with _JOBS_LOCK:
                j = _GEN_JOBS.get(job.job_id)
                if j:
                    j.status = "succeeded"
                    j.result = result
                    j.progress = f"完成（{len(result.get('images', []))} 张，{result.get('method')}）"
                    j.finished_at = time.time()
        except Exception as e:  # noqa: BLE001
            with _JOBS_LOCK:
                j = _GEN_JOBS.get(job.job_id)
                if j:
                    j.status = "failed"
                    j.error = str(e)
                    j.progress = "失败"
                    j.finished_at = time.time()
            _t2i_logger.info(f"[t2i-gen:{role}] 生成失败: {e}")

    threading.Thread(target=_run, name=f"t2i-gen-{role}", daemon=True).start()
    return job


def build_metadata(role: str) -> Path:
    """为角色生成 metadata.csv（image_path, caption），供 train_lora_sd15.py 使用。"""
    ref_dir = config.DATASET_ROOT / role
    if not ref_dir.exists():
        raise FileNotFoundError(f"参考图目录不存在: {ref_dir}")

    exts = (".jpg", ".jpeg", ".png", ".webp")
    paths = sorted(p for p in ref_dir.iterdir() if p.suffix.lower() in exts)
    if not paths:
        raise FileNotFoundError(f"{ref_dir} 下无图片")

    out_dir = config.LORA_DIR / f"{role}_v1"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "metadata.csv"
    # 基础 caption：角色 token + 动漫风格；足够 PoC 级 LoRA，后续可接 prepare_captions 精标
    caption = f"{role}, solo character, anime style, high quality, detailed"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "caption"])
        for p in paths:
            w.writerow([str(p), caption])
    return csv_path


def start_training(
    role: str,
    rank: int = 16,
    epochs: int = 10,
    resolution: int = 512,
    lr: float = 1e-4,
    batch_size: int = 2,
) -> TrainJob:
    csv_path = build_metadata(role)
    out_dir = config.LORA_DIR / f"{role}_v1"

    job = TrainJob(
        job_id=uuid.uuid4().hex[:12],
        role=role,
        output_dir=str(out_dir),
        status="running",
    )
    with _JOBS_LOCK:
        _JOBS[job.job_id] = job

    venv_py = config.T2I_VENV_PYTHON
    train_script = config.SCRIPTS_T2I / "train_lora_sd15.py"
    cmd = [
        str(venv_py), str(train_script),
        "--metadata", str(csv_path),
        "--role", role,
        "--output-dir", str(out_dir),
        "--rank", str(rank),
        "--resolution", str(resolution),
        "--num-train-epochs", str(epochs),
        "--learning-rate", str(lr),
        "--train-batch-size", str(batch_size),
    ]

    def _pump():
        try:
            t0 = time.time()
            metrics.inc_counter("t2i.train.jobs")
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                line = line.rstrip("\n")
                with _JOBS_LOCK:
                    j = _JOBS.get(job.job_id)
                    if j:
                        j.log.append(line)
                        if "loss" in line:
                            j.progress = line
                _t2i_logger.info(f"[t2i-train:{role}] {line}")
            rc = proc.wait()
            with _JOBS_LOCK:
                j = _JOBS.get(job.job_id)
                if j:
                    j.finished_at = time.time()
                    j.status = "succeeded" if rc == 0 else "failed"
                    j.progress = f"exit_code={rc}"
            # ---- E: 训练耗时 / 设备 / 完成计数 ----
            metrics.record_latency("t2i.train.duration", time.time() - t0)
            metrics.set_gauge("t2i.train.device_is_mps", 0.0)  # 训练当前强制 CPU（见 train_lora_sd15._detect_device）
            metrics.inc_counter("t2i.train.succeeded" if rc == 0 else "t2i.train.failed")
        except Exception as e:  # noqa: BLE001
            with _JOBS_LOCK:
                j = _JOBS.get(job.job_id)
                if j:
                    j.status = "failed"
                    j.log.append(f"[error] {e}")
                    j.finished_at = time.time()

    threading.Thread(target=_pump, name=f"t2i-train-{role}", daemon=True).start()
    return job


def get_job(job_id: str):
    """按 job_id 查找训练或生成作业（两者共用 id 命名空间，互不冲突）。"""
    with _JOBS_LOCK:
        j = _JOBS.get(job_id)
        if j is not None:
            return j
        return _GEN_JOBS.get(job_id)


def list_jobs():
    """返回全部训练 + 生成作业（按创建时间倒序）。"""
    with _JOBS_LOCK:
        all_jobs = list(_JOBS.values()) + list(_GEN_JOBS.values())
    all_jobs.sort(key=lambda j: j.created_at, reverse=True)
    return all_jobs
