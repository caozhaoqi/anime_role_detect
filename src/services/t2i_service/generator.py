#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""角色图像生成核心：SD1.5 + IP-Adapter（免训练）/ LoRA（需训练）。

逻辑直接复用已验证可跑的 scripts/t2i/ip_adapter_poc.py，封装为可复用服务组件：
- 懒加载 SD1.5 + IP-Adapter 权重（单例，首次生成时加载，之后常驻）
- 支持两条路线：
    ip_adapter : 吃 data/final_dataset/<role> 参考图，免训练秒级出图（一致性 70~85%）
    lora       : 加载 outputs/t2i_lora/<role>_v1/ 下训练好的 LoRA（一致性 95%+），
                 若该角色未训练则回退 ip_adapter 并在返回中标注
- 线程池执行阻塞式推理，避免阻塞事件循环
"""
from __future__ import annotations

import atexit
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

from src.services.t2i_service import config
from src.core.metrics import metrics
import logging
import sys
from typing import Optional, Callable

# t2i 专用 logger：带时间戳，解决 supervisord 捕获的 stdout/stderr 无时间问题
# （uvicorn 默认格式缺 asctime；裸 logging 也无 formatter）
_t2i_logger = logging.getLogger("t2i")
if not _t2i_logger.handlers:
    _t2i_handler = logging.StreamHandler(sys.stdout)
    _t2i_handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
    )
    _t2i_logger.addHandler(_t2i_handler)
    _t2i_logger.setLevel(logging.INFO)
    _t2i_logger.propagate = False

_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="t2i-gen")

# 空闲超过该秒数后自动卸载权重释放内存（默认 5 分钟；可在 config 覆盖）
IDLE_UNLOAD_SECONDS = int(getattr(config, "IDLE_UNLOAD_SECONDS", 300))

# 进程退出时关闭线程池，避免 leaked semaphore 警告与孤儿内存
atexit.register(lambda: _executor.shutdown(wait=False, cancel_futures=True))


def _resolve_device(req: str | None) -> str:
    if req and req != "auto":
        return req
    import torch
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        dev = "mps"
    else:
        dev = "cpu"
    _t2i_logger.info(f"[t2i] 解析设备: {dev}" + ("（警告：CPU 推理极慢，单步可达数百秒）" if dev == "cpu" else ""))
    return dev


def _load_role_refs(role: str, num_ref: int, seed: int = 42):
    """从 data/final_dataset/<role> 载入参考图，必要时随机采样上限。"""
    from PIL import Image

    ref_dir = config.DATASET_ROOT / role
    if not ref_dir.exists():
        raise FileNotFoundError(f"参考图目录不存在: {ref_dir}")

    exts = (".jpg", ".jpeg", ".png", ".webp")
    paths = sorted(p for p in ref_dir.iterdir() if p.suffix.lower() in exts)
    if not paths:
        raise FileNotFoundError(f"{ref_dir} 下无图片")

    refs = [Image.open(p).convert("RGB") for p in paths]
    if num_ref and len(refs) > num_ref:
        import random as _rnd
        rng = _rnd.Random(seed)
        refs = rng.sample(refs, num_ref)
    return refs


class T2IGenerator:
    """单例生成器：持有 SD1.5 pipeline + IP-Adapter 权重。"""

    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        self._pipe = None
        self._image_encoder = None
        self._device = None
        self._dtype = None  # 与设备匹配的权重精度；mps→fp16 / cpu→fp32
        self._ip_loaded = False
        self._lora_loaded_for = None  # 当前已注入的 LoRA 角色
        self._busy = False            # 是否正在推理（idle 卸载时需跳过）
        self._last_used = 0.0        # 上次生成的时刻，用于空闲判定
        self._idle_thread = None     # 空闲监控线程

    @classmethod
    def get_instance(cls) -> "T2IGenerator":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    # ---- 懒加载 ----
    def _ensure_base(self, device: str):
        import torch
        from diffusers import StableDiffusionPipeline

        if self._pipe is not None and self._device == device:
            return
        # 重要：MPS 上 fp16 会显著削弱 IP-Adapter 角色一致性（实测生成图与角色偏离较大），
        # 故统一使用 fp32 保真。fp16 的提速收益已让位于画质；若确需提速，应改用其他手段
        # （如 torch.compile、降低 steps），而非牺牲 IP-Adapter 精度。
        # CPU 同样 fp32。
        dtype = torch.float32
        _t2i_logger.info(f"[t2i] 加载 SD1.5 底座: {config.SD15_DIR} (device={device}, dtype={dtype})")
        pipe = StableDiffusionPipeline.from_pretrained(
            str(config.SD15_DIR),
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )
        pipe = pipe.to(device)
        # MPS 统一内存峰值治理：VAE tiling 显著降低解码阶段峰值（最关键）。
        # 注意：enable_attention_slicing 与 IP-Adapter 注意力处理器不兼容，
        # load_ip_adapter 时会触发 SlicedAttnProcessor 缺参崩溃，故不启用。
        try:
            pipe.enable_vae_tiling()
        except Exception:
            pass
        self._pipe = pipe
        self._device = device
        self._dtype = dtype
        self._ip_loaded = False
        self._lora_loaded_for = None

    def _ensure_ip_adapter(self, device: str, scale: float):
        import torch
        from transformers import CLIPVisionModelWithProjection

        if self._ip_loaded:
            self._pipe.set_ip_adapter_scale(scale)
            return
        self._ensure_base(device)
        _t2i_logger.info(f"[t2i] 加载 IP-Adapter 权重: {config.IP_MODELS_DIR / config.IP_WEIGHT_NAME}")
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            str(config.IP_MODELS_DIR / "image_encoder"),
            torch_dtype=self._dtype or torch.float32,
        ).to(device)
        self._pipe.load_ip_adapter(
            str(config.IP_MODELS_DIR),
            subfolder="",
            weight_name=config.IP_WEIGHT_NAME,
            image_encoder=image_encoder,
        )
        self._pipe.set_ip_adapter_scale(scale)
        self._ip_loaded = True
        self._lora_loaded_for = None

    def _ensure_lora(self, role: str, device: str):
        lora_dir = config.LORA_DIR / f"{role}_v1"
        lora_file = lora_dir / "pytorch_lora_weights.safetensors"
        if not lora_file.exists():
            lora_file = lora_dir / "pytorch_lora_weights.bin"
        if not lora_file.exists():
            return False
        self._ensure_base(device)
        if self._lora_loaded_for != role:
            # 若之前挂了 IP-Adapter，先卸载以免冲突
            if self._ip_loaded:
                try:
                    self._pipe.unload_ip_adapter()
                except Exception:
                    pass
                self._ip_loaded = False
            _t2i_logger.info(f"[t2i] 加载 LoRA: {lora_file}")
            self._pipe.load_lora_weights(str(lora_dir))
            self._lora_loaded_for = role
        return True

    # ---- 生成 ----
    def generate_sync(
        self,
        role: str,
        prompt: str | None = None,
        negative: str | None = None,
        scale: float = config.DEFAULT_SCALE,
        steps: int = config.DEFAULT_STEPS,
        cfg: float = config.DEFAULT_CFG,
        num: int = config.DEFAULT_NUM,
        method: str = "ip_adapter",
        num_ref: int = config.DEFAULT_NUM_REF,
        seed: int = 42,
        device: str | None = None,
        out_dir: Path | None = None,
        on_progress: Optional[Callable[[int, int, int], None]] = None,
    ) -> dict:
        import torch
        from PIL import Image

        device = _resolve_device(device)
        self._busy = True
        self._last_used = time.time()
        self._ensure_idle_monitor()
        _t2i_logger.info(f"[t2i-gen:{role}] 开始生成 role={role} method={method} num={num} steps={steps} cfg={cfg} device={device}")
        # ---- E: metrics 埋点（耗时 / 设备 / MPS 峰值）----
        metrics.inc_counter("t2i.generate.jobs")
        _t_start = time.time()
        _mps_peak_mb = 0.0
        try:
            neg = negative or "low quality, blurry, extra characters, multiple people, deformed, watermark"
            _t2i_logger.info(f"[t2i-gen:{role}] 进入生成流程 requested_method={method}")

            used_method = method
            if method == "lora":
                ok = self._ensure_lora(role, device)
                if not ok:
                    _t2i_logger.info(f"[t2i-gen:{role}] 无 LoRA 权重，回退 ip_adapter")
                    used_method = "ip_adapter"  # 回退
            if used_method == "ip_adapter":
                _t2i_logger.info(f"[t2i-gen:{role}] 加载/校验 IP-Adapter 权重中…")
                self._ensure_ip_adapter(device, scale)
                _t2i_logger.info(f"[t2i-gen:{role}] IP-Adapter 权重就绪")
                refs = _load_role_refs(role, num_ref, seed)
                _t2i_logger.info(f"[t2i-gen:{role}] 参考图载入完成: {len(refs)} 张 (num_ref={num_ref})")
            else:
                refs = None

            # IP-Adapter 的文本 prompt 不应含角色名（避免 CLIP 漂移），由参考图注入身份
            if used_method == "ip_adapter":
                txt = prompt or "solo character, anime style, high quality, detailed background"
            else:
                txt = prompt or f"{role}, solo, anime style, high quality, detailed"

            out_dir = out_dir or (config.T2I_OUTPUT_DIR / role)
            out_dir.mkdir(parents=True, exist_ok=True)

            images_b64 = []
            saved = []
            _t2i_logger.info(f"[t2i-gen:{role}] 进入推理循环 num={num} steps={steps} device={device}")
            for i in range(num):
                _t2i_logger.info(f"[t2i-gen:{role}] 开始第 {i+1}/{num} 张推理 (steps={steps})")
                # 必须让随机源与模型同设备，否则 mps 模型会触发 CPU 回退慢路径
                if device == "cpu":
                    gen = torch.Generator().manual_seed(seed + i)
                else:
                    gen = torch.Generator(device=device).manual_seed(seed + i)
                kwargs = dict(
                    prompt=txt,
                    negative_prompt=neg,
                    num_inference_steps=steps,
                    guidance_scale=cfg,
                    generator=gen,
                )
                if used_method == "ip_adapter":
                    kwargs["ip_adapter_image"] = [refs]
                # 逐步骤度：把第 i 张图的去噪进度回传给上层 on_progress(img_idx, completed, total)
                if on_progress is not None:
                    _img_idx = i
                    def _step_cb(pipe, step_i, timestep, cb_kwargs, _idx=_img_idx):
                        try:
                            on_progress(_idx, step_i + 1, steps)
                        except Exception:
                            pass
                        return cb_kwargs
                    kwargs["callback_on_step_end"] = _step_cb
                _t_img = time.time()
                out: Image.Image = self._pipe(**kwargs).images[0]
                _dt = time.time() - _t_img
                _t2i_logger.info(f"[t2i-gen:{role}] 第 {i+1}/{num} 张推理完成 耗时 {_dt:.1f}s")

                fname = f"{role}_{used_method}_{i + 1}.png"
                save_path = out_dir / fname
                out.save(save_path)
                saved.append(str(save_path))

                buf = BytesIO()
                out.save(buf, format="PNG")
                images_b64.append(
                    "data:image/png;base64," + __import__("base64").b64encode(buf.getvalue()).decode()
                )
                # MPS 分配器缓存不主动归还系统，逐张释放以拉平内存尖峰
                if device == "mps":
                    try:
                        torch.mps.empty_cache()
                        _mps_peak_mb = max(_mps_peak_mb, torch.mps.current_allocated_memory() / 1024 / 1024)
                    except Exception:
                        pass
            _t2i_logger.info(f"[t2i-gen:{role}] 全部 {num} 张推理完成，进入保存/编码阶段")
        finally:
            self._busy = False
            self._last_used = time.time()
            # ---- E: 记录本次生成耗时 / MPS 峰值 / 设备 ----
            try:
                _elapsed = time.time() - _t_start
                metrics.record_latency("t2i.generate.duration", _elapsed)
                metrics.set_gauge("t2i.generate.mps_peak_mb", round(_mps_peak_mb, 1))
                metrics.set_gauge("t2i.generate.device_is_mps", 1.0 if device == "mps" else 0.0)
            except Exception:
                pass
            # MPS 分配器会缓存显存不立即归还系统，推理后主动释放降低内存占用
            if device == "mps":
                try:
                    torch.mps.empty_cache()
                except Exception:
                    pass
        return {
            "role": role,
            "method": used_method,
            "requested_method": method,
            "fell_back": used_method != method,
            "prompt": txt,
            "images": images_b64,
            "saved_paths": saved,
            "device": device,
        }

    async def generate(self, **kwargs) -> dict:
        loop = __import__("asyncio").get_event_loop()
        return await loop.run_in_executor(_executor, lambda: self.generate_sync(**kwargs))

    def _ensure_idle_monitor(self):
        """懒启动一个 daemon 线程，空闲超过阈值后自动卸载权重释放内存。"""
        if self._idle_thread is None or not self._idle_thread.is_alive():
            self._idle_thread = threading.Thread(
                target=self._idle_loop, name="t2i-idle", daemon=True
            )
            self._idle_thread.start()

    def _idle_loop(self):
        while True:
            time.sleep(30)
            try:
                # 仅在「已加载 + 空闲 + 当前未在推理」时卸载，避免推理中途误杀
                if (
                    self._pipe is not None
                    and not self._busy
                    and (time.time() - self._last_used) > IDLE_UNLOAD_SECONDS
                ):
                    _t2i_logger.info(
                        f"[t2i] 空闲超过 {IDLE_UNLOAD_SECONDS}s，自动卸载权重以释放内存"
                    )
                    metrics.inc_counter("t2i.idle_unloads")
                    self.unload()
            except Exception:  # noqa: BLE001
                pass

    def unload(self):
        """释放权重（显存/内存）。"""
        self._pipe = None
        self._image_encoder = None
        self._ip_loaded = False
        self._lora_loaded_for = None
        self._device = None
        self._last_used = 0.0
        self._busy = False
        import gc, torch
        gc.collect()
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            torch.mps.empty_cache()
