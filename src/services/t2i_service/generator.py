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
    """从 data/final_dataset/<role> 载入参考图，必要时随机采样上限。

    P2 修复（2026-08-20）：先采样路径、再打开图片，避免头部角色（100+ 图）
    全部载入内存后才丢弃。
    """
    from PIL import Image

    ref_dir = config.DATASET_ROOT / role
    if not ref_dir.exists():
        raise FileNotFoundError(f"参考图目录不存在: {ref_dir}")

    exts = (".jpg", ".jpeg", ".png", ".webp")
    paths = sorted(p for p in ref_dir.iterdir() if p.suffix.lower() in exts)
    if not paths:
        raise FileNotFoundError(f"{ref_dir} 下无图片")

    if num_ref and len(paths) > num_ref:
        import random as _rnd
        rng = _rnd.Random(seed)
        paths = rng.sample(paths, num_ref)
    return [Image.open(p).convert("RGB") for p in paths]


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
        # 调度器优化（零画质风险提速）：DPM++(多步) 收敛远快于默认 PNDM，
        # 同质量下 steps 可从 30 降到 ~20（见 config.DEFAULT_STEPS）。它只替换 pipe.scheduler，
        # 与 IP-Adapter（UNet 注意力钩子）完全不冲突。try/except 兜底，失败则保留原调度器。
        try:
            from diffusers import DPMSolverMultistepScheduler
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
            _t2i_logger.info(
                f"[t2i] 调度器已切换为 DPMSolverMultistepScheduler（steps 默认 {config.DEFAULT_STEPS}，同质量更省步）"
            )
        except Exception as _se:  # noqa: BLE001
            _t2i_logger.warning(f"[t2i] 调度器切换失败，保留默认 PNDM：{_se}")
        # 实验性提速：编译 UNet 计算图。不牺牲精度（区别于 fp16），仅融合/重排算子提速。
        # - 必须在 IP-Adapter/LoRA 注入前编译，使编译图包含已挂载的注意力处理器。
        # - MPS 上 inductor 后端支持有限，可能 graph break 或回退 eager（无提速但不崩）。
        #   开启 suppress_errors：即使首步编译失败也自动降级 eager，不会让整次生成 500。
        # - 首次推理触发编译开销（首图更慢），后续复用缓存。
        # - torch.compile 返回的 OptimizedModule 会转发属性访问（已验证 2.3.1），
        #   故 load_ip_adapter / set_ip_adapter_scale 仍能作用于内部原始 UNet。
        if getattr(config, "COMPILE_UNET", False):
            if device == "mps":
                # torch 2.3.1 的 inductor 后端显式断言 "Device mps not supported"：
                # 首次推理会触发 BackendCompilerFailed（靠 suppress_errors 不崩、自动回退 eager），
                # 但每个去噪步都会重试验图编译，把 .err.log 刷成吨警告且零提速。故 MPS 直接跳过。
                _t2i_logger.info(
                    "[t2i] 跳过 torch.compile：当前 torch 2.3.1 的 inductor 后端不支持 MPS，"
                    "编译只会回退 eager 且刷警告、无提速"
                )
            else:
                try:
                    import torch as _torch
                    _torch._dynamo.config.suppress_errors = True
                    # MPS 不支持 CUDA-graph 类 backend（reduce-overhead），统一用 default(inductor)
                    pipe.unet = _torch.compile(pipe.unet, mode="default", fullgraph=False)
                    _t2i_logger.info(f"[t2i] UNet 已提交 torch.compile（device={device}, mode=default）；首次推理将编译")
                except Exception as _ce:  # noqa: BLE001
                    _t2i_logger.warning(f"[t2i] torch.compile(UNet) 提交失败，回退 eager UNet: {_ce}")
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
        # P1 修复（2026-08-20）：若之前挂了 LoRA，先卸载再挂 IP-Adapter，
        # 否则 LoRA 权重仍挂在 UNet 上，双重条件注入污染生成结果。
        # （_ensure_lora 挂 LoRA 前已有 unload_ip_adapter，此处补上反向清理。）
        if self._lora_loaded_for is not None:
            try:
                self._pipe.unload_lora_weights()
                _t2i_logger.info(f"[t2i] 卸载残留 LoRA({self._lora_loaded_for})，准备挂载 IP-Adapter")
            except Exception as _le:  # noqa: BLE001
                _t2i_logger.warning(f"[t2i] 卸载 LoRA 失败（忽略）：{_le}")
            self._lora_loaded_for = None
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
            # 安全批生成：把 num 张按 T2I_MAX_BATCH 切分，每批一次 pipeline 调用
            # （num_images_per_prompt）并行出多张，摊薄 UNet 开销提速。
            # - MPS 统一内存峰值敏感（本机曾 89.8% OOM），故限制批大小上限；OOM 时把
            #   T2I_MAX_BATCH 调小（=1 即退化为逐张）即可。
            # - 画质/角色一致性不受影响：同模型同 IP-Adapter 条件，只是并行噪声采样。
            # - 种子：每个 chunk 用独立 generator（seed + chunk*batch_size）顺序抽 batch_size
            #   张，与旧逐张 seed+i 分布不同（不同随机样本，但非更差）。
            max_batch = int(getattr(config, "T2I_MAX_BATCH", 2))
            batch_size = 1 if num <= 1 else max(1, min(num, max_batch))
            _t2i_logger.info(
                f"[t2i-gen:{role}] 进入推理（批生成） num={num} batch_size={batch_size} steps={steps} device={device}"
            )
            global_idx = 0
            chunk = 0
            while global_idx < num:
                b = min(batch_size, num - global_idx)
                _t2i_logger.info(f"[t2i-gen:{role}] 开始第 {global_idx+1}..{global_idx+b}/{num} 张（batch={b}）推理")
                # 必须让随机源与模型同设备，否则 mps 模型会触发 CPU 回退慢路径
                if device == "cpu":
                    gen = torch.Generator().manual_seed(seed + chunk * batch_size)
                else:
                    gen = torch.Generator(device=device).manual_seed(seed + chunk * batch_size)
                kwargs = dict(
                    prompt=txt,
                    negative_prompt=neg,
                    num_inference_steps=steps,
                    guidance_scale=cfg,
                    generator=gen,
                    num_images_per_prompt=b,
                )
                if used_method == "ip_adapter":
                    kwargs["ip_adapter_image"] = [refs]
                # 注意：曾在此处挂 callback_on_step_end 做"真·逐步骤度"，但实测在
                # MPS + diffusers 0.30.1 上会导致每步强制设备同步，去噪步从 ~1.5s 暴涨到
                # ~400s/步（270× 变慢），整轮生成卡死 5.5h。故移除，进度改由 training.py
                # 的"时间估算线程"平滑驱动（零 MPS 开销）。
                _t_img = time.time()
                outs = self._pipe(**kwargs).images  # 列表：长度 = b
                _dt = time.time() - _t_img
                _t2i_logger.info(
                    f"[t2i-gen:{role}] 第 {global_idx+1}..{global_idx+b}/{num} 张推理完成 耗时 {_dt:.1f}s"
                )
                for _j, out in enumerate(outs):
                    _i = global_idx + _j
                    # P2 修复（2026-08-20）：文件名加时间戳前缀，避免同一角色同方法多次
                    # 生成覆盖旧文件（原 {role}_{method}_{i}.png 每次从 1 重算）
                    fname = f"{role}_{used_method}_{int(time.time())}_{_i + 1}.png"
                    save_path = out_dir / fname
                    out.save(save_path)
                    saved.append(str(save_path))
                    buf = BytesIO()
                    out.save(buf, format="PNG")
                    images_b64.append(
                        "data:image/png;base64," + __import__("base64").b64encode(buf.getvalue()).decode()
                    )
                global_idx += b
                chunk += 1
                # MPS 分配器缓存不主动归还系统，逐批释放以拉平内存尖峰
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
