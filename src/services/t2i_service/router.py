#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""t2i_service HTTP 路由：角色生成 / 训练 / 对话生成。"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.services.t2i_service import config
from src.services.t2i_service.training import get_job, list_jobs, start_training, start_generation
from src.core.service_registry import registry
from src.core.metrics import metrics

router = APIRouter()

# 自注册路由：t2i 服务在自己的模块里声明路由，网关无需回改 routing.py（装饰器自动发现）。
# 与网关默认注册的 "t2i" 规则同名 → 去重后仅保留一条，二者并存安全。
registry.register_route(
    name="t2i", service="t2i", match_prefix=["t2i/"], strip="t2i/",
    template="{base}/api/t2i/{path}",
)


# ----------------------------------------------------------------------
# 请求/响应模型
# ----------------------------------------------------------------------
class GenerateRequest(BaseModel):
    role: str
    prompt: Optional[str] = None
    negative: Optional[str] = None
    method: str = "ip_adapter"          # ip_adapter | lora
    scale: float = config.DEFAULT_SCALE
    steps: int = config.DEFAULT_STEPS
    cfg: float = config.DEFAULT_CFG
    num: int = config.DEFAULT_NUM
    num_ref: int = config.DEFAULT_NUM_REF
    seed: int = 42
    device: Optional[str] = None


class TrainRequest(BaseModel):
    role: str
    rank: int = 16
    epochs: int = 10
    resolution: int = 512
    lr: float = 1e-4
    batch_size: int = 2


class ChatRequest(BaseModel):
    message: str
    method: str = "ip_adapter"
    num: int = 1


# ----------------------------------------------------------------------
# 健康检查（网关聚合探测依赖此端点）
# ----------------------------------------------------------------------
@router.get("/api/health")
async def health():
    return {"status": "healthy", "service": "t2i", "version": "1.0.0"}


# ----------------------------------------------------------------------
# 角色列表（数据集中有参考图的角色）
# ----------------------------------------------------------------------
@router.get("/api/t2i/roles")
async def list_roles():
    exts = (".jpg", ".jpeg", ".png", ".webp")
    roles = []
    if config.DATASET_ROOT.exists():
        for d in sorted(config.DATASET_ROOT.iterdir()):
            if not d.is_dir():
                continue
            imgs = [p for p in d.iterdir() if p.suffix.lower() in exts]
            if imgs:
                lora_ready = (config.LORA_DIR / f"{d.name}_v1").exists()
                roles.append({"role": d.name, "image_count": len(imgs), "lora_ready": lora_ready})
    return {"success": True, "count": len(roles), "roles": roles}


# ----------------------------------------------------------------------
# 生成
# ----------------------------------------------------------------------
@router.post("/api/t2i/generate")
async def generate(req: GenerateRequest):
    """提交一次图像生成作业，立即返回 job_id；前端用 /jobs/{id} 轮询进度。

    采用与训练一致的异步模式：避免长连接超时（首次加载 SD1.5+IP-Adapter 需数十秒）与 UI 卡死。
    """
    try:
        job = start_generation(
            role=req.role,
            method=req.method,
            prompt=req.prompt,
            negative=req.negative,
            scale=req.scale,
            steps=req.steps,
            cfg=req.cfg,
            num=req.num,
            num_ref=req.num_ref,
            seed=req.seed,
            device=req.device,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {
        "success": True,
        "job_id": job.job_id,
        "role": job.role,
        "status": job.status,
        "message": "已提交生成任务，请使用 job_id 轮询 /api/t2i/jobs/{id} 获取进度与结果",
    }


# ----------------------------------------------------------------------
# 训练（异步作业）
# ----------------------------------------------------------------------
@router.post("/api/t2i/train")
async def train(req: TrainRequest):
    try:
        job = start_training(
            role=req.role,
            rank=req.rank,
            epochs=req.epochs,
            resolution=req.resolution,
            lr=req.lr,
            batch_size=req.batch_size,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"success": True, "job_id": job.job_id, "role": job.role, "status": job.status}


@router.get("/api/t2i/jobs/{job_id}")
async def job_status(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="作业不存在")
    return {"success": True, **job.to_dict()}


@router.get("/api/t2i/jobs")
async def jobs():
    return {"success": True, "jobs": [j.to_dict() for j in list_jobs()]}


# ----------------------------------------------------------------------
# 对话生成：解析"角色名 + 生成意图" → 调生成
# ----------------------------------------------------------------------
_INTENT_KW = ["生成", "画", "出图", "图片", "角色图", "绘", "做一张", "来一张",
              "generate", "make", "draw", "create", "image of", "picture"]


@router.post("/api/t2i/chat")
async def chat(req: ChatRequest):
    msg = req.message
    lower = msg.lower()

    # 列出角色名（含子串匹配，避免大小写/别名问题）
    role_names = [r["role"] for r in (await list_roles())["roles"]]
    hit_role = None
    for name in role_names:
        if name.lower() in lower:
            hit_role = name
            break

    has_intent = any(k in msg.lower() for k in _INTENT_KW) or any(k in lower for k in _INTENT_KW)

    if hit_role and has_intent:
        try:
            job = start_generation(role=hit_role, method=req.method, num=req.num)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"提交生成失败: {e}")
        return {
            "success": True,
            "job_id": job.job_id,
            "matched_role": hit_role,
            "reply": f"已提交为角色「{hit_role}」生成 {req.num} 张图的任务，正在生成中…",
        }

    if hit_role and not has_intent:
        return {
            "success": True,
            "matched_role": hit_role,
            "reply": f"识别到角色「{hit_role}」。说 '生成 / 画一张 {hit_role}' 即可出图。",
        }

    # 未识别角色：给出提示 + 部分角色名
    sample = "、".join(role_names[:12])
    return {
        "success": True,
        "matched_role": None,
        "reply": f"未识别到数据集中的角色。可用角色示例：{sample} …（说 '生成 <角色名>' 试试）",
        "available_roles": role_names,
    }


# ----------------------------------------------------------------------
# 显式卸载权重（释放显存/内存）
# ----------------------------------------------------------------------
@router.post("/api/t2i/unload")
async def unload():
    T2IGenerator.get_instance().unload()
    return {"success": True, "message": "已卸载生成模型权重"}


# ----------------------------------------------------------------------
# 指标端点（E：轻量可观测，量化生成耗时 / MPS 内存尖峰 / 重启次数）
# ----------------------------------------------------------------------
@router.get("/api/t2i/metrics")
async def get_metrics():
    """返回进程内 metrics 聚合（耗时 P50/P95、MPS 峰值、各计数器）。"""
    return {"success": True, **metrics.summary()}
