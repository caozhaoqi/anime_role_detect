#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T2I 服务回归测试：固化 2026-08-20 审查修复（P0 训练链路 / P1 双路径/鉴权/作业清理）。

覆盖点（均不依赖 torch/diffusers，纯逻辑测试）：
- P0-1: build_examples 兼容两列/四列 CSV（不再 KeyError）；build_metadata 输出四列
- P0-2: start_training 子进程 cmd 显式传 --base-model 指向本地 SD15_DIR
- P1-1: _ensure_ip_adapter 挂载前卸载残留 LoRA（unload_lora_weights）
- P1-2: app 存在 internal-token 认证中间件（非本机访问需令牌）
- P1-3: 作业 TTL 清理 + to_dict(include_images=False) 裁剪 base64
"""
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.services.t2i_service import training as T  # noqa: E402
from src.services.t2i_service.training import GenerateJob  # noqa: E402


# =============================================================================
# P0-1: CSV 格式兼容
# =============================================================================
class TestMetadataCSVCompat:
    def test_build_examples_four_column_csv(self, tmp_path):
        """四列 CSV（新版 build_metadata 产物）+ --role 过滤正确。"""
        csv_path = tmp_path / "meta.csv"
        csv_path.write_text(
            "image_path,caption,role,has_identity_token\n"
            "/x/amber/1.jpg,amber solo,amber,True\n"
            "/x/paimon/1.jpg,paimon solo,paimon,True\n",
            encoding="utf-8",
        )
        from scripts.t2i.train_lora_sd15 import build_examples
        examples = build_examples(str(csv_path), role="amber")
        assert len(examples) == 1
        assert examples[0][0] == "/x/amber/1.jpg"

    def test_build_examples_two_column_csv_no_keyerror(self, tmp_path):
        """旧版两列 CSV（缺 role 列）不抛 KeyError（row.get 防御）。"""
        csv_path = tmp_path / "meta_old.csv"
        csv_path.write_text(
            "image_path,caption\n"
            "/x/amber/2.jpg,amber solo\n",
            encoding="utf-8",
        )
        from scripts.t2i.train_lora_sd15 import build_examples
        examples = build_examples(str(csv_path), role="amber")
        assert len(examples) == 1

    def test_build_metadata_writes_role_column(self, tmp_path, monkeypatch):
        """build_metadata 输出四列（image_path, caption, role, has_identity_token）。"""
        import src.services.t2i_service.training as training_mod
        ref_dir = tmp_path / "amber"
        ref_dir.mkdir()
        (ref_dir / "1.jpg").write_bytes(b"x")
        monkeypatch.setattr(training_mod.config, "DATASET_ROOT", tmp_path)
        lora_dir = tmp_path / "out_lora"
        monkeypatch.setattr(training_mod.config, "LORA_DIR", lora_dir)

        csv_path = training_mod.build_metadata("amber")
        header = csv_path.read_text(encoding="utf-8").splitlines()[0]
        assert "role" in header and "has_identity_token" in header


# =============================================================================
# P0-2: 训练子进程 base-model
# =============================================================================
class TestTrainingCmd:
    def test_start_training_passes_base_model(self):
        """start_training 构造的 cmd 必须显式传 --base-model（本地路径，非 HF id）。"""
        import inspect
        src = inspect.getsource(T.start_training)
        assert "--base-model" in src
        assert "config.SD15_DIR" in src


# =============================================================================
# P1-1: 双路径切换 LoRA 清理
# =============================================================================
class TestPathSwitching:
    def test_ensure_ip_adapter_unloads_lora(self):
        """_ensure_ip_adapter 挂载前必须卸载残留 LoRA，防双重条件注入。"""
        import inspect
        from src.services.t2i_service.generator import T2IGenerator
        src = inspect.getsource(T2IGenerator._ensure_ip_adapter)
        assert "unload_lora_weights" in src
        assert "_lora_loaded_for is not None" in src


# =============================================================================
# P1-2: 鉴权中间件
# =============================================================================
class TestAuthMiddleware:
    def test_app_has_internal_auth_middleware(self):
        """app.py 含 internal-token 认证中间件与豁免路径。"""
        app_path = ROOT / "src/services/t2i_service/app.py"
        src = app_path.read_text(encoding="utf-8")
        assert "internal_auth_middleware" in src
        assert "X-Internal-Service-Token" in src or "INTERNAL_SERVICE_TOKEN" in src
        assert "/api/health" in src  # 健康检查豁免


# =============================================================================
# P1-3: 作业 TTL 清理 + base64 裁剪
# =============================================================================
class TestJobLifecycle:
    def test_cleanup_expired_jobs(self):
        """已完成且超 TTL 的作业被清理，运行中作业保留。"""
        old = GenerateJob(job_id="old1", role="x", status="succeeded",
                          finished_at=time.time() - T.JOB_TTL_SECONDS - 60)
        fresh = GenerateJob(job_id="new1", role="x", status="running")
        with T._JOBS_LOCK:
            T._GEN_JOBS["old1"] = old
            T._GEN_JOBS["new1"] = fresh
        try:
            T._cleanup_expired_jobs()
            with T._JOBS_LOCK:
                assert "old1" not in T._GEN_JOBS, "过期作业未被清理"
                assert "new1" in T._GEN_JOBS, "运行中作业被误删"
        finally:
            with T._JOBS_LOCK:
                T._GEN_JOBS.pop("old1", None)
                T._GEN_JOBS.pop("new1", None)

    def test_to_dict_trims_images(self):
        """列表接口 to_dict(include_images=False) 裁剪 base64，单查保留全量。"""
        job = GenerateJob(job_id="abc", role="amber", status="succeeded")
        job.result = {
            "role": "amber", "method": "ip_adapter",
            "images": ["data:image/png;base64,AAAA", "data:image/png;base64,BBBB"],
            "saved_paths": ["/x/1.png"],
        }
        full = job.to_dict()
        trim = job.to_dict(include_images=False)
        assert len(full["result"]["images"]) == 2
        assert "images" not in trim["result"]
        assert trim["result"]["image_count"] == 2

    def test_job_ttl_constant(self):
        assert T.JOB_TTL_SECONDS > 0
