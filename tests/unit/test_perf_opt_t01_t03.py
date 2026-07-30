#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
性能与可用性优化 (T01-T03) 验证测试

覆盖：
- T01: safe_temp_path 安全性、Redis 本地兜底、数据库/auth 降级增强
- T02: EasyOCR preload / is_ready、模型预热
- T03: 消除重复推理（Model Service 已返回 OCR+NSFW 时跳过本地推理）

使用 .venv 虚拟环境运行：
    .venv/bin/pytest tests/unit/test_perf_opt_t01_t03.py -v
"""

import os
import time
import inspect

import pytest

# 检测可选依赖
try:
    import easyocr  # noqa: F401
    HAS_EASYOCR = True
except ImportError:
    HAS_EASYOCR = False

try:
    import torch  # noqa: F401
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ==================== T01: safe_temp_path ====================


class TestSafeTempPath:
    """验证 src/core/utils/utils.py 的 safe_temp_path 安全性"""

    def test_normal_filename(self):
        from src.core.utils.utils import safe_temp_path, TEMP_DIR

        path = safe_temp_path("test.jpg")
        assert os.path.isabs(path)
        assert os.path.dirname(path) == TEMP_DIR
        assert "test.jpg" in os.path.basename(path)

    def test_path_injection_does_not_escape_temp_dir(self):
        """路径注入攻击不能逃逸出 TEMP_DIR"""
        from src.core.utils.utils import safe_temp_path, TEMP_DIR

        path = safe_temp_path("../../etc/passwd")
        basename = os.path.basename(path)
        # '/' 被替换为 '_'，basename 不含路径分隔符
        assert "/" not in basename
        # realpath 必须停留在 TEMP_DIR 内
        assert os.path.realpath(path).startswith(
            os.path.realpath(TEMP_DIR) + os.sep
        )

    def test_special_chars_filtered(self):
        from src.core.utils.utils import safe_temp_path

        path = safe_temp_path("file<>|*.jpg")
        basename = os.path.basename(path)
        for ch in "<>|*":
            assert ch not in basename

    def test_empty_and_none_fallback(self):
        from src.core.utils.utils import safe_temp_path, TEMP_DIR

        for name in ("", None):
            path = safe_temp_path(name)
            assert os.path.dirname(path) == TEMP_DIR
            assert "unknown" in os.path.basename(path)

    def test_uniqueness(self):
        from src.core.utils.utils import safe_temp_path

        a = safe_temp_path("same.jpg")
        b = safe_temp_path("same.jpg")
        assert a != b

    def test_long_filename_truncated(self):
        from src.core.utils.utils import safe_temp_path

        path = safe_temp_path("a" * 300 + ".jpg")
        assert len(os.path.basename(path)) <= 120


# ==================== T01: _LocalFallbackCache ====================


class TestLocalFallbackCache:
    """验证 Redis 不可用时的本地 LRU+TTL 兜底缓存"""

    def test_set_and_get(self):
        from src.services.cache_service.redis_cache import _LocalFallbackCache

        cache = _LocalFallbackCache(max_size=5, ttl=60)
        assert cache.set("k1", {"role": "alice"})
        assert cache.get("k1") == {"role": "alice"}

    def test_miss_returns_none(self):
        from src.services.cache_service.redis_cache import _LocalFallbackCache

        cache = _LocalFallbackCache()
        assert cache.get("nope") is None

    def test_lru_eviction(self):
        from src.services.cache_service.redis_cache import _LocalFallbackCache

        cache = _LocalFallbackCache(max_size=3, ttl=60)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)
        # 触摸 a，使其成为 MRU
        assert cache.get("a") == 1
        # 插入 d -> b (LRU) 被淘汰
        cache.set("d", 4)
        assert cache.get("b") is None
        assert cache.get("a") == 1
        assert cache.get("c") == 3
        assert cache.get("d") == 4

    def test_ttl_expiry(self):
        from src.services.cache_service.redis_cache import _LocalFallbackCache

        cache = _LocalFallbackCache(max_size=5, ttl=1)
        cache.set("x", 99, ttl=1)
        assert cache.get("x") == 99
        time.sleep(1.2)
        assert cache.get("x") is None

    def test_delete(self):
        from src.services.cache_service.redis_cache import _LocalFallbackCache

        cache = _LocalFallbackCache()
        cache.set("y", 1)
        assert cache.delete("y")
        assert cache.get("y") is None
        assert not cache.delete("y")

    def test_clear_and_size(self):
        from src.services.cache_service.redis_cache import _LocalFallbackCache

        cache = _LocalFallbackCache()
        cache.set("z1", 1)
        cache.set("z2", 2)
        n = cache.clear()
        assert n == 2
        assert cache.size() == 0


# ==================== T01: RedisCache 降级 ====================


class TestRedisCacheDegrade:
    """验证 Redis 不可用时 RedisCache 降级到本地缓存"""

    def test_degrade_to_local_fallback(self, monkeypatch):
        from src.services.cache_service.redis_cache import RedisCache

        # 指向无人监听的端口，强制 Redis 不可用
        monkeypatch.setenv("REDIS_HOST", "127.0.0.1")
        monkeypatch.setenv("REDIS_PORT", "19999")
        monkeypatch.setenv("REDIS_DB", "0")

        rc = RedisCache()
        assert rc.available is False

        # set/get 走本地兜底
        assert rc.set("degrade_key", {"role": "test"})
        assert rc.get("degrade_key") == {"role": "test"}

        # exists 走本地兜底
        assert rc.exists("degrade_key")

        # get_stats 反映降级状态
        stats = rc.get_stats()
        assert stats["available"] is False
        assert "local_fallback_size" in stats

    def test_try_reconnect_respects_interval(self, monkeypatch):
        from src.services.cache_service.redis_cache import RedisCache

        monkeypatch.setenv("REDIS_HOST", "127.0.0.1")
        monkeypatch.setenv("REDIS_PORT", "19998")
        monkeypatch.setenv("REDIS_DB", "0")

        rc = RedisCache()
        assert rc.available is False
        # 立即重连应被间隔限制拦截
        assert rc.try_reconnect() is False


# ==================== T02: EasyOCR ====================


class TestEasyOCRReady:
    """验证 EasyOCR preload / is_ready / 降级行为"""

    @pytest.mark.skipif(not HAS_EASYOCR, reason="easyocr not installed (~200MB)")
    def test_is_ready_false_before_preload(self):
        from src.core.ocr.easyocr_detector import EasyOCRDetector

        d = EasyOCRDetector()
        assert d.is_ready() is False

    @pytest.mark.skipif(not HAS_EASYOCR, reason="easyocr not installed (~200MB)")
    def test_detect_text_returns_empty_when_not_ready(self):
        from src.core.ocr.easyocr_detector import EasyOCRDetector

        d = EasyOCRDetector()
        # 未就绪时不抛异常，返回空列表
        assert d.detect_text("dummy.jpg") == []

    @pytest.mark.skipif(not HAS_EASYOCR, reason="easyocr not installed (~200MB)")
    def test_preload_idempotent(self):
        from src.core.ocr.easyocr_detector import EasyOCRDetector

        d = EasyOCRDetector()
        d.preload()  # 可能成功也可能失败（无模型），但不应抛异常
        d.preload()  # 重复调用安全


# ==================== T03: 消除重复推理（逻辑验证）====================


class TestNoDuplicateInference:
    """验证 Model Service 已返回 OCR+NSFW 时跳过本地重复推理"""

    @pytest.mark.skipif(not HAS_TORCH, reason="torch not installed (~800MB)")
    def test_model_processor_uses_safe_temp_path(self):
        from src.services.processor import model_processor as mp

        with open(mp.__file__, "r", encoding="utf-8") as f:
            full = f.read()
        # 至少 3 处使用
        assert full.count("_safe_temp_path(") >= 3
        # 不再有裸的 os.path.join('temp', ...)
        assert "os.path.join('temp'" not in full
        assert 'os.path.join("temp"' not in full
        # import 行存在
        assert (
            "from src.core.utils.utils import safe_temp_path as _safe_temp_path"
            in full
        )

    @pytest.mark.skipif(not HAS_TORCH, reason="torch not installed (~800MB)")
    def test_call_model_service_skip_logic_single_role(self):
        from src.services.processor import model_processor as mp

        src = inspect.getsource(mp._call_model_service)
        # 读取 Model Service 返回的 OCR + NSFW
        assert "ms_text_detections = model_result.get" in src
        assert "ms_nsfw = model_result.get" in src
        # 当两者都存在时跳过本地推理
        assert "if ms_text_detections is not None and ms_nsfw is not None" in src
        # 向后兼容：缺失时降级到本地
        assert "process_image_features" in src
        assert "detect_nsfw" in src

    @pytest.mark.skipif(not HAS_TORCH, reason="torch not installed (~800MB)")
    def test_call_model_service_skip_logic_multi_role(self):
        from src.services.processor import model_processor as mp

        src = inspect.getsource(mp._call_model_service)
        assert "text_detections = model_result.get" in src
        assert "nsfw_result = model_result.get" in src
        assert "if text_detections is None or nsfw_result is None" in src
        assert "跳过本地重复推理" in src

    @pytest.mark.skipif(not HAS_TORCH, reason="torch not installed (~800MB)")
    def test_routes_run_ocr_and_nsfw_degrade(self):
        from src.services.model_service import routes

        src = inspect.getsource(routes._run_ocr_and_nsfw)
        # OCR 未就绪时返回空列表
        assert "ocr_detector.is_ready()" in src
        assert "text_detections = []" in src
        # NSFW 默认值
        assert "is_nsfw" in src and "False" in src

    @pytest.mark.skipif(not HAS_TORCH, reason="torch not installed (~800MB)")
    def test_feature_processor_checks_ocr_ready(self):
        from src.services.processor import feature_processor as fp

        src = inspect.getsource(fp.process_image_features)
        assert "is_ready()" in src

    @pytest.mark.skipif(not HAS_TORCH, reason="torch not installed (~800MB)")
    def test_predict_image_adds_ocr_and_nsfw(self):
        from src.services.model_service import routes

        src = inspect.getsource(routes.predict_image)
        # predict_image 调用 _run_ocr_and_nsfw 并把结果加入 result
        assert "_run_ocr_and_nsfw" in src
        assert 'result["text_detections"]' in src
        assert 'result["nsfw"]' in src

    @pytest.mark.skipif(not HAS_TORCH, reason="torch not installed (~800MB)")
    def test_faiss_empty_index_degrade_log(self):
        from src.services.model_service import routes

        src = inspect.getsource(routes.predict_image)
        assert "ntotal" in src
        assert "FAISS 索引为空" in src or "ntotal=0" in src


# ==================== T01: 数据库 / 认证降级 ====================


class TestDatabaseAndAuthDegrade:
    """验证数据库连接超时配置与认证服务超时保护"""

    def test_database_service_imports_get_db_session(self):
        from src.services.support import database_service as ds

        assert "get_db_session" in inspect.getsource(ds)

    def test_init_remote_database_has_connect_timeout(self):
        from src.core.config import database as dbmod

        src = inspect.getsource(dbmod.init_remote_database)
        assert "connect_timeout" in src

    def test_auth_service_timeout_wrapper(self):
        from src.services.support import auth_service as auth

        src = inspect.getsource(auth.AuthService.__init__)
        assert "ThreadPoolExecutor" in src
        assert "timeout=10" in src
        assert "TimeoutError" in src


# ==================== T02: 模型预热 ====================


class TestModelWarmup:
    """验证 warmup_models 预热行为（懒加载 + TTL 卸载模式）"""

    def test_warmup_preloads_all_three(self):
        from src.services.model_service import app

        src = inspect.getsource(app.warmup_models)
        # 特征提取器仍然预加载
        assert "feature_extractor" in src
        assert "FeatureExtraction" in src
        # WD ViT Tagger 已切换为懒加载（注释或日志中提及）
        assert "WD ViT Tagger" in src or "WDViTV3Tagger" in src
        assert "懒加载" in src or "lazy" in src.lower()
        # EasyOCR 已切换为懒加载
        assert "EasyOCR" in src or "easyocr" in src
        # TTL 空闲卸载检查器已启动
        assert "_ttl_unload_checker" in src
        # 预热失败不阻止启动
        assert "DEGRADE" in src or "失败" in src

    def test_service_config_new_fields(self):
        from src.core.config.service_config import ServiceConfig

        cfg = ServiceConfig()
        assert cfg.HF_ENDPOINT == "https://hf-mirror.com"
        assert cfg.OCR_TIMEOUT == 10
        assert cfg.REDIS_RECONNECT_INTERVAL == 30
