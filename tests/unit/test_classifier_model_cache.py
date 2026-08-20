#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3.2 分类器（模型实例）缓存 TTL + LRU 验证测试。

验证 src.core.cache.model_cache.ModelCache 作为 model_loader._model_cache 时的：
1. 同一 key 命中返回同一实例（不重复初始化）
2. TTL 过期后 get 返回 None（触发重新加载）
3. 超过 max_size 时 LRU 淘汰最久未用条目
4. 向后兼容接口：bool / len / clear（供 classification.py `if not _model_cache`、
   health.py `bool(_model_cache)` 使用）

运行：
    .venv/bin/python3 tests/unit/test_classifier_model_cache.py
    .venv/bin/python3 -m pytest tests/unit/test_classifier_model_cache.py -v
"""
import sys
import time
from pathlib import Path

import pytest

# 与项目运行方式一致：src 作为包根
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.cache.model_cache import ModelCache


def _fake_model(name: str):
    """用简单对象模拟模型实例，便于身份比较。"""

    class _Fake:
        pass

    m = _Fake()
    m.name = name
    return m


class TestModelCacheTTLAndLRU:
    def test_same_key_returns_same_instance(self):
        cache = ModelCache(max_size=5, ttl_seconds=600)
        model = _fake_model("efficientnet_b0")
        c2i = {"a": 0, "b": 1}
        cache.set("efficientnet_b0", (model, c2i))

        got = cache.get("efficientnet_b0")
        assert got is not None
        assert got[0] is model  # 同一实例（不重复初始化）
        assert got[1] == c2i  # 同一 class_to_idx

    def test_expired_entry_returns_none(self):
        cache = ModelCache(max_size=5, ttl_seconds=600)
        model = _fake_model("resnet18")
        cache.set("resnet18", (model, {}))

        # 模拟过期：把写入时间戳改到很久以前
        cache.timestamps["resnet18"] = time.time() - 10_000

        assert cache.get("resnet18") is None  # 过期视为未命中，触发重载
        assert not cache.contains("resnet18")  # 已被惰性删除

    def test_lru_eviction(self):
        cache = ModelCache(max_size=2, ttl_seconds=600)
        a = _fake_model("a")
        b = _fake_model("b")
        c = _fake_model("c")

        cache.set("a", (a, {}))
        cache.set("b", (b, {}))
        cache.get("a")  # 访问 a，使其成为「最近使用」
        cache.set("c", (c, {}))  # 超过 max_size，应淘汰最久未用的 b

        assert cache.contains("a")
        assert cache.contains("c")
        assert not cache.contains("b")  # 被 LRU 淘汰
        assert cache.size() == 2

    def test_clear_and_bool_len(self):
        cache = ModelCache(max_size=5, ttl_seconds=600)
        cache.set("m", (_fake_model("m"), {}))
        assert len(cache) == 1
        assert cache  # __bool__ 为真

        cache.clear()
        assert len(cache) == 0
        assert not cache  # __bool__ 为假（供 `if not _model_cache` 使用）


class TestGlobalModelCacheCompat:
    def test_model_loader_global_is_model_cache(self):
        # 验证 model_loader._model_cache 已升级为 ModelCache 且行为兼容路由层
        from src.services.processor.model_loader import _model_cache

        assert isinstance(_model_cache, ModelCache)

        _model_cache.clear()
        assert not _model_cache  # classification.py: `if not _model_cache`

        _model_cache.set("compat_model", (_fake_model("compat_model"), {"x": 0}))
        assert _model_cache  # health.py: `bool(_model_cache)`

        got = _model_cache.get("compat_model")
        assert got[0].name == "compat_model"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
