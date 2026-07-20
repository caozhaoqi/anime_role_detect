#!/usr/bin/env python3
"""
测试类别映射
"""
import json
import os
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

# 加载类别映射
mapping_path = os.path.join(PROJECT_ROOT, "models", "efficientnet_b3", "character_classifier_best_improved_class_mapping.json")


@pytest.fixture(scope="module")
def idx_to_class():
    """加载类别映射，若文件不存在则跳过所有测试"""
    if not os.path.exists(mapping_path):
        pytest.skip(f"类别映射文件不存在: {mapping_path}")
    with open(mapping_path, "r") as f:
        mapping = json.load(f)
    return mapping["idx_to_class"]


def test_mapping_loaded(idx_to_class):
    """验证类别映射已加载"""
    assert idx_to_class is not None
    assert len(idx_to_class) > 0
    print(f"类别映射加载成功，包含 {len(idx_to_class)} 个类别")


@pytest.mark.parametrize("idx", [127, 0, 73, 130])
def test_index_lookup(idx_to_class, idx):
    """测试不同索引的查找"""
    # 字符串查找
    str_value = idx_to_class.get(str(idx))
    assert str_value is not None, f"索引 {idx}（字符串查找）无结果"
    assert isinstance(str_value, str), f"索引 {idx} 结果不是字符串: {type(str_value)}"


def test_get_role_name_function(idx_to_class):
    """测试角色名获取函数"""

    def get_role_name(predicted_idx, mapping):
        if mapping:
            if str(predicted_idx) in mapping:
                return mapping[str(predicted_idx)]
            else:
                return f"类别_{predicted_idx}"
        return f"类别_{predicted_idx}"

    test_indices = [127, 0, 73, 130]
    for idx in test_indices:
        role = get_role_name(idx, idx_to_class)
        assert role is not None
        assert isinstance(role, str)
