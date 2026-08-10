#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Core 端口（Port）注册表。

core（领域 / 算法层）运行所需的基础设施能力（如模型加载、特征处理）
通过端口抽象与 services（基础设施层）解耦：

- core 在**导入期**只依赖本模块，不再 `import src.services.*`；
- 具体实现由 services 层在启动时通过 ``register_port`` 注入
  （见 ``src/api/lifecycle._init_services``）。

这样 core 可独立单元测试（注入 mock 实现），且不承担基础设施的导入成本。

可逆性：若某入口未装配端口即调用，``get_port`` 会抛出清晰错误，
提示缺失的装配步骤，而非静默失败。
"""

from __future__ import annotations

_PORTS: dict = {}


def register_port(name: str, impl) -> None:
    """注册一个端口实现。services 层启动时调用。

    Args:
        name: 端口名（如 ``"role_predictor"`` / ``"feature_processor"``）。
        impl: 具体实现（函数或可调用对象）。
    """
    _PORTS[name] = impl


def is_port_registered(name: str) -> bool:
    """端口是否已注入实现。"""
    return name in _PORTS


def get_port(name: str):
    """获取端口实现。

    Raises:
        RuntimeError: 端口未注入（装配缺失）时。
    """
    impl = _PORTS.get(name)
    if impl is None:
        raise RuntimeError(
            f"Core 端口 '{name}' 未注入实现。请在服务启动时调用 "
            f"register_port('{name}', impl)（参考 src/api/lifecycle._init_services）。"
        )
    return impl


def clear_ports() -> None:
    """清空所有端口（仅用于测试隔离）。"""
    _PORTS.clear()
