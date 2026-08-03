#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_k8s_ports.py — 防御性 K8s 端口一致性校验 (OPTIMIZATION_PLAN 1.2)

目的
----
扫描 ``k8s/`` 目录下所有 ``.yaml``（递归 base + overlays），校验每个
Deployment/StatefulSet 的 ``containerPort`` 与每个 Service 的 ``targetPort``
（未显式设置时回退到 ``port``）是否等于"代码真实运行时端口"。

历史背景：曾发生 4 处端口映射错误（multimedia 8002→8000、search 8003→8000 等），
导致探针 connection refused / 服务不可达。这些错误当前已被修复，本脚本的价值是
**防御性 CI 校验**——防止未来再次引入不一致。

权威端口映射来源（每项均附代码文件:行号，已逐项核实）
-------------------------------------------------------
  api-gateway        8080  src/services/api_gateway/app.py:716            (argparse --port default=8080)
  api-service        8001  src/api/app.py:78                             (uvicorn.run port=8001)
  model-service      8000  src/core/config/service_config.py:20         (MODEL_SERVICE_PORT=8000;
                                                                          model_service/app.py:273 default=config.MODEL_SERVICE_PORT)
  multimedia-service 8002  src/services/multimedia_service/multimedia_service_app.py:61 (--port default=8002)
  search-service     8003  src/services/search_service/app_queue.py:223 (uvicorn.run port=8003)
  monitoring         9000  src/run/monitor/monitor_dashboard.py:38     (MONITOR_PORT=9000)
  frontend           3000  Next.js dev/build 默认端口
  mysql              3306  基础设施
  redis              6379  基础设施
  rabbitmq           5672/15672 基础设施

重要说明
--------
* K8s 中监控资源名为 ``monitoring``（容器 args 用 ``monitor-dashboard``，但
  metadata.name / Service 名均为 ``monitoring``），故映射键为 ``"monitoring"``。
* ``video-service`` 当前**不在** K8s base 部署（仅存在于代码
  src/services/video_service/video_service_app.py:242 port=8003），故不纳入必须校验的
  ``K8S_PORT_MAP``，仅列入 ``CODE_ONLY_SERVICES`` 供参考，避免误杀未来新增部署。
* 端口本质稳定不易变，脚本零重型依赖（不 import torch 等），仅依赖 PyYAML。
* 设计原则：
  - 正向校验：映射中每个服务，若 K8s 中存在对应资源，其端口必须等于映射值；否则 ERROR 并 exit 1。
  - 反向校验：K8s 中出现的端口型服务若不在映射里 → 仅 WARNING（不 FAIL），避免误杀未来新增服务。
  - 缺失的映射项（K8s 中找不到对应资源）→ WARNING，提醒维护者同步 ``K8S_PORT_MAP``。

退出码
------
  0  全部通过（含 WARNING 不阻断）
  1  发现端口不一致 / YAML 解析失败 / 必要资源缺失
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover - 环境校验
    sys.stderr.write(
        "ERROR: PyYAML 未安装。请在 CI 中执行 `pip install pyyaml` 或使用项目 .venv。\n"
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# 权威端口映射：K8s 资源名 (metadata.name) -> 期望端口 (int 或 [int, ...])
# 仅包含当前已在 K8s 中部署的服务（base + overlays）。
# ---------------------------------------------------------------------------
K8S_PORT_MAP = {
    "api-gateway": 8080,
    "api-service": 8001,
    "model-service": 8000,
    "multimedia-service": 8002,
    "search-service": 8003,
    "monitoring": 9000,
    "frontend": 3000,
    "mysql": 3306,
    "redis": 6379,
    "rabbitmq": [5672, 15672],
}

# 仅存在于代码、尚未部署到 K8s 的服务（反向校验时跳过，避免误报）。
CODE_ONLY_SERVICES = {
    "video-service": 8003,
}

# 需要在 K8s 中作为资源存在的 Kind（其余 Kind 如 Kustomization/ConfigMap 直接跳过）。
WORKLOAD_KINDS = {"Deployment", "StatefulSet"}
SERVICE_KINDS = {"Service"}


def as_set(value):
    """将映射值规范为端口集合。"""
    if isinstance(value, (list, tuple, set)):
        return set(int(v) for v in value)
    return {int(value)}


def coerce_port(value):
    """将 YAML 端口值规范为 int；命名端口保持 str；无法解析返回原值。"""
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        s = value.strip()
        if s.isdigit():
            return int(s)
        return s  # 命名端口（如 "http"），无法数值校验
    return value


def collect_resources(k8s_dir: Path):
    """递归扫描 k8s/ 下所有 .yaml，收集 workloads 与 services 的端口。

    返回:
        deployments: dict[name] -> set(containerPorts)
        services:    dict[name] -> list[{port, targetPort}]   (targetPort 未设时为 None，表示回退 port)
    """
    deployments: dict[str, set] = {}
    services: dict[str, list] = {}

    yaml_files = sorted(k8s_dir.rglob("*.yaml"))
    if not yaml_files:
        raise FileNotFoundError(f"未在 {k8s_dir} 下找到任何 .yaml 文件")

    for fpath in yaml_files:
        try:
            with open(fpath, "r", encoding="utf-8") as fh:
                docs = list(yaml.safe_load_all(fh))
        except yaml.YAMLError as exc:
            raise RuntimeError(f"YAML 解析失败: {fpath}\n  {exc}") from exc

        for doc in docs:
            if not isinstance(doc, dict):
                continue
            kind = doc.get("kind")
            meta = doc.get("metadata") or {}
            name = meta.get("name")
            if not name:
                continue

            if kind in WORKLOAD_KINDS:
                ports: set = set()
                spec = doc.get("spec") or {}
                template = spec.get("template") or {}
                pod_spec = template.get("spec") or {}
                containers = pod_spec.get("containers") or []
                for c in containers:
                    for p in (c.get("ports") or []):
                        cp = coerce_port(p.get("containerPort"))
                        if isinstance(cp, int):
                            ports.add(cp)
                deployments[name] = ports

            elif kind in SERVICE_KINDS:
                spec = doc.get("spec") or {}
                entries = []
                for p in (spec.get("ports") or []):
                    port = coerce_port(p.get("port"))
                    target = p.get("targetPort")
                    if target is None:
                        target_port = None  # 回退到 port
                    else:
                        target_port = coerce_port(target)
                    entries.append({"port": port, "targetPort": target_port})
                services[name] = entries

    return deployments, services


def port_str(port_set) -> str:
    return ",".join(str(p) for p in sorted(port_set))


def run(k8s_dir: Path) -> int:
    deployments, services = collect_resources(k8s_dir)

    errors: list[str] = []
    warnings: list[str] = []
    ok_lines: list[str] = []

    # ---- 正向校验：映射中每个服务的端口一致性 ----
    for svc_name, expected in K8S_PORT_MAP.items():
        expected_set = as_set(expected)
        svc_ok = True

        # Deployment / StatefulSet containerPort 校验
        if svc_name in deployments:
            actual = deployments[svc_name]
            missing = expected_set - actual
            extra = actual - expected_set
            if missing:
                svc_ok = False
                errors.append(
                    f"ERROR: {svc_name} containerPort 缺少期望端口: "
                    f"expected={port_str(expected_set)} actual={port_str(actual)}"
                )
            if extra:
                svc_ok = False
                errors.append(
                    f"ERROR: {svc_name} containerPort 存在非期望端口: "
                    f"actual={port_str(actual)} expected={port_str(expected_set)}"
                )
        else:
            warnings.append(
                f"WARN: k8s/ 中未找到 deployment/statefulset '{svc_name}' "
                f"(期望端口 {port_str(expected_set)})；若服务已移除请同步更新 K8S_PORT_MAP"
            )

        # Service targetPort 校验
        if svc_name in services:
            for entry in services[svc_name]:
                tp = entry["targetPort"] if entry["targetPort"] is not None else entry["port"]
                if isinstance(tp, str):
                    warnings.append(
                        f"WARN: service '{svc_name}' 使用命名 targetPort '{tp}'，无法数值校验"
                    )
                    continue
                if isinstance(tp, int) and tp not in expected_set:
                    svc_ok = False
                    errors.append(
                        f"ERROR: {svc_name} service targetPort={tp} "
                        f"expected={port_str(expected_set)}"
                    )
        else:
            warnings.append(
                f"WARN: k8s/ 中未找到 service '{svc_name}' "
                f"(期望端口 {port_str(expected_set)})"
            )

        if svc_ok:
            ok_lines.append(f"OK: {svc_name} -> {port_str(expected_set)}")

    # ---- 反向校验：K8s 中出现但不在映射里的端口型资源 → 仅 WARNING ----
    known = set(K8S_PORT_MAP) | set(CODE_ONLY_SERVICES)
    for name, ports in deployments.items():
        if name in known:
            continue
        if ports:  # 仅对"带端口的"app 类资源告警，worker 无端口不告警
            warnings.append(
                f"WARN: deployment '{name}' 含端口 {port_str(ports)} 但不在 K8S_PORT_MAP，请人工核对"
            )
    for name, entries in services.items():
        if name in known:
            continue
        warnings.append(
            f"WARN: service '{name}' 不在 K8S_PORT_MAP，请人工核对端口"
        )

    # ---- 输出 ----
    for line in ok_lines:
        print(line)
    for line in warnings:
        print(line)
    for line in errors:
        print(line)

    if errors:
        print(
            f"\n发现 {len(errors)} 处端口不一致，K8s 端口校验失败。",
            file=sys.stderr,
        )
        return 1

    print("\nAll K8s port checks passed.")
    return 0


def main(argv=None) -> int:
    repo_root = Path(__file__).resolve().parent.parent.parent
    parser = argparse.ArgumentParser(
        description="防御性校验 K8s manifest 端口与代码真实端口一致"
    )
    parser.add_argument(
        "--k8s-dir",
        default=str(repo_root / "k8s"),
        help="k8s manifests 目录 (默认: <repo>/k8s)",
    )
    args = parser.parse_args(argv)

    k8s_dir = Path(args.k8s_dir)
    if not k8s_dir.is_dir():
        sys.stderr.write(f"ERROR: k8s 目录不存在: {k8s_dir}\n")
        return 1

    try:
        return run(k8s_dir)
    except (FileNotFoundError, RuntimeError) as exc:
        sys.stderr.write(f"ERROR: {exc}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
