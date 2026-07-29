"""应用版本与构建元数据。

参考 K8s 文档的「版本端点 / 镜像 tag 对齐」实践：
- APP_VERSION 为单一真相源（与 pyproject.toml 保持一致）
- 构建期由 deployment/Dockerfile.* 生成 _build_info.py 写入 BUILD_TIME / GIT_COMMIT
- 本地开发无 _build_info.py 时回退到实时 git 估算
"""
import os
import subprocess

APP_VERSION = "2.3.0"


def _git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")),
        )
        return out.decode().strip() or "unknown"
    except Exception:
        return "unknown"


def _build_info() -> dict:
    try:
        from src.core.version._build_info import BUILD_TIME, GIT_COMMIT  # type: ignore

        return {"build_time": BUILD_TIME, "git_commit": GIT_COMMIT}
    except Exception:
        return {"build_time": "dev", "git_commit": _git_commit()}


def get_app_version_info() -> dict:
    """聚合应用版本信息，供 /api/version 端点返回。"""
    info = _build_info()
    model_version = "unknown"
    try:
        from src.core.version.model_version_manager import ModelVersionManager

        latest = ModelVersionManager().get_latest_version()
        if latest:
            model_version = latest.get("name") or "unknown"
    except Exception:
        pass
    return {
        "app_version": APP_VERSION,
        "build_time": info["build_time"],
        "git_commit": info["git_commit"],
        "model_version": model_version,
    }
