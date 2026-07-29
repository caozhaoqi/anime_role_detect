"""应用版本端点（参考 K8s 文档的版本对齐实践）"""
from fastapi import APIRouter

from src.core.version import get_app_version_info

router = APIRouter()


@router.get("/api/version")
async def version():
    """返回应用版本、构建时间、git commit 与当前模型版本"""
    return get_app_version_info()
