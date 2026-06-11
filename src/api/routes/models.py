"""
模型版本管理和多模型路由
"""
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, project_root)

from fastapi import APIRouter, Form, Depends
from typing import Optional

from src.core.logging.global_logger import get_logger
from src.middleware.auth_enhanced import get_current_admin
from src.services.model.model_version_service import (
    get_model_versions,
    get_model_path,
    register_model,
    enable_ab_test,
    disable_ab_test,
    get_ab_test_config,
    select_model_for_ab_test,
    update_model_description,
    delete_model_version,
)
from src.services.model.multi_model_service import (
    add_model,
    remove_model,
    set_fusion_strategy,
    get_multi_model_service,
)

logger = get_logger("api.routes.models")

router = APIRouter()


# ===== 模型列表 =====

@router.get("/api/models")
async def get_available_models():
    """获取可用的模型列表"""
    models_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "models")
    models_path = os.path.normpath(models_path)
    logger.info(f"查找模型路径: {models_path}")

    model_dirs = []
    if os.path.exists(models_path):
        model_dirs = [d for d in os.listdir(models_path) if os.path.isdir(os.path.join(models_path, d))]
        logger.info(f"找到模型目录: {model_dirs}")
    else:
        logger.warning(f"模型目录不存在: {models_path}")

    return {"success": True, "models": model_dirs, "default_model": "default"}


# ===== 模型版本管理 =====

@router.get("/api/model-versions")
async def get_model_versions_list(model_name: Optional[str] = None):
    """获取模型版本列表"""
    try:
        versions = get_model_versions(model_name)
        return {"success": True, "message": "获取模型版本成功", "data": versions}
    except Exception as e:
        logger.error(f"获取模型版本失败: {e}")
        return {"success": False, "message": "获取模型版本失败，请稍后重试"}


@router.post("/api/model-versions/register")
async def register_model_version(
    model_name: str = Form(...),
    version: str = Form(...),
    path: str = Form(...),
    description: str = Form(""),
    current_admin: dict = Depends(get_current_admin),
):
    """注册模型版本"""
    try:
        success = register_model(model_name, version, path, description)
        return {"success": True, "message": "模型版本注册成功"} if success else {"success": False, "message": "模型版本注册失败"}
    except Exception as e:
        logger.error(f"注册模型版本失败: {e}")
        return {"success": False, "message": "注册模型版本失败，请稍后重试"}


@router.get("/api/model-versions/path")
async def get_model_version_path(
    model_name: str = Form(...),
    version: str = Form("latest"),
):
    """获取模型路径"""
    try:
        path = get_model_path(model_name, version)
        if path:
            return {"success": True, "message": "获取模型路径成功", "data": {"path": path}}
        return {"success": False, "message": "模型路径不存在"}
    except Exception as e:
        logger.error(f"获取模型路径失败: {e}")
        return {"success": False, "message": "获取模型路径失败，请稍后重试"}


# ===== A/B测试 =====

@router.post("/api/model-versions/ab-test/enable")
async def enable_ab_test_endpoint(
    test_models: str = Form(...),
    weights: str = Form(...),
    control_model: str = Form(...),
    current_admin: dict = Depends(get_current_admin),
):
    """启用A/B测试"""
    try:
        test_models_list = []
        for model_str in test_models.split(","):
            if ":" in model_str:
                model_name, version = model_str.split(":")
                test_models_list.append((model_name.strip(), version.strip()))
        weights_list = [float(w.strip()) for w in weights.split(",")]
        enable_ab_test(test_models_list, weights_list, control_model)
        return {"success": True, "message": "A/B测试启用成功"}
    except Exception as e:
        logger.error(f"启用A/B测试失败: {e}")
        return {"success": False, "message": "启用A/B测试失败，请稍后重试"}


@router.post("/api/model-versions/ab-test/disable")
async def disable_ab_test_endpoint(current_admin: dict = Depends(get_current_admin)):
    """禁用A/B测试"""
    try:
        disable_ab_test()
        return {"success": True, "message": "A/B测试禁用成功"}
    except Exception as e:
        logger.error(f"禁用A/B测试失败: {e}")
        return {"success": False, "message": "禁用A/B测试失败，请稍后重试"}


@router.get("/api/model-versions/ab-test/config")
async def get_ab_test_config_endpoint():
    """获取A/B测试配置"""
    try:
        config = get_ab_test_config()
        return {"success": True, "message": "获取A/B测试配置成功", "data": config}
    except Exception as e:
        logger.error(f"获取A/B测试配置失败: {e}")
        return {"success": False, "message": "获取A/B测试配置失败，请稍后重试"}


@router.get("/api/model-versions/ab-test/select")
async def select_model_for_ab_test_endpoint():
    """为A/B测试选择模型"""
    try:
        model_name, version = select_model_for_ab_test()
        return {
            "success": True,
            "message": "选择模型成功",
            "data": {"model_name": model_name, "version": version},
        }
    except Exception as e:
        logger.error(f"选择模型失败: {e}")
        return {"success": False, "message": "选择模型失败，请稍后重试"}


@router.post("/api/model-versions/update-description")
async def update_model_description_endpoint(
    model_name: str = Form(...),
    version: str = Form(...),
    description: str = Form(...),
    current_admin: dict = Depends(get_current_admin),
):
    """更新模型描述"""
    try:
        success = update_model_description(model_name, version, description)
        return {"success": True, "message": "模型描述更新成功"} if success else {"success": False, "message": "模型描述更新失败"}
    except Exception as e:
        logger.error(f"更新模型描述失败: {e}")
        return {"success": False, "message": "更新模型描述失败，请稍后重试"}


@router.post("/api/model-versions/delete")
async def delete_model_version_endpoint(
    model_name: str = Form(...),
    version: str = Form(...),
    current_admin: dict = Depends(get_current_admin),
):
    """删除模型版本"""
    try:
        success = delete_model_version(model_name, version)
        return {"success": True, "message": "模型版本删除成功"} if success else {"success": False, "message": "模型版本删除失败"}
    except Exception as e:
        logger.error(f"删除模型版本失败: {e}")
        return {"success": False, "message": "删除模型版本失败，请稍后重试"}


# ===== 多模型集成 =====

@router.get("/api/multi-model/config")
async def get_multi_model_config():
    """获取多模型配置"""
    try:
        service = get_multi_model_service()
        configs = service.get_model_configs()
        strategy = service.get_fusion_strategy()
        return {
            "success": True,
            "message": "获取多模型配置成功",
            "data": {"models": configs, "fusion_strategy": strategy},
        }
    except Exception as e:
        logger.error(f"获取多模型配置失败: {e}")
        return {"success": False, "message": "获取多模型配置失败，请稍后重试"}


@router.post("/api/multi-model/add")
async def add_multi_model(
    model_name: str = Form(...),
    model_type: str = Form(...),
    weight: float = Form(...),
    current_admin: dict = Depends(get_current_admin),
):
    """添加模型到多模型集成"""
    try:
        add_model(model_name, model_type, weight)
        return {"success": True, "message": "模型添加成功"}
    except Exception as e:
        logger.error(f"添加模型失败: {e}")
        return {"success": False, "message": "添加模型失败，请稍后重试"}


@router.post("/api/multi-model/remove")
async def remove_multi_model(
    model_name: str = Form(...),
    current_admin: dict = Depends(get_current_admin),
):
    """从多模型集成中移除模型"""
    try:
        remove_model(model_name)
        return {"success": True, "message": "模型移除成功"}
    except Exception as e:
        logger.error(f"移除模型失败: {e}")
        return {"success": False, "message": "移除模型失败，请稍后重试"}


@router.post("/api/multi-model/strategy")
async def set_multi_model_strategy(
    strategy: str = Form(...),
    current_admin: dict = Depends(get_current_admin),
):
    """设置多模型融合策略"""
    try:
        set_fusion_strategy(strategy)
        return {"success": True, "message": "融合策略设置成功"}
    except Exception as e:
        logger.error(f"设置融合策略失败: {e}")
        return {"success": False, "message": "设置融合策略失败，请稍后重试"}