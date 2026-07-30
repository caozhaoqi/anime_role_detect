"""
pytest 共享 fixtures 配置
为所有测试提供通用 fixtures，减少重复代码。
"""
import os
import sys
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 检测 ML 依赖
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

try:
    import easyocr
    HAS_EASYOCR = True
except ImportError:
    HAS_EASYOCR = False

# ============================================================
# 通用 fixtures
# ============================================================

@pytest.fixture(scope="session")
def project_root():
    """项目根目录路径"""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def test_data_dir(project_root):
    """测试数据目录"""
    return project_root / "data"


@pytest.fixture(scope="function")
def mock_logger():
    """Mock 日志记录器，避免测试时产生真实日志"""
    with patch("src.core.logging.global_logger.get_logger") as mock_get_logger:
        mock = MagicMock()
        mock_get_logger.return_value = mock
        yield mock


@pytest.fixture(scope="function")
def mock_redis():
    """Mock Redis 连接"""
    with patch("redis.Redis") as mock:
        mock_instance = MagicMock()
        mock.return_value = mock_instance
        yield mock_instance


@pytest.fixture(scope="function")
def mock_db_session():
    """Mock 数据库 session"""
    with patch("src.core.config.database.get_db") as mock_get_db:
        mock_session = MagicMock()
        mock_get_db.return_value = iter([mock_session])
        yield mock_session


@pytest.fixture(scope="function")
def mock_httpx():
    """Mock httpx 异步客户端"""
    with patch("httpx.AsyncClient") as mock:
        mock_instance = MagicMock()
        mock.return_value.__aenter__.return_value = mock_instance
        yield mock_instance


# ============================================================
# ML 相关 fixtures
# ============================================================

@pytest.fixture(scope="session")
def data_dir(project_root):
    """数据目录（用于模型测试）"""
    data_path = project_root / "data"
    if not data_path.exists():
        data_path.mkdir(parents=True, exist_ok=True)
    return data_path


@pytest.fixture(scope="session")
def model_path(project_root):
    """模型路径（返回第一个可用的模型文件）"""
    model_dir = project_root / "models"
    if model_dir.exists():
        for ext in ["*.pth", "*.pt"]:
            for model_file in model_dir.glob(ext):
                return model_file
    return None


@pytest.fixture(scope="session")
def mlmodel_path(project_root):
    """Core ML 模型路径"""
    model_dir = project_root / "models"
    if model_dir.exists():
        for model_file in model_dir.glob("*.mlpackage"):
            return model_file
    return None


@pytest.fixture(scope="session")
def onnx_path(project_root):
    """ONNX 模型路径"""
    model_dir = project_root / "models"
    if model_dir.exists():
        for model_file in model_dir.glob("*.onnx"):
            return model_file
    return None


@pytest.fixture(scope="session")
def test_images(tmp_path):
    """生成测试图像数据"""
    from PIL import Image
    test_images_list = []
    for i in range(3):
        img_path = tmp_path / f"test_{i}.jpg"
        img = Image.new('RGB', (224, 224), color=(128 + i * 30, 128, 128))
        img.save(img_path)
        test_images_list.append(str(img_path))
    return test_images_list


# ============================================================
# ML 依赖自动跳过逻辑
# ============================================================

def pytest_collection_modifyitems(config, items):
    """根据 ML 依赖自动跳过测试"""
    skip_torch = pytest.mark.skip(reason="需要 torch 依赖 (~800MB)")
    skip_faiss = pytest.mark.skip(reason="需要 faiss 依赖")
    skip_easyocr = pytest.mark.skip(reason="需要 easyocr 依赖 (~200MB)")

    for item in items:
        # 检查测试文件路径
        file_path = str(item.fspath)
        
        # torch 相关测试
        if "test_model_performance" in file_path or \
           "test_all_models" in file_path or \
           "test_model_accuracy" in file_path or \
           "test_deduplication" in file_path or \
           "test_classification_direct" in file_path or \
           "test_individual_models" in file_path or \
           "test_model_fix" in file_path or \
           "test_model_on_collected_data" in file_path or \
           "test_processing" in file_path or \
           "test_single_model" in file_path or \
           "test_weight_loading" in file_path or \
           "test_wuthering_waves" in file_path or \
           "test_all_inference_modes" in file_path or \
           "test_coreml_performance" in file_path or \
           "test_model_management" in file_path or \
           "test_infinity_fix" in file_path or \
           "test_classification_debug" in file_path or \
           "test_midterm_optimization" in file_path or \
           "test_extended_classifier" in file_path or \
           "test_yolo_detector" in file_path:
            if not HAS_TORCH:
                item.add_marker(skip_torch)
        
        # faiss 相关测试
        if "test_cleaning_pipeline" in file_path or \
           "test_pipeline_full" in file_path or \
           "test_pipeline_integration" in file_path or \
           "test_with_real_data" in file_path or \
           "test_recognition_system" in file_path or \
           "test_classification.py" in file_path:
            if not HAS_FAISS:
                item.add_marker(skip_faiss)
        
        # easyocr 相关测试
        if "test_easyocr" in file_path.lower():
            if not HAS_EASYOCR:
                item.add_marker(skip_easyocr)
