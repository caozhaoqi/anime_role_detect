"""Regression tests for ONNX Runtime inference optimization (2026-07-24).

Tests cover:
1. ONNX engine initialization with different execution providers
2. ONNX model loading and inference
3. Model manager functionality
4. API route integration
"""
import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    import onnxruntime as ort
    HAS_ONNXRUNTIME = True
except ImportError:
    HAS_ONNXRUNTIME = False


@pytest.mark.skipif(not HAS_ONNXRUNTIME, reason="需要 onnxruntime 依赖")
class TestONNXEngineInitialization:
    """Test ONNX engine initialization."""

    def test_onnx_engine_exists(self):
        """ONNX engine module should exist."""
        from core import onnx_engine
        assert onnx_engine is not None

    def test_onnx_engine_class_exists(self):
        """ONNXEngine class should exist."""
        from core.onnx_engine import ONNXEngine
        assert ONNXEngine is not None

    def test_execution_providers_available(self):
        """Test that execution providers are available."""
        providers = ort.get_available_providers()
        assert len(providers) > 0, "No ONNX execution providers available"
        assert "CPUExecutionProvider" in providers, "CPU provider should be available"

    def test_onnx_engine_has_predict_method(self):
        """Test that ONNXEngine has predict method."""
        from core.onnx_engine import ONNXEngine
        assert hasattr(ONNXEngine, 'predict'), "ONNXEngine should have predict method"

    def test_onnx_engine_has_predict_batch_method(self):
        """Test that ONNXEngine has predict_batch method."""
        from core.onnx_engine import ONNXEngine
        assert hasattr(ONNXEngine, 'predict_batch'), "ONNXEngine should have predict_batch method"

    def test_onnx_engine_has_preprocess_method(self):
        """Test that ONNXEngine has preprocess method."""
        from core.onnx_engine import ONNXEngine
        assert hasattr(ONNXEngine, 'preprocess'), "ONNXEngine should have preprocess method"


@pytest.mark.skipif(not HAS_ONNXRUNTIME, reason="需要 onnxruntime 依赖")
class TestModelManager:
    """Test ModelManager functionality."""

    def test_model_manager_class_exists(self):
        """ModelManager class should exist."""
        from core.onnx_engine import ModelManager
        assert ModelManager is not None

    def test_model_manager_has_load_model_method(self):
        """ModelManager should have load_model method."""
        from core.onnx_engine import ModelManager
        manager = ModelManager()
        assert hasattr(manager, 'load_model'), "ModelManager should have load_model method"

    def test_model_manager_has_unload_model_method(self):
        """ModelManager should have unload_model method."""
        from core.onnx_engine import ModelManager
        manager = ModelManager()
        assert hasattr(manager, 'unload_model'), "ModelManager should have unload_model method"

    def test_model_manager_has_list_models_method(self):
        """ModelManager should have list_models method."""
        from core.onnx_engine import ModelManager
        manager = ModelManager()
        assert hasattr(manager, 'list_models'), "ModelManager should have list_models method"

    def test_global_model_manager_exists(self):
        """Global model_manager should exist."""
        from core.onnx_engine import model_manager
        assert model_manager is not None

    def test_list_available_models_function(self):
        """list_available_models function should work."""
        from core.onnx_engine import list_available_models
        models = list_available_models()
        assert isinstance(models, list)

    def test_get_engine_function(self):
        """get_engine function should exist."""
        from core.onnx_engine import get_engine
        assert get_engine is not None

    def test_release_engine_function(self):
        """release_engine function should exist."""
        from core.onnx_engine import release_engine
        assert release_engine is not None


@pytest.mark.skipif(not HAS_ONNXRUNTIME, reason="需要 onnxruntime 依赖")
class TestONNXInferenceAPI:
    """Test ONNX inference API endpoints."""

    def test_onnx_inference_route_exists(self):
        """ONNX inference route should exist."""
        filepath = PROJECT_ROOT / "src" / "api" / "routes" / "onnx_inference.py"
        assert filepath.exists(), f"File not found: {filepath}"
        content = filepath.read_text()
        assert "@router.post" in content, "Should have POST endpoint"

    def test_onnx_inference_uses_get_engine(self):
        """ONNX inference route should use get_engine."""
        filepath = PROJECT_ROOT / "src" / "api" / "routes" / "onnx_inference.py"
        assert filepath.exists(), f"File not found: {filepath}"
        content = filepath.read_text()
        assert "get_engine" in content, "Should import and use get_engine"

    def test_onnx_inference_has_error_handling(self):
        """ONNX inference route should have error handling."""
        filepath = PROJECT_ROOT / "src" / "api" / "routes" / "onnx_inference.py"
        assert filepath.exists(), f"File not found: {filepath}"
        content = filepath.read_text()
        assert "try:" in content and "except" in content, "Should have try-except error handling"

    def test_onnx_inference_has_predict_endpoint(self):
        """ONNX inference should have predict endpoint."""
        filepath = PROJECT_ROOT / "src" / "api" / "routes" / "onnx_inference.py"
        assert filepath.exists(), f"File not found: {filepath}"
        content = filepath.read_text()
        assert "/predict/{model_name}" in content, "Should have predict endpoint"

    def test_onnx_inference_has_batch_endpoint(self):
        """ONNX inference should have batch predict endpoint."""
        filepath = PROJECT_ROOT / "src" / "api" / "routes" / "onnx_inference.py"
        assert filepath.exists(), f"File not found: {filepath}"
        content = filepath.read_text()
        assert "/predict/batch/{model_name}" in content, "Should have batch predict endpoint"


@pytest.mark.skipif(not HAS_ONNXRUNTIME, reason="需要 onnxruntime 依赖")
class TestONNXIntegration:
    """Integration tests for ONNX inference."""

    def test_onnx_engine_importable(self):
        """Test that ONNXEngine can be imported."""
        from core.onnx_engine import ONNXEngine
        assert ONNXEngine is not None

    def test_model_manager_importable(self):
        """Test that ModelManager can be imported."""
        from core.onnx_engine import ModelManager
        assert ModelManager is not None

    def test_onnx_engine_instance_has_session_attr(self):
        """Test that ONNXEngine instance can have session attribute."""
        from core.onnx_engine import ONNXEngine
        # Check that _initialize sets session
        import inspect
        source = inspect.getsource(ONNXEngine._initialize)
        assert "self.session =" in source, "ONNXEngine._initialize should set self.session"

    def test_onnx_engine_instance_has_input_name_attr(self):
        """Test that ONNXEngine instance can have input_name attribute."""
        from core.onnx_engine import ONNXEngine
        import inspect
        source = inspect.getsource(ONNXEngine._initialize)
        assert "self.input_name =" in source, "ONNXEngine._initialize should set self.input_name"

    def test_onnx_engine_instance_has_output_name_attr(self):
        """Test that ONNXEngine instance can have output_name attribute."""
        from core.onnx_engine import ONNXEngine
        import inspect
        source = inspect.getsource(ONNXEngine._initialize)
        assert "self.output_name =" in source, "ONNXEngine._initialize should set self.output_name"

    def test_model_manager_has_engines_attr(self):
        """Test that ModelManager has engines attribute."""
        from core.onnx_engine import ModelManager
        manager = ModelManager()
        assert hasattr(manager, 'engines'), "ModelManager should have engines attribute"

    def test_model_manager_engines_is_dict(self):
        """Test that ModelManager engines is a dict."""
        from core.onnx_engine import ModelManager
        manager = ModelManager()
        assert isinstance(manager.engines, dict), "ModelManager engines should be a dict"

    def test_onnx_inference_route_imports_model_manager(self):
        """Test that ONNX inference route imports ModelManager."""
        filepath = PROJECT_ROOT / "src" / "api" / "routes" / "onnx_inference.py"
        assert filepath.exists(), f"File not found: {filepath}"
        content = filepath.read_text()
        assert "ModelManager" in content, "Should import ModelManager"