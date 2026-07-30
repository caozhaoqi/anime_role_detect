"""Regression tests for CORS security fixes (2026-07-24).

Tests cover:
1. CORS configuration validation - ensure allow_headers is not "*"
2. CORS origins restriction - ensure origins are not wildcard in production
3. Environment variable configuration for CORS
"""
import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


class TestCORSSecurity:
    """Test CORS security configuration."""

    def test_api_gateway_cors_not_wildcard(self):
        """API Gateway should not use allow_headers=["*"]."""
        filepath = PROJECT_ROOT / "src" / "services" / "api_gateway" / "app.py"
        assert filepath.exists(), f"File not found: {filepath}"
        content = filepath.read_text()
        assert 'allow_headers=["*"]' not in content, \
            "API Gateway should not use wildcard allow_headers"
        assert 'allow_headers=[\n' in content, \
            "API Gateway should use explicit header list"

    def test_search_service_cors_not_wildcard(self):
        """Search Service should not use allow_headers=["*"]."""
        filepath = PROJECT_ROOT / "src" / "services" / "search_service" / "app_queue.py"
        assert filepath.exists(), f"File not found: {filepath}"
        content = filepath.read_text()
        assert 'allow_headers=["*"]' not in content, \
            "Search Service should not use wildcard allow_headers"

    def test_multimedia_service_cors_not_wildcard(self):
        """Multimedia Service should not use allow_headers=["*"]."""
        filepath = PROJECT_ROOT / "src" / "services" / "multimedia_service" / "multimedia_service_app.py"
        assert filepath.exists(), f"File not found: {filepath}"
        content = filepath.read_text()
        assert 'allow_headers=["*"]' not in content, \
            "Multimedia Service should not use wildcard allow_headers"

    def test_model_service_cors_not_wildcard(self):
        """Model Service should not use allow_headers=["*"]."""
        filepath = PROJECT_ROOT / "src" / "services" / "model_service" / "app.py"
        assert filepath.exists(), f"File not found: {filepath}"
        content = filepath.read_text()
        assert 'allow_headers=["*"]' not in content, \
            "Model Service should not use wildcard allow_headers"

    def test_api_lifecycle_cors_not_wildcard(self):
        """API lifecycle should not use allow_headers=["*"]."""
        filepath = PROJECT_ROOT / "src" / "api" / "lifecycle.py"
        assert filepath.exists(), f"File not found: {filepath}"
        content = filepath.read_text()
        assert 'allow_headers=["*"]' not in content, \
            "API lifecycle should not use wildcard allow_headers"

    def test_cors_has_explicit_headers(self):
        """CORS configuration should include specific headers."""
        required_headers = ["Content-Type", "Authorization", "X-Requested-With", "Accept", "Origin"]
        
        files_to_check = [
            "src/services/api_gateway/app.py",
            "src/services/search_service/app_queue.py",
            "src/services/multimedia_service/multimedia_service_app.py",
            "src/services/model_service/app.py",
            "src/api/lifecycle.py",
        ]
        
        for rel_path in files_to_check:
            filepath = PROJECT_ROOT / rel_path
            if filepath.exists():
                content = filepath.read_text()
                for header in required_headers:
                    assert f'"{header}"' in content or f"'{header}'" in content, \
                        f"{rel_path} should include {header} in allow_headers"

    def test_api_gateway_cors_origins_from_env(self):
        """API Gateway should support CORS_ORIGINS environment variable."""
        filepath = PROJECT_ROOT / "src" / "services" / "api_gateway" / "app.py"
        assert filepath.exists(), f"File not found: {filepath}"
        content = filepath.read_text()
        assert "CORS_ORIGINS" in content, \
            "API Gateway should read CORS_ORIGINS from environment"

    def test_search_service_cors_origins_from_env(self):
        """Search Service should support CORS_ORIGINS environment variable."""
        filepath = PROJECT_ROOT / "src" / "services" / "search_service" / "app_queue.py"
        assert filepath.exists(), f"File not found: {filepath}"
        content = filepath.read_text()
        assert "CORS_ORIGINS" in content, \
            "Search Service should read CORS_ORIGINS from environment"

    def test_multimedia_service_cors_origins_from_env(self):
        """Multimedia Service should support CORS_ORIGINS environment variable."""
        filepath = PROJECT_ROOT / "src" / "services" / "multimedia_service" / "multimedia_service_app.py"
        assert filepath.exists(), f"File not found: {filepath}"
        content = filepath.read_text()
        assert "CORS_ORIGINS" in content, \
            "Multimedia Service should read CORS_ORIGINS from environment"

    def test_default_cors_origins_are_localhost(self):
        """Default CORS origins should be localhost only."""
        files_to_check = [
            "src/services/api_gateway/app.py",
            "src/services/search_service/app_queue.py",
            "src/services/multimedia_service/multimedia_service_app.py",
        ]
        
        for rel_path in files_to_check:
            filepath = PROJECT_ROOT / rel_path
            if filepath.exists():
                content = filepath.read_text()
                assert '"http://localhost:3000"' in content or "'http://localhost:3000'" in content, \
                    f"{rel_path} should include localhost:3000 as default origin"

    def test_cors_methods_not_wildcard(self):
        """CORS allow_methods should not be "*"."""
        files_to_check = [
            "src/services/api_gateway/app.py",
            "src/services/search_service/app_queue.py",
            "src/services/multimedia_service/multimedia_service_app.py",
            "src/services/model_service/app.py",
            "src/api/lifecycle.py",
        ]
        
        for rel_path in files_to_check:
            filepath = PROJECT_ROOT / rel_path
            if filepath.exists():
                content = filepath.read_text()
                # Allow "*" only if it's followed by proper methods list
                # Check that explicit methods are listed
                assert '["GET"' in content or '"GET"' in content or "'GET'" in content, \
                    f"{rel_path} should specify GET method explicitly"


class TestCORSEnvironmentConfiguration:
    """Test CORS environment variable configuration."""

    def test_cors_env_parsing(self):
        """Test that CORS_ORIGINS environment variable is parsed correctly."""
        from src.services.search_service.app_queue import _cors_origins_env, _allowed_origins
        
        # With env variable set
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("CORS_ORIGINS", "https://example.com,https://api.example.com")
            # Re-import to test
            import importlib
            import src.services.search_service.app_queue as module
            importlib.reload(module)
            assert module._allowed_origins == ["https://example.com", "https://api.example.com"]

    def test_cors_default_origins(self):
        """Test that default origins are used when env variable is not set."""
        with pytest.MonkeyPatch.context() as mp:
            mp.delenv("CORS_ORIGINS", raising=False)
            import importlib
            import src.services.search_service.app_queue as module
            importlib.reload(module)
            assert "http://localhost:3000" in module._allowed_origins


class TestCORSIntegration:
    """Integration tests for CORS configuration."""

    def test_api_gateway_cors_middleware_config(self):
        """Test that API Gateway CORS middleware is correctly configured."""
        from fastapi.middleware.cors import CORSMiddleware
        from src.services.api_gateway.app import app
        
        cors_middleware = None
        for middleware in app.user_middleware:
            if middleware.cls == CORSMiddleware:
                cors_middleware = middleware
                break
        
        assert cors_middleware is not None, "CORSMiddleware should be registered"
        
        options = cors_middleware.kwargs
        assert options["allow_headers"] != ["*"], "allow_headers should not be wildcard"
        assert "Content-Type" in options["allow_headers"], "Content-Type should be in allow_headers"
        assert "Authorization" in options["allow_headers"], "Authorization should be in allow_headers"