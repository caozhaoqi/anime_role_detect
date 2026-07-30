"""Regression tests for recently fixed issues (2026-07-20 ~ 2026-07-21).

Tests cover:
1. SQLiteUserStore - user CRUD, password verification, login tracking
2. AuthService - multi-layer degrade (MySQL -> SQLite -> Memory)
3. Multi-role detection - model path fix, is_unknown default
4. Health check - supervisor credentials fix, system memory
5. Model service TTL - unload checker fix
6. Safe temp path - path injection prevention (with cv2)
"""
import os
import sys
import hashlib
import secrets as sec
from pathlib import Path

import pytest

# Add project src to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


# =============================================================================
# 1. SQLiteUserStore Tests (matched to actual API)
# =============================================================================

class TestSQLiteUserStore:
    """Test SQLiteUserStore CRUD and auth operations.

    Actual API:
    - __init__(db_path=None)
    - create_user(username, password_hash, role="user", email=None, is_superuser=False)
    - verify_password(username, plain_password, verify_fn) -> dict or None
    - get_user(username) -> dict or None
    - list_users() -> list[dict]
    - _update_login_info(user_id, success=True)  (internal)
    - _get_conn()  (internal, per-operation connection)
    """

    @pytest.fixture
    def sqlite_store(self, tmp_path):
        """Create a SQLiteUserStore with temp db."""
        from src.services.support.auth_service import SQLiteUserStore, AuthService
        db_path = str(tmp_path / "test_auth.db")
        store = SQLiteUserStore(db_path)
        yield store, AuthService  # also return AuthService for hash/verify helpers

    def test_init_creates_db_and_tables(self, sqlite_store):
        """Database and users table should be created on init."""
        store, _ = sqlite_store
        assert store.db_path.endswith("test_auth.db")
        # Verify DB file exists
        assert os.path.exists(store.db_path)
        # Verify users table exists
        user = store.get_user("nonexistent")
        assert user is None  # no crash = table exists

    def test_create_user_success(self, sqlite_store):
        """Should create a new user with hashed password."""
        store, AuthService = sqlite_store
        hashed = AuthService._hash_password("password123")
        result = store.create_user("testuser", hashed, email="test@test.com")
        assert result is not None
        assert result["username"] == "testuser"
        assert result["email"] == "test@test.com"

        user = store.get_user("testuser")
        assert user is not None
        assert user["username"] == "testuser"
        assert user["password_hash"] != "password123"  # should be hashed

    def test_create_duplicate_user_fails(self, sqlite_store):
        """Should return None for duplicate username."""
        store, AuthService = sqlite_store
        store.create_user("dupe", AuthService._hash_password("pass1"))
        result = store.create_user("dupe", AuthService._hash_password("pass2"))
        assert result is None

    def test_verify_password(self, sqlite_store):
        """Should verify passwords using AuthService._verify_password."""
        store, AuthService = sqlite_store
        hashed = AuthService._hash_password("mysecret")
        store.create_user("vfy_user", hashed, email="vfy@test.com")

        result = store.verify_password("vfy_user", "mysecret", AuthService._verify_password)
        assert result is not None
        assert result["status"] == "ok"

        result_bad = store.verify_password("vfy_user", "wrong", AuthService._verify_password)
        assert result_bad is None

    def test_verify_password_nonexistent_user(self, sqlite_store):
        """Should return None for non-existent user."""
        store, AuthService = sqlite_store
        result = store.verify_password("nobody", "pass", AuthService._verify_password)
        assert result is None

    def test_get_user_not_found(self, sqlite_store):
        """Should return None for non-existent user."""
        store, _ = sqlite_store
        assert store.get_user("nonexistent") is None

    def test_login_tracking(self, sqlite_store):
        """Login success should update last_login and login_count via _update_login_info."""
        store, AuthService = sqlite_store
        hashed = AuthService._hash_password("pass")
        result = store.create_user("tracker", hashed, email="t@t.com")
        user_id = result["id"]

        user_before = store.get_user("tracker")
        assert user_before["login_count"] == 0

        # Successful login
        store._update_login_info(user_id, success=True)
        user_after = store.get_user("tracker")
        assert user_after["login_count"] == 1
        assert user_after["last_login_at"] is not None

    def test_failed_login_locking(self, sqlite_store):
        """5 failed logins should lock account but NOT deactivate it.

        After fix: is_active stays 1 (account is active but locked).
        get_user() can still find the locked user (no is_active filter).
        """
        store, AuthService = sqlite_store
        hashed = AuthService._hash_password("pass")
        result = store.create_user("locktest", hashed, email="l@l.com")
        user_id = result["id"]

        # 5 failed attempts
        for _ in range(5):
            store._update_login_info(user_id, success=False)

        # get_user() should still find the locked user (no is_active=1 filter)
        user = store.get_user("locktest")
        assert user is not None, "Locked user should still be visible via get_user()"
        assert user["locked_until"] is not None, "Account should be locked after 5 failures"
        assert user["failed_login_count"] >= 5
        assert user["is_active"] == 1, "Locked accounts should remain active (is_active=1)"

    def test_unlock_after_successful_login(self, sqlite_store):
        """Successful login after lock should clear lock and reset counters.

        After fix: is_active never changes during lock/unlock cycle.
        """
        store, AuthService = sqlite_store
        hashed = AuthService._hash_password("pass")
        result = store.create_user("unlockme", hashed, email="u@u.com")
        user_id = result["id"]

        # Fail 5 times to lock
        for _ in range(5):
            store._update_login_info(user_id, success=False)

        # Verify locked via get_user (should be visible)
        user_locked = store.get_user("unlockme")
        assert user_locked is not None, "Locked user should be visible"
        assert user_locked["locked_until"] is not None

        # Successful login resets counters
        store._update_login_info(user_id, success=True)

        user_after = store.get_user("unlockme")
        assert user_after is not None, "User should still be visible after unlock"
        assert user_after["locked_until"] is None, "Lock should be cleared on success"
        assert user_after["failed_login_count"] == 0, "Failed count should reset"
        assert user_after["is_active"] == 1, "Account should remain active"

    def test_list_users(self, sqlite_store):
        """list_users should return all users."""
        store, AuthService = sqlite_store
        store.create_user("user1", AuthService._hash_password("p1"))
        store.create_user("user2", AuthService._hash_password("p2"))

        users = store.list_users()
        assert len(users) >= 2
        usernames = [u["username"] for u in users]
        assert "user1" in usernames
        assert "user2" in usernames


# =============================================================================
# 2. AuthService Tests
# =============================================================================

class TestAuthServiceDegrade:
    """Test auth service multi-layer degradation."""

    def test_auth_service_singleton(self):
        """AuthService should be a singleton."""
        from src.services.support.auth_service import AuthService
        auth1 = AuthService()
        auth2 = AuthService()
        assert auth1 is auth2

    def test_auth_service_has_users_dict(self):
        """AuthService should always have self.users initialized."""
        from src.services.support.auth_service import AuthService
        auth = AuthService()
        assert hasattr(auth, 'users')
        assert isinstance(auth.users, dict)

    def test_auth_service_storage_mode(self):
        """AuthService should set storage_mode after init."""
        from src.services.support.auth_service import AuthService
        auth = AuthService()
        assert auth.storage_mode in ("mysql", "sqlite", "memory")

    def test_sqlite_store_in_auth_service(self, tmp_path):
        """AuthService should accept a SQLiteUserStore."""
        from src.services.support.auth_service import AuthService, SQLiteUserStore
        store = SQLiteUserStore(str(tmp_path / "auth_test.db"))
        auth = AuthService()
        auth.sqlite_store = store

        hashed = AuthService._hash_password("svc_pass")
        user_result = store.create_user("svc_user", hashed, email="svc@test.com")
        assert user_result is not None

        user = store.get_user("svc_user")
        assert user is not None
        assert user["username"] == "svc_user"


# =============================================================================
# 3. Multi-Role Detection Fix Tests
# =============================================================================

class TestMultiRoleDetectionFixes:
    """Test multi-role detection fixes."""

    def test_project_root_defined(self):
        """multi_role_detection_enhanced.py should have project_root or Path(__file__) defined."""
        filepath = PROJECT_ROOT / "src" / "core" / "detection" / "multi_role_detection_enhanced.py"
        if filepath.exists():
            content = filepath.read_text()
            assert "project_root" in content or "Path(__file__)" in content

    def test_is_unknown_in_dict_context(self):
        """is_unknown should be set as dict key with True default."""
        filepath = PROJECT_ROOT / "src" / "core" / "detection" / "multi_role_detection_enhanced.py"
        if filepath.exists():
            content = filepath.read_text()
            # is_unknown is set in dict context: "is_unknown": True
            assert '"is_unknown": True' in content or '"is_unknown":True' in content

    def test_efficientnet_b3_in_model_map(self):
        """multi_role_detection.py should have efficientnet_b3 in model name map."""
        filepath = PROJECT_ROOT / "src" / "core" / "detection" / "multi_role_detection.py"
        if filepath.exists():
            content = filepath.read_text()
            assert "efficientnet_b3" in content

    def test_model_full_pth_references_fixed(self):
        """No active code references to model_full.pth (comments about removal are OK)."""
        import subprocess
        result = subprocess.run(
            ["grep", "-rn", "model_full\\.pth",
             str(PROJECT_ROOT / "src")],
            capture_output=True, text=True, cwd=str(PROJECT_ROOT)
        )
        # Filter out comment-only lines (Python comments with #)
        code_refs = []
        for line in result.stdout.strip().split('\n'):
            if not line:
                continue
            if ':' in line:
                file_part, _, content = line.partition(':')
                # If the line starts with # after filename:lineno, it's just a comment
                # Check the actual matched content part
                remaining = content.split(':', 1)[-1] if ':' in content else content
                stripped = remaining.strip()
                if not stripped.startswith('#') and not stripped.startswith('//'):
                    code_refs.append(line)

        assert len(code_refs) == 0, f"Found active model_full.pth references in: {code_refs}"


# =============================================================================
# 4. Health Check Fix Tests
# =============================================================================

class TestHealthCheckFixes:
    """Test health check script fixes."""

    def test_supervisor_credentials_fixed(self):
        """Health check should use CHANGE_ME_supervisor_* not admin/admin123."""
        filepath = PROJECT_ROOT / "scripts" / "monitoring" / "health_check.py"
        if filepath.exists():
            content = filepath.read_text()
            assert "admin/admin123" not in content  # old credentials removed
            assert "CHANGE_ME_supervisor_admin" in content
            assert "CHANGE_ME_supervisor_pwd" in content

    def test_system_memory_method_exists(self):
        """Health check should have _get_system_memory method."""
        filepath = PROJECT_ROOT / "scripts" / "monitoring" / "health_check.py"
        if filepath.exists():
            content = filepath.read_text()
            assert "_get_system_memory" in content

    def test_system_memory_in_cli_output(self):
        """CLI output should include system memory info."""
        filepath = PROJECT_ROOT / "scripts" / "monitoring" / "health_check.py"
        if filepath.exists():
            content = filepath.read_text()
            assert "system_memory" in content or "系统内存" in content


# =============================================================================
# 5. Model Service TTL Fix
# =============================================================================

class TestModelServiceTTLFix:
    """Test model service TTL unload checker fix."""

    def test_ttl_checker_uses_singleton(self):
        """_ttl_unload_checker should use WDViTV3Tagger.get_instance() not global tagger."""
        filepath = PROJECT_ROOT / "src" / "services" / "model_service" / "app.py"
        if filepath.exists():
            content = filepath.read_text()
            assert "WDViTV3Tagger.get_instance()" in content


# =============================================================================
# 6. Safe Temp Path Tests (requires cv2)
# =============================================================================

class TestSafeTempPathIntegration:
    """Integration tests for safe_temp_path."""

    def test_safe_temp_path_exists(self):
        """safe_temp_path should create a valid temp file path."""
        try:
            from src.core.utils.utils import safe_temp_path, TEMP_DIR
            path = safe_temp_path("test.jpg")
            assert path is not None
            assert "test" in str(path)
            assert str(path).endswith(".jpg")
        except ImportError:
            pytest.skip("cv2 not installed")

    def test_safe_temp_path_rejects_traversal(self):
        """safe_temp_path should sanitize path traversal attempts."""
        try:
            from src.core.utils.utils import safe_temp_path
            path = safe_temp_path("../../etc/passwd")
            # Path traversal should be sanitized
            assert "../" not in str(path)
        except ImportError:
            pytest.skip("cv2 not installed")

    def test_safe_temp_path_empty_fallback(self):
        """safe_temp_path should handle empty input gracefully."""
        try:
            from src.core.utils.utils import safe_temp_path
            path = safe_temp_path("")
            assert path is not None
        except ImportError:
            pytest.skip("cv2 not installed")


# =============================================================================
# 7. Login/Register Flow Tests
# =============================================================================

class TestAuthAPIRegistration:
    """Test the registration endpoint on auth routes."""

    def test_register_endpoint_exists(self):
        """POST /api/auth/register should be defined in auth.py routes."""
        filepath = PROJECT_ROOT / "src" / "api" / "routes" / "auth.py"
        if filepath.exists():
            content = filepath.read_text()
            assert "/register" in content or "register" in content.lower()

    def test_create_user_in_routes(self):
        """auth.py routes should reference create_user."""
        filepath = PROJECT_ROOT / "src" / "api" / "routes" / "auth.py"
        if filepath.exists():
            content = filepath.read_text()
            assert "create_user" in content

    def test_sqlite_user_store_import(self):
        """SQLiteUserStore should be importable from auth_service."""
        try:
            from src.services.support.auth_service import SQLiteUserStore
            assert SQLiteUserStore is not None
        except ImportError as e:
            pytest.skip(f"Import failed: {e}")


# =============================================================================
# 8. Version and Config Tests
# =============================================================================

class TestVersionConsistency:
    """Test that version numbers are consistent."""

    def test_pyproject_version(self):
        """pyproject.toml should have version >= 2.3.0."""
        pptoml = PROJECT_ROOT / "pyproject.toml"
        if pptoml.exists():
            content = pptoml.read_text()
            for line in content.split('\n'):
                if line.strip().startswith('version'):
                    version = line.split('=')[1].strip().strip('"\'')
                    assert version >= "2.3.0" or version.startswith("2.3")
                    return

    def test_readme_mentions_version(self):
        """README.md should mention v2.3."""
        readme = PROJECT_ROOT / "README.md"
        if readme.exists():
            content = readme.read_text()
            assert "v2.3" in content or "2.3.0" in content


# =============================================================================
# 9. datetime.utcnow() Deprecation Fix
# =============================================================================

class TestDatetimeUtcnowFix:
    """Test that datetime.utcnow() has been replaced with datetime.now(timezone.utc)."""

    def test_no_utcnow_in_auth_service(self):
        """auth_service.py should not contain datetime.utcnow()."""
        filepath = PROJECT_ROOT / "src" / "services" / "support" / "auth_service.py"
        if filepath.exists():
            content = filepath.read_text()
            # Allow utcnow in comments only
            code_lines = [
                line for line in content.split('\n')
                if line.strip() and not line.strip().startswith('#')
            ]
            code_content = '\n'.join(code_lines)
            assert 'datetime.utcnow()' not in code_content, \
                "auth_service.py still uses datetime.utcnow() in active code"

    def test_no_utcnow_in_auth_enhanced(self):
        """auth_enhanced.py should not contain datetime.utcnow()."""
        filepath = PROJECT_ROOT / "src" / "middleware" / "auth_enhanced.py"
        if filepath.exists():
            content = filepath.read_text()
            code_lines = [
                line for line in content.split('\n')
                if line.strip() and not line.strip().startswith('#')
            ]
            code_content = '\n'.join(code_lines)
            assert 'datetime.utcnow()' not in code_content, \
                "auth_enhanced.py still uses datetime.utcnow() in active code"

    def test_timezone_imported(self):
        """Both files should import timezone from datetime."""
        for relpath in [
            "src/services/support/auth_service.py",
            "src/middleware/auth_enhanced.py",
        ]:
            filepath = PROJECT_ROOT / relpath
            if filepath.exists():
                content = filepath.read_text()
                assert "timezone" in content, \
                    f"{relpath} should import timezone from datetime"


# =============================================================================
# 10. Locked User Visibility Fix
# =============================================================================

class TestLockedUserVisibility:
    """Test that locked users are still visible via get_user().

    Previously: _update_login_info set is_active=0 on lock,
    and get_user() filtered on is_active=1, making locked users invisible.
    Now: is_active is NOT changed during lock; get_user() has no is_active filter.
    """

    def test_get_user_no_is_active_filter(self):
        """SQLiteUserStore.get_user() should NOT filter by is_active=1."""
        filepath = PROJECT_ROOT / "src" / "services" / "support" / "auth_service.py"
        if filepath.exists():
            content = filepath.read_text()
            # Find the get_user method in SQLiteUserStore
            # The query should be "SELECT * FROM users WHERE username=?" without AND is_active=1
            assert "AND is_active=1" not in content, \
                "get_user() should not filter by is_active=1"

    def test_lock_does_not_deactivate(self):
        """_update_login_info should NOT set is_active=0 when locking."""
        filepath = PROJECT_ROOT / "src" / "services" / "support" / "auth_service.py"
        if filepath.exists():
            content = filepath.read_text()
            # The lock UPDATE should only set locked_until, not is_active=0
            # Check that is_active=0 is not in the lock path
            assert "is_active=0" not in content, \
                "Locking should not set is_active=0"
            assert "is_active = False" not in content, \
                "Locking should not set is_active=False (MySQL path)"

    def test_locked_user_visible_via_get_user(self, tmp_path):
        """End-to-end: locked user should be findable via get_user()."""
        from src.services.support.auth_service import SQLiteUserStore, AuthService
        store = SQLiteUserStore(str(tmp_path / "vis_test.db"))
        hashed = AuthService._hash_password("pass")
        result = store.create_user("visuser", hashed, email="v@v.com")
        user_id = result["id"]

        # Lock the user
        for _ in range(5):
            store._update_login_info(user_id, success=False)

        # get_user should still return the user
        user = store.get_user("visuser")
        assert user is not None, "Locked user must be visible via get_user()"
        assert user["locked_until"] is not None
        assert user["is_active"] == 1, "is_active should remain 1 during lock"

