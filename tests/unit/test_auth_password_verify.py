"""
回归测试：UserModel 密码 set/verify 必须一致。

修复前 bug：bcrypt 不可用时 set_password 以明文存储密码，但 verify_password
在 `not HAS_BCRYPT` 分支直接 `return False`，导致「注册成功、登录永远失败」的
认证死锁（与 MySQL 是否可达无关，任何未安装 bcrypt 的环境都会触发）。

本测试验证：无论 bcrypt 是否可用，set_password(pw) 后 verify_password(pw)
必须为 True、verify_password(错误密码) 必须为 False。
"""
import pytest

from src.models.database_models import UserModel


def test_usermodel_set_verify_consistent():
    """set_password 与 verify_password 的哈希方案必须对称。"""
    user = UserModel(username="regr_user", email="regr_user@example.com")
    pw = "Secret@12345"
    user.set_password(pw)
    assert user.verify_password(pw) is True
    assert user.verify_password("wrong-password") is False


def test_usermodel_plaintext_fallback_when_no_bcrypt():
    """bcrypt 不可用时，verify_password 必须支持明文校验（而非无条件返回 False）。"""
    if getattr(UserModel, "HAS_BCRYPT", True):
        pytest.skip("需要 bcrypt 不可用环境以覆盖明文回退分支")
    user = UserModel(username="plain_user", email="plain_user@example.com")
    user.set_password("Plain@12345")
    # 明文回退：password_hash 即明文
    assert user.password_hash == "Plain@12345"
    assert user.verify_password("Plain@12345") is True
    assert user.verify_password("nope") is False
