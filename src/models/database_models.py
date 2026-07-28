#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from datetime import datetime

try:
    import bcrypt
    HAS_BCRYPT = True
except ImportError:
    HAS_BCRYPT = False

try:
    from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text
    from src.core.config.database import Base
    HAS_SQLALCHEMY = True
except ImportError:
    HAS_SQLALCHEMY = False
    Base = None
    Column = None
    Integer = None
    String = None
    DateTime = None
    Boolean = None
    Text = None


def make_model_class(class_name, table_name, columns_def):
    """动态创建模型类"""
    if HAS_SQLALCHEMY:
        attrs = {"__tablename__": table_name}
        for col_name, col_type in columns_def.items():
            attrs[col_name] = col_type
        
        if table_name == "recognition_records":
            attrs["__description__"] = "识别记录"
            attrs["__blur_field__"] = ['image_filename', 'username']
            attrs["__ignore_fields__"] = []
        elif table_name == "api_keys":
            attrs["__description__"] = "API密钥"
            attrs["__ignore_fields__"] = ['key']
        elif table_name == "system_configs":
            attrs["__description__"] = "系统配置"
            attrs["__blur_field__"] = ['key', 'description']
        elif table_name == "cleaning_records":
            attrs["__description__"] = "清理记录"
            attrs["__blur_field__"] = ['input_dir', 'output_dir']
        elif table_name == "user_feedback":
            attrs["__description__"] = "用户反馈"
            attrs["__blur_field__"] = ['username', 'comment']
        
        return type(class_name, (Base,), attrs)
    else:
        attrs = {"__tablename__": table_name}

        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

        attrs["__init__"] = __init__
        return type(class_name, (), attrs)


# 只在 sqlalchemy 可用时定义列定义
if HAS_SQLALCHEMY:
    columns_def_user = {
        "id": Column(Integer, primary_key=True, index=True),
        "username": Column(String(50), unique=True, index=True, nullable=False),
        "password_hash": Column(String(255), nullable=False),
        "role": Column(String(20), default="user", nullable=False),
        "email": Column(String(100), unique=True, index=True),
        "nickname": Column(String(50)),
        "avatar_url": Column(String(255)),
        "is_active": Column(Boolean, default=True),
        "is_superuser": Column(Boolean, default=False),
        "created_at": Column(DateTime, default=datetime.utcnow),
        "updated_at": Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow),
        "last_login_at": Column(DateTime),
        "login_count": Column(Integer, default=0),
        "failed_login_count": Column(Integer, default=0),
        "locked_until": Column(DateTime),
    }
else:
    columns_def_user = {}


class UserModel(Base if HAS_SQLALCHEMY else object):
    """用户模型"""
    __tablename__ = "users"
    __description__ = "系统用户"
    __blur_field__ = ['username', 'email', 'nickname']
    __ignore_fields__ = ['password_hash', 'failed_login_count', 'locked_until']
    __display_field__ = 'username'

    if HAS_SQLALCHEMY:
        id = Column(Integer, primary_key=True, index=True)
        username = Column(String(50), unique=True, index=True, nullable=False)
        password_hash = Column(String(255), nullable=False)
        role = Column(String(20), default="user", nullable=False)
        email = Column(String(100), unique=True, index=True)
        nickname = Column(String(50))
        avatar_url = Column(String(255))
        is_active = Column(Boolean, default=True)
        is_superuser = Column(Boolean, default=False)
        created_at = Column(DateTime, default=datetime.utcnow)
        updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
        last_login_at = Column(DateTime)
        login_count = Column(Integer, default=0)
        failed_login_count = Column(Integer, default=0)
        locked_until = Column(DateTime)
    else:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    def verify_password(self, password: str) -> bool:
        """验证密码"""
        if not HAS_BCRYPT:
            # bcrypt 不可用：set_password 会以明文存储密码，此处做明文比较以保持一致，
            # 否则会出现「注册成功但登录永远失败」的认证死锁。
            return self.password_hash == password
        try:
            return bcrypt.checkpw(password.encode('utf-8'), self.password_hash.encode('utf-8'))
        except Exception:
            return False

    def set_password(self, password: str):
        """设置密码"""
        if not HAS_BCRYPT:
            self.password_hash = password
            return
        self.password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')


# 动态创建其他模型类
if HAS_SQLALCHEMY:
    RecognitionRecordModel = make_model_class("RecognitionRecordModel", "recognition_records", {
        "id": Column(String(64), primary_key=True, index=True),
        "user_id": Column(String(64), index=True, nullable=False),
        "username": Column(String(100), default="anonymous"),
        "image_filename": Column(String(500)),
        "image_path": Column(String(500)),
        "recognition_result": Column(Text),
        "model_used": Column(String(100)),
        "processing_time": Column(Integer),
        "is_multi_role": Column(Boolean, default=False),
        "nsfw_status": Column(Boolean, default=False),
        "detected_text": Column(Boolean, default=False),
        "created_at": Column(DateTime, default=datetime.utcnow),
    })

    ApiKeyModel = make_model_class("ApiKeyModel", "api_keys", {
        "id": Column(Integer, primary_key=True, index=True),
        "user_id": Column(Integer, index=True),
        "key": Column(String(64), unique=True, index=True, nullable=False),
        "name": Column(String(100)),
        "description": Column(Text),
        "permissions": Column(Text),
        "is_active": Column(Boolean, default=True),
        "expires_at": Column(DateTime),
        "created_at": Column(DateTime, default=datetime.utcnow),
        "last_used_at": Column(DateTime),
        "usage_count": Column(Integer, default=0),
    })

    SystemConfigModel = make_model_class("SystemConfigModel", "system_configs", {
        "id": Column(Integer, primary_key=True, index=True),
        "key": Column(String(100), unique=True, index=True, nullable=False),
        "value": Column(Text),
        "description": Column(String(500)),
        "category": Column(String(50)),
        "created_at": Column(DateTime, default=datetime.utcnow),
        "updated_at": Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow),
    })

    CleaningRecordModel = make_model_class("CleaningRecordModel", "cleaning_records", {
        "id": Column(String(64), primary_key=True, index=True),
        "user_id": Column(String(64), index=True, nullable=False),
        "username": Column(String(100), default="anonymous"),
        "input_dir": Column(String(500)),
        "output_dir": Column(String(500)),
        "config": Column(Text),
        "status": Column(String(20), default="pending"),
        "total_files": Column(Integer, default=0),
        "processed_files": Column(Integer, default=0),
        "valid_files": Column(Integer, default=0),
        "rejected_files": Column(Integer, default=0),
        "duplicate_files": Column(Integer, default=0),
        "report_path": Column(String(500)),
        "error_message": Column(Text),
        "duration_seconds": Column(Integer, default=0),
        "created_at": Column(DateTime, default=datetime.utcnow),
        "started_at": Column(DateTime),
        "completed_at": Column(DateTime),
    })

    UserFeedbackModel = make_model_class("UserFeedbackModel", "user_feedback", {
        "id": Column(String(64), primary_key=True, index=True),
        "user_id": Column(String(64), index=True, nullable=False),
        "username": Column(String(100)),
        "record_id": Column(String(64), index=True),
        "image_filename": Column(String(500)),
        "feedback_type": Column(String(50), nullable=False),
        "rating": Column(Integer),
        "comment": Column(Text),
        "correction_role": Column(String(200)),
        "is_correct": Column(Boolean),
        "extra_data": Column(Text),
        "created_at": Column(DateTime, default=datetime.utcnow),
    })
else:
    RecognitionRecordModel = make_model_class("RecognitionRecordModel", "recognition_records", {})
    ApiKeyModel = make_model_class("ApiKeyModel", "api_keys", {})
    SystemConfigModel = make_model_class("SystemConfigModel", "system_configs", {})
    CleaningRecordModel = make_model_class("CleaningRecordModel", "cleaning_records", {})
    UserFeedbackModel = make_model_class("UserFeedbackModel", "user_feedback", {})
