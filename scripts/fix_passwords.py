#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""修复数据库密码"""

import hashlib
import os
from database import SessionLocal, User


def hash_password(password):
    """哈希密码"""
    salt = hashlib.md5(os.urandom(16)).hexdigest()
    hashed = hashlib.sha256(f"{password}{salt}".encode()).hexdigest()
    return f"sha256${salt}${hashed}"


def fix_passwords():
    """修复数据库中的密码"""
    db = SessionLocal()
    try:
        # 更新 admin 用户密码
        admin = db.query(User).filter(User.username == "admin").first()
        if admin:
            admin.password = hash_password("admin123")
            print("Updated admin password")

        # 更新 developer 用户密码
        developer = db.query(User).filter(User.username == "developer").first()
        if developer:
            developer.password = hash_password("developer123")
            print("Updated developer password")

        # 更新 user 用户密码
        user = db.query(User).filter(User.username == "user").first()
        if user:
            user.password = hash_password("user123")
            print("Updated user password")

        db.commit()
        print("Passwords updated successfully!")
    finally:
        db.close()


if __name__ == "__main__":
    fix_passwords()
