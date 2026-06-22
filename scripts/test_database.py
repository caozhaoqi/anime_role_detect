#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试双数据库连接配置
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.config.database import (
    init_database,
    get_db,
    get_local_db,
    get_remote_db,
    is_remote_available,
    get_database_mode,
    get_engine_info,
    create_tables,
    sync_local_to_remote,
)
from src.core.logging.global_logger import get_logger

logger = get_logger("test_database")


def test_database_connection():
    """测试数据库连接"""
    print("=" * 60)
    print("双数据库配置测试")
    print("=" * 60)

    mode = get_database_mode()
    print(f"\n当前数据库模式: {mode}")
    print(f"远程数据库可用: {is_remote_available()}")

    info = get_engine_info()
    print("\n引擎信息:")
    for key, value in info.items():
        print(f"  {key}: {value}")

    print("\n--- 初始化数据库 ---")
    init_database()

    info = get_engine_info()
    print("\n初始化后引擎信息:")
    for key, value in info.items():
        print(f"  {key}: {value}")

    print("\n--- 测试本地数据库连接 ---")
    try:
        db = next(get_local_db())
        print("✓ 本地数据库连接成功")
        db.close()
    except Exception as e:
        print(f"✗ 本地数据库连接失败: {e}")

    print("\n--- 测试远程数据库连接 ---")
    try:
        db = next(get_remote_db())
        print("✓ 远程数据库连接成功")
        db.close()
    except Exception as e:
        print(f"✗ 远程数据库连接失败: {e}")

    print("\n--- 测试默认数据库连接 ---")
    try:
        db = next(get_db())
        print(f"✓ 默认数据库连接成功（模式: {mode}）")
        db.close()
    except Exception as e:
        print(f"✗ 默认数据库连接失败: {e}")

    print("\n--- 创建表 ---")
    try:
        create_tables()
        print("✓ 表创建成功")
    except Exception as e:
        print(f"✗ 表创建失败: {e}")

    if mode == "dual" or is_remote_available():
        print("\n--- 测试数据同步 ---")
        try:
            result = sync_local_to_remote()
            if result:
                print("✓ 数据同步成功")
            else:
                print("✗ 数据同步失败")
        except Exception as e:
            print(f"✗ 数据同步失败: {e}")

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)


if __name__ == "__main__":
    test_database_connection()