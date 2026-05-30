#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARD Skill Hub - 用户管理工具
用于创建、管理开发者账户和普通用户
"""

import json
import argparse
import getpass
from pathlib import Path
from datetime import datetime
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["sha256_crypt"], deprecated="auto")

USER_DATA_PATH = Path.home() / ".ardc" / "users.json"


def load_users():
    """加载用户数据"""
    if USER_DATA_PATH.exists():
        with open(USER_DATA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_users(users):
    """保存用户数据"""
    USER_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(USER_DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2, ensure_ascii=False)


def create_user(username: str, email: str, password: str, is_developer: bool = False):
    """创建用户账户"""
    users = load_users()

    # 检查用户是否已存在
    if username in users:
        print(f"❌ 错误: 用户 '{username}' 已存在")
        return False

    # 检查邮箱是否已存在
    for existing_user in users.values():
        if existing_user.get("email") == email:
            print(f"❌ 错误: 邮箱 '{email}' 已被注册")
            return False

    # 检查密码强度
    if len(password) < 6:
        print("❌ 错误: 密码长度至少需要6位")
        return False

    # 创建用户账户
    now = datetime.now().isoformat()
    hashed_password = pwd_context.hash(password[:72])

    users[username] = {
        "id": username,
        "username": username,
        "email": email,
        "hashed_password": hashed_password,
        "is_developer": is_developer,
        "created_at": now,
        "updated_at": now,
    }

    save_users(users)

    role = "开发者" if is_developer else "普通用户"
    print(f"\n✅ 用户创建成功!")
    print(f"   ├── 用户名: {username}")
    print(f"   ├── 邮箱: {email}")
    print(f"   └── 权限: {role}")
    return True


def delete_user(username: str):
    """删除用户账户"""
    users = load_users()

    if username not in users:
        print(f"❌ 错误: 用户 '{username}' 不存在")
        return False

    del users[username]
    save_users(users)

    print(f"✅ 用户 '{username}' 已删除")
    return True


def list_users():
    """列出所有用户"""
    users = load_users()

    if not users:
        print("📭 暂无用户")
        return

    print("\n📋 用户列表:")
    print("-" * 80)
    print(f"{'用户名':<15} {'邮箱':<30} {'权限':<10} {'创建时间':<25}")
    print("-" * 80)

    for username, user in users.items():
        role = "👑 开发者" if user.get("is_developer") else "👤 普通用户"
        print(
            f"{username:<15} {user.get('email', ''):<30} {role:<10} {user.get('created_at', '')[:25]}"
        )

    print("-" * 80)
    print(f"总计: {len(users)} 个用户")


def promote_user(username: str):
    """提升用户为开发者"""
    users = load_users()

    if username not in users:
        print(f"❌ 错误: 用户 '{username}' 不存在")
        return False

    if users[username].get("is_developer"):
        print(f"⚠️  用户 '{username}' 已经是开发者")
        return False

    users[username]["is_developer"] = True
    users[username]["updated_at"] = datetime.now().isoformat()
    save_users(users)

    print(f"✅ 用户 '{username}' 已提升为开发者")
    return True


def demote_user(username: str):
    """取消开发者权限"""
    users = load_users()

    if username not in users:
        print(f"❌ 错误: 用户 '{username}' 不存在")
        return False

    if not users[username].get("is_developer"):
        print(f"⚠️  用户 '{username}' 不是开发者")
        return False

    users[username]["is_developer"] = False
    users[username]["updated_at"] = datetime.now().isoformat()
    save_users(users)

    print(f"✅ 用户 '{username}' 已降为普通用户")
    return True


def interactive_mode():
    """交互式模式"""
    print(
        """
╔══════════════════════════════════════════════╗
║     ARD Skill Hub 用户管理工具                ║
╚══════════════════════════════════════════════╝
    """
    )

    while True:
        print("\n请选择操作:")
        print("1. 创建用户")
        print("2. 创建开发者")
        print("3. 列出用户")
        print("4. 提升为开发者")
        print("5. 取消开发者权限")
        print("6. 删除用户")
        print("0. 退出")

        choice = input("\n输入选项 [0-6]: ").strip()

        if choice == "0":
            print("👋 退出")
            break

        elif choice == "1":
            username = input("用户名: ").strip()
            email = input("邮箱: ").strip()
            password = getpass.getpass("密码: ")
            confirm = getpass.getpass("确认密码: ")

            if password != confirm:
                print("❌ 密码不一致")
                continue

            create_user(username, email, password, False)

        elif choice == "2":
            username = input("用户名: ").strip()
            email = input("邮箱: ").strip()
            password = getpass.getpass("密码: ")
            confirm = getpass.getpass("确认密码: ")

            if password != confirm:
                print("❌ 密码不一致")
                continue

            create_user(username, email, password, True)

        elif choice == "3":
            list_users()

        elif choice == "4":
            username = input("要提升的用户名: ").strip()
            promote_user(username)

        elif choice == "5":
            username = input("要降级的用户名: ").strip()
            demote_user(username)

        elif choice == "6":
            username = input("要删除的用户名: ").strip()
            confirm = input(f"确认删除用户 '{username}'? (y/N): ").strip().lower()
            if confirm == "y":
                delete_user(username)

        else:
            print("❌ 无效选项")


def main():
    """主入口"""
    parser = argparse.ArgumentParser(
        prog="init_admin.py",
        description="ARD Skill Hub 用户管理工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 交互式模式
  python3 init_admin.py -i
  
  # 创建普通用户
  python3 init_admin.py create -u user -e user@test.com -p pass123
  
  # 创建开发者
  python3 init_admin.py create -u admin -e admin@test.com -p pass123 --developer
  
  # 列出用户
  python3 init_admin.py list
  
  # 提升为开发者
  python3 init_admin.py promote -u username
  
  # 删除用户
  python3 init_admin.py delete -u username
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # interactive
    subparsers.add_parser("interactive", aliases=["i"], help="交互式模式")

    # create
    create_parser = subparsers.add_parser("create", help="创建用户")
    create_parser.add_argument("-u", "--username", required=True, help="用户名")
    create_parser.add_argument("-e", "--email", required=True, help="邮箱")
    create_parser.add_argument("-p", "--password", required=True, help="密码")
    create_parser.add_argument("--developer", action="store_true", help="创建为开发者")

    # list
    subparsers.add_parser("list", help="列出所有用户")

    # promote
    promote_parser = subparsers.add_parser("promote", help="提升为开发者")
    promote_parser.add_argument("-u", "--username", required=True, help="用户名")

    # demote
    demote_parser = subparsers.add_parser("demote", help="取消开发者权限")
    demote_parser.add_argument("-u", "--username", required=True, help="用户名")

    # delete
    delete_parser = subparsers.add_parser("delete", help="删除用户")
    delete_parser.add_argument("-u", "--username", required=True, help="用户名")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    if args.command in ["interactive", "i"]:
        interactive_mode()

    elif args.command == "create":
        create_user(args.username, args.email, args.password, args.developer)

    elif args.command == "list":
        list_users()

    elif args.command == "promote":
        promote_user(args.username)

    elif args.command == "demote":
        demote_user(args.username)

    elif args.command == "delete":
        delete_user(args.username)


if __name__ == "__main__":
    main()
