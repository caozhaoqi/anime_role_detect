#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARD Skill Hub CLI 工具
提供技能管理命令行接口
"""

import argparse
import json
import os
import sys
import subprocess

try:
    import requests
except ImportError:
    print("❌ 缺少 requests 模块，请安装: pip install requests")
    sys.exit(1)

API_URL = "http://47.79.91.89:8888/api"
CONFIG_DIR = os.path.expanduser("~/.ardc")
CONFIG_FILE = os.path.join(CONFIG_DIR, "config.json")

def load_config():
    """加载配置文件"""
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_config(config):
    """保存配置文件"""
    os.makedirs(CONFIG_DIR, exist_ok=True)
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

def api_request(method, endpoint, data=None, token=None):
    """发送 API 请求"""
    url = f"{API_URL}{endpoint}"
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    
    try:
        if method == "GET":
            response = requests.get(url, headers=headers, params=data)
        elif method == "POST":
            response = requests.post(url, headers=headers, json=data)
        elif method == "DELETE":
            response = requests.delete(url, headers=headers)
        else:
            print(f"❌ 不支持的方法: {method}")
            return None
        
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 401:
            print("❌ 未授权，请先登录")
            return None
        else:
            print(f"❌ 请求失败 [{response.status_code}]: {response.text}")
            return None
    except Exception as e:
        print(f"❌ 网络错误: {e}")
        return None

def cmd_version(args):
    """显示版本信息"""
    print("ARD Skill Hub CLI v1.0.0")
    print("技能仓库命令行工具")

def cmd_login(args):
    """用户登录"""
    import getpass
    username = args.username or input("用户名: ")
    password = args.password or getpass.getpass("密码: ")
    
    data = {"username": username, "password": password}
    response = requests.post(f"{API_URL}/auth/login", data=data)
    
    if response.status_code == 200:
        result = response.json()
        config = load_config()
        config["token"] = result["access_token"]
        config["user"] = result["user"]
        save_config(config)
        print(f"✅ 登录成功! 欢迎, {result['user']['username']}")
    else:
        print(f"❌ 登录失败: {response.text}")

def cmd_logout(args):
    """用户登出"""
    config = load_config()
    token = config.get("token")
    
    if token:
        api_request("POST", "/auth/logout", token=token)
    
    config.pop("token", None)
    config.pop("user", None)
    save_config(config)
    print("✅ 已登出")

def cmd_whoami(args):
    """显示当前用户"""
    config = load_config()
    user = config.get("user")
    
    if user:
        print(f"用户名: {user['username']}")
        print(f"邮箱: {user['email']}")
        print(f"开发者: {'是' if user.get('is_developer') else '否'}")
    else:
        print("❌ 未登录")

def cmd_skill_list(args):
    """列出所有技能"""
    category = args.category
    endpoint = f"/skills?category={category}" if category else "/skills"
    result = api_request("GET", endpoint)
    
    if result:
        skills = result.get("skills", [])
        if not skills:
            print("暂无技能")
            return
        
        print(f"共 {len(skills)} 个技能:")
        print("-" * 80)
        for skill in skills:
            print(f"📦 {skill['id']}")
            print(f"   名称: {skill['name']}")
            print(f"   版本: {skill['version']}")
            print(f"   分类: {skill['category']}")
            print(f"   描述: {skill['description'][:50]}..." if len(skill['description']) > 50 else f"   描述: {skill['description']}")
            print("-" * 80)

def cmd_skill_search(args):
    """搜索技能"""
    keyword = args.keyword
    result = api_request("GET", f"/search?keyword={keyword}")
    
    if result:
        total = result.get("total", 0)
        skills = result.get("skills", [])
        print(f"找到 {total} 个结果:")
        print("-" * 80)
        for skill in skills:
            print(f"📦 {skill['id']} - {skill['name']}")
            print(f"   版本: {skill['version']}")
            print(f"   标签: {', '.join(skill.get('tags', []))}")
            print("-" * 80)

def cmd_skill_info(args):
    """显示技能详情"""
    skill_id = args.skill_id
    result = api_request("GET", f"/skills/{skill_id}")
    
    if result:
        print("技能详情:")
        print("-" * 80)
        print(f"ID: {result['id']}")
        print(f"名称: {result['name']}")
        print(f"版本: {result['version']}")
        print(f"分类: {result['category']}")
        print(f"作者: {result['author']}")
        print(f"描述: {result['description']}")
        print(f"标签: {', '.join(result.get('tags', []))}")
        print(f"运行时: {result.get('runtime', '未知')}")
        print(f"状态: {result.get('status', '未知')}")
        print("-" * 80)

def cmd_skill_install(args):
    """安装技能"""
    skill_id = args.skill_id
    config = load_config()
    token = config.get("token")
    
    result = api_request("POST", f"/skills/{skill_id}/install", token=token)
    
    if result:
        print(f"✅ {result.get('message', '安装成功')}")

def cmd_skill_uninstall(args):
    """卸载技能"""
    skill_id = args.skill_id
    config = load_config()
    token = config.get("token")
    
    result = api_request("DELETE", f"/skills/{skill_id}/uninstall", token=token)
    
    if result:
        print(f"✅ {result.get('message', '卸载成功')}")

def cmd_skill_versions(args):
    """查看技能版本"""
    skill_id = args.skill_id
    result = api_request("GET", f"/skills/{skill_id}/versions")
    
    if result:
        versions = result.get("versions", [])
        print(f"技能 {skill_id} 的版本:")
        for v in versions:
            print(f"  • {v.get('version', '未知')}")

def cmd_stats(args):
    """显示统计信息"""
    result = api_request("GET", "/stats")
    
    if result:
        print("技能仓库统计:")
        print("-" * 40)
        print(f"技能总数: {result.get('total_skills', 0)}")
        print(f"分类总数: {result.get('total_categories', 0)}")
        print(f"标签总数: {result.get('total_tags', 0)}")
        print("\n分类分布:")
        for category, count in result.get("categories", {}).items():
            print(f"  • {category}: {count}")

def cmd_categories(args):
    """列出所有分类"""
    result = api_request("GET", "/categories")
    
    if result:
        print("技能分类:")
        for category, count in result.items():
            print(f"  • {category}: {count} 个技能")

def cmd_tags(args):
    """列出所有标签"""
    result = api_request("GET", "/tags")
    
    if result:
        print(f"所有标签 ({len(result)} 个):")
        print(", ".join(result.keys()))

def main():
    """主入口"""
    parser = argparse.ArgumentParser(
        prog="ardc",
        description="ARD Skill Hub - 技能仓库命令行工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  ardc --version                    # 显示版本
  ardc login                        # 用户登录
  ardc logout                       # 用户登出
  ardc whoami                       # 显示当前用户
  ardc skill list                   # 列出所有技能
  ardc skill search 数据            # 搜索技能
  ardc skill info ardc-cleaner      # 查看技能详情
  ardc skill install ardc-cleaner   # 安装技能
  ardc skill uninstall ardc-cleaner # 卸载技能
  ardc stats                        # 显示统计信息
  ardc categories                   # 列出分类
  ardc tags                         # 列出标签
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="可用命令")
    
    # version
    subparsers.add_parser("version", help="显示版本信息")
    
    # login
    login_parser = subparsers.add_parser("login", help="用户登录")
    login_parser.add_argument("-u", "--username", help="用户名")
    login_parser.add_argument("-p", "--password", help="密码")
    
    # logout
    subparsers.add_parser("logout", help="用户登出")
    
    # whoami
    subparsers.add_parser("whoami", help="显示当前用户")
    
    # skill
    skill_parser = subparsers.add_parser("skill", help="技能管理")
    skill_subparsers = skill_parser.add_subparsers(dest="skill_command")
    
    skill_subparsers.add_parser("list", help="列出所有技能").add_argument("-c", "--category", help="按分类筛选")
    skill_subparsers.add_parser("search", help="搜索技能").add_argument("keyword", help="搜索关键词")
    skill_subparsers.add_parser("info", help="查看技能详情").add_argument("skill_id", help="技能ID")
    skill_subparsers.add_parser("install", help="安装技能").add_argument("skill_id", help="技能ID")
    skill_subparsers.add_parser("uninstall", help="卸载技能").add_argument("skill_id", help="技能ID")
    skill_subparsers.add_parser("versions", help="查看技能版本").add_argument("skill_id", help="技能ID")
    
    # stats
    subparsers.add_parser("stats", help="显示统计信息")
    
    # categories
    subparsers.add_parser("categories", help="列出所有分类")
    
    # tags
    subparsers.add_parser("tags", help="列出所有标签")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # 路由命令
    commands = {
        "version": cmd_version,
        "login": cmd_login,
        "logout": cmd_logout,
        "whoami": cmd_whoami,
        "stats": cmd_stats,
        "categories": cmd_categories,
        "tags": cmd_tags,
    }
    
    if args.command == "skill":
        skill_commands = {
            "list": cmd_skill_list,
            "search": cmd_skill_search,
            "info": cmd_skill_info,
            "install": cmd_skill_install,
            "uninstall": cmd_skill_uninstall,
            "versions": cmd_skill_versions,
        }
        if args.skill_command in skill_commands:
            skill_commands[args.skill_command](args)
        else:
            skill_parser.print_help()
    elif args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()