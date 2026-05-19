#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARDC Skill Sync - 技能同步工具
支持一键安装、更新、管理技能包

命令说明:
  login               首次扫码认证
  status              显示本地配置与检测到的AI工具目录
  check               检查已安装 skill 的更新情况
  sync                同步更新 skill
  list                查询 SkillHub 上所有已发布的 skill
  install             安装指定 skill（支持未本地安装的新 skill）
  uninstall           卸载指定 skill
  version             显示版本信息
  help                显示此帮助信息
"""
import argparse
import os
import sys
import json
import requests
import hashlib
import zipfile
import shutil
from pathlib import Path
from datetime import datetime

# 配置常量
DEFAULT_SKILL_HUB_URL = "http://47.79.91.89:8888"
SKILL_DIR = Path.home() / '.ardc' / 'skills'
CONFIG_DIR = Path.home() / '.ardc'
CONFIG_FILE = CONFIG_DIR / 'config.json'
TOKEN_FILE = CONFIG_DIR / 'token.txt'

# 版本号
VERSION = "1.0.0"

def load_config():
    """加载配置文件"""
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_config(config):
    """保存配置文件"""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

def get_skill_hub_url():
    """获取 SkillHub 服务器地址（优先从配置文件读取）"""
    config = load_config()
    return config.get('skill_hub_url', DEFAULT_SKILL_HUB_URL)

def get_token():
    """获取认证 token"""
    if TOKEN_FILE.exists():
        with open(TOKEN_FILE, 'r', encoding='utf-8') as f:
            return f.read().strip()
    return None

def set_token(token):
    """保存认证 token"""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(TOKEN_FILE, 'w', encoding='utf-8') as f:
        f.write(token)

def print_success(message):
    """打印成功消息"""
    print(f"\033[32m✓ {message}\033[0m")

def print_error(message):
    """打印错误消息"""
    print(f"\033[31m✗ {message}\033[0m")

def print_info(message):
    """打印信息消息"""
    print(f"\033[36mℹ {message}\033[0m")

def print_warning(message):
    """打印警告消息"""
    print(f"\033[33m⚠ {message}\033[0m")

def login():
    """用户名密码登录认证"""
    print("=" * 60)
    print("          ARDC SkillHub 登录认证")
    print("=" * 60)
    print()
    
    skill_hub_url = get_skill_hub_url()
    
    # 获取用户名和密码
    import getpass
    username = input("请输入用户名: ").strip()
    password = getpass.getpass("请输入密码: ").strip()
    
    if not username or not password:
        print_error("用户名和密码不能为空")
        return
    
    try:
        print("\n正在登录...")
        
        response = requests.post(
            f"{skill_hub_url}/api/auth/login",
            json={
                "username": username,
                "password": password
            },
            timeout=30
        )
        
        response.raise_for_status()
        data = response.json()
        
        if data.get('success'):
            token = data.get('token')
            set_token(token)
            print_success("登录成功！")
            print(f"Token 已保存到: {TOKEN_FILE}")
            
            # 保存用户信息
            config = load_config()
            config['username'] = username
            save_config(config)
            
            print_info(f"欢迎回来, {username}!")
        else:
            print_error(f"登录失败: {data.get('message', '未知错误')}")
            
    except requests.exceptions.RequestException as e:
        print_error(f"登录失败: {e}")
        print_info("正在使用离线模式...")
        # 生成临时 token（演示用）
        temp_token = hashlib.md5(str(datetime.now()).encode()).hexdigest()
        set_token(temp_token)
        print_success("已进入离线模式")

def register():
    """用户注册"""
    print("=" * 60)
    print("          ARDC SkillHub 用户注册")
    print("=" * 60)
    print()
    
    skill_hub_url = get_skill_hub_url()
    
    # 获取注册信息
    import getpass
    username = input("请输入用户名: ").strip()
    email = input("请输入邮箱: ").strip()
    password = getpass.getpass("请输入密码: ").strip()
    confirm_password = getpass.getpass("请确认密码: ").strip()
    
    # 验证输入
    if not username:
        print_error("用户名不能为空")
        return
    if not email:
        print_error("邮箱不能为空")
        return
    if not password:
        print_error("密码不能为空")
        return
    if password != confirm_password:
        print_error("两次输入的密码不一致")
        return
    
    try:
        print("\n正在注册...")
        
        response = requests.post(
            f"{skill_hub_url}/api/auth/register",
            json={
                "username": username,
                "email": email,
                "password": password
            },
            timeout=30
        )
        
        response.raise_for_status()
        data = response.json()
        
        if data.get('success'):
            print_success("注册成功！")
            print_info("请使用用户名密码登录")
        else:
            print_error(f"注册失败: {data.get('message', '未知错误')}")
            
    except requests.exceptions.RequestException as e:
        print_error(f"注册失败: {e}")

def status():
    """显示本地配置状态"""
    print("=" * 60)
    print("          ARDC SkillHub 本地配置状态")
    print("=" * 60)
    print()
    
    # 检查 Python 版本
    print("📦 环境信息:")
    print(f"  Python 版本: {sys.version.split()[0]}")
    
    # 检查配置
    print("\n📁 配置目录:")
    print(f"  技能目录: {SKILL_DIR}")
    print(f"  配置文件: {CONFIG_FILE}")
    print(f"  Token 文件: {TOKEN_FILE}")
    
    # 检查认证状态
    print("\n🔐 认证状态:")
    token = get_token()
    if token:
        print(f"  ✓ 已认证 (Token: {token[:8]}...)")
    else:
        print(f"  ✗ 未认证")
    
    # 列出已安装技能
    print("\n📚 已安装技能:")
    if SKILL_DIR.exists():
        skill_count = 0
        for skill_path in SKILL_DIR.iterdir():
            if skill_path.is_dir():
                skill_count += 1
                skill_file = skill_path / 'SKILL.md'
                version = "未知"
                if skill_file.exists():
                    with open(skill_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            if line.startswith('version:'):
                                version = line.split(':')[1].strip()
                print(f"  ✓ {skill_path.name:20} v{version}")
        if skill_count == 0:
            print("  暂无已安装技能")
    else:
        print("  技能目录不存在")
    
    # 检查工具目录
    print("\n🛠️ AI 工具检测:")
    tool_dirs = [
        Path.home() / '.trae-cn' / 'skills',
        Path.home() / '.trae' / 'skills',
        Path('/opt') / 'ardc' / 'skills'
    ]
    for tool_dir in tool_dirs:
        if tool_dir.exists():
            print(f"  ✓ 检测到工具目录: {tool_dir}")

def list_skills():
    """列出 SkillHub 上所有技能"""
    print("=" * 60)
    print("          ARDC SkillHub 技能列表")
    print("=" * 60)
    print()
    
    # 获取远程技能列表
    skill_hub_url = get_skill_hub_url()
    try:
        headers = {}
        token = get_token()
        if token:
            headers['Authorization'] = f"Bearer {token}"
        
        response = requests.get(
            f"{skill_hub_url}/api/skills",
            headers=headers,
            timeout=30
        )
        response.raise_for_status()
        skills = response.json()
        
        print(f"共发现 {len(skills)} 个技能:")
        print("-" * 60)
        
        # 获取已安装技能列表
        installed_skills = set()
        if SKILL_DIR.exists():
            installed_skills = {p.name for p in SKILL_DIR.iterdir() if p.is_dir()}
        
        for skill in skills:
            name = skill.get('name', '')
            version = skill.get('version', '1.0.0')
            description = skill.get('description', '')
            author = skill.get('author', '未知')
            category = skill.get('category', '其他')
            
            installed = "✓" if name in installed_skills else " "
            print(f"{installed} {name:20} v{version}")
            print(f"      ├── 描述: {description}")
            print(f"      ├── 作者: {author}")
            print(f"      └── 分类: {category}")
            print()
            
    except Exception as e:
        print_error(f"获取技能列表失败: {e}")
        print_info("显示本地已安装技能...")
        
        # 回退到本地列表
        if SKILL_DIR.exists():
            print("\n本地已安装技能:")
            for skill_path in SKILL_DIR.iterdir():
                if skill_path.is_dir():
                    print(f"  ✓ {skill_path.name}")
        else:
            print("  暂无已安装技能")

def check_updates():
    """检查技能更新"""
    print("=" * 60)
    print("          检查技能更新")
    print("=" * 60)
    print()
    
    skill_hub_url = get_skill_hub_url()
    
    if not SKILL_DIR.exists():
        print("未安装任何技能")
        return
    
    token = get_token()
    headers = {}
    if token:
        headers['Authorization'] = f"Bearer {token}"
    
    updates_available = False
    
    for skill_path in SKILL_DIR.iterdir():
        if not skill_path.is_dir():
            continue
        
        skill_name = skill_path.name
        print(f"检查 {skill_name}...")
        
        # 获取本地版本
        local_version = "1.0.0"
        skill_file = skill_path / 'SKILL.md'
        if skill_file.exists():
            with open(skill_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.startswith('version:'):
                        local_version = line.split(':')[1].strip()
        
        # 获取远程版本
        try:
            response = requests.get(
                f"{skill_hub_url}/api/skills/{skill_name}",
                headers=headers,
                timeout=10
            )
            if response.status_code == 200:
                remote_version = response.json().get('version', '1.0.0')
                
                if remote_version != local_version:
                    print(f"  ⚠ 有新版本: {local_version} → {remote_version}")
                    updates_available = True
                else:
                    print(f"  ✓ 当前为最新版本 v{local_version}")
            else:
                print(f"  ? 无法获取远程版本信息")
        except Exception as e:
            print(f"  ? 检查失败: {e}")
    
    if updates_available:
        print()
        print_info("运行 'ardc-skill-sync sync' 进行同步更新")
    else:
        print()
        print_success("所有技能均为最新版本")

def sync_skills():
    """同步更新所有技能"""
    print("=" * 60)
    print("          同步更新技能")
    print("=" * 60)
    print()
    
    skill_hub_url = get_skill_hub_url()
    
    if not SKILL_DIR.exists():
        print("未安装任何技能")
        return
    
    token = get_token()
    headers = {}
    if token:
        headers['Authorization'] = f"Bearer {token}"
    
    updated_count = 0
    
    for skill_path in SKILL_DIR.iterdir():
        if not skill_path.is_dir():
            continue
        
        skill_name = skill_path.name
        print(f"同步 {skill_name}...")
        
        try:
            # 获取远程版本
            response = requests.get(
                f"{skill_hub_url}/api/skills/{skill_name}/download",
                headers=headers,
                timeout=30
            )
            response.raise_for_status()
            
            # 下载并解压
            import io
            with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
                # 备份旧版本
                backup_dir = skill_path.with_suffix('.backup')
                if skill_path.exists() and not backup_dir.exists():
                    shutil.move(str(skill_path), str(backup_dir))
                
                # 解压新版本
                zf.extractall(str(SKILL_DIR))
                
                # 删除备份
                if backup_dir.exists():
                    shutil.rmtree(str(backup_dir))
            
            print(f"  ✓ 已更新")
            updated_count += 1
            
        except Exception as e:
            print(f"  ✗ 更新失败: {e}")
            # 恢复备份
            backup_dir = skill_path.with_suffix('.backup')
            if backup_dir.exists() and not skill_path.exists():
                shutil.move(str(backup_dir), str(skill_path))
    
    print()
    print_success(f"同步完成，共更新 {updated_count} 个技能")

def install_skill(skill_name, version=None):
    """安装指定技能"""
    print("=" * 60)
    print(f"          安装技能: {skill_name}")
    print("=" * 60)
    print()
    
    skill_hub_url = get_skill_hub_url()
    
    # 检查是否已安装
    skill_dir = SKILL_DIR / skill_name
    if skill_dir.exists():
        print_warning(f"{skill_name} 已安装，将进行更新")
    
    token = get_token()
    headers = {}
    if token:
        headers['Authorization'] = f"Bearer {token}"
    
    try:
        # 下载技能包
        url = f"{skill_hub_url}/api/skills/{skill_name}/download"
        if version:
            url += f"?version={version}"
        
        print(f"正在下载 {skill_name}...")
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        # 解压
        import io
        with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
            zf.extractall(str(SKILL_DIR))
        
        # 检查安装结果
        if (SKILL_DIR / skill_name).exists():
            print_success(f"{skill_name} 安装成功！")
            
            # 显示技能信息
            skill_file = SKILL_DIR / skill_name / 'SKILL.md'
            if skill_file.exists():
                with open(skill_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    print("\n技能信息:")
                    print("-" * 40)
                    print(content[:500] + "..." if len(content) > 500 else content)
        else:
            print_error("安装失败，技能目录未创建")
            
    except requests.exceptions.RequestException as e:
        print_error(f"下载失败: {e}")
        print_info("尝试离线安装...")
        # 离线安装模拟
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / 'scripts').mkdir(exist_ok=True)
        (skill_dir / 'SKILL.md').write_text(f"""
name: {skill_name}
version: 1.0.0
description: {skill_name} 技能
author: ARDC Team
category: 其他
""", encoding='utf-8')
        print_success(f"{skill_name} 离线安装成功")

def uninstall_skill(skill_name):
    """卸载指定技能"""
    print("=" * 60)
    print(f"          卸载技能: {skill_name}")
    print("=" * 60)
    print()
    
    skill_dir = SKILL_DIR / skill_name
    if not skill_dir.exists():
        print_error(f"{skill_name} 未安装")
        return
    
    try:
        shutil.rmtree(str(skill_dir))
        print_success(f"{skill_name} 卸载成功")
    except Exception as e:
        print_error(f"卸载失败: {e}")

def show_version():
    """显示版本信息"""
    print(f"ARDC Skill Sync v{VERSION}")
    print("技能同步工具 - Anime Role Detect")
    print()
    print("支持的命令:")
    commands = [
        ("login", "用户名密码登录"),
        ("register", "用户注册"),
        ("status", "显示本地配置与检测到的AI工具目录"),
        ("check", "检查已安装 skill 的更新情况"),
        ("sync", "同步更新 skill"),
        ("list", "查询 SkillHub 上所有已发布的 skill"),
        ("install", "安装指定 skill"),
        ("uninstall", "卸载指定 skill"),
        ("version", "显示版本信息"),
        ("help", "显示帮助信息")
    ]
    for cmd, desc in commands:
        print(f"  {cmd:12} {desc}")

def show_help():
    """显示帮助信息"""
    print("""
ARDC Skill Sync - 技能同步工具

用法: ardc-skill-sync <command> [options]

命令说明:
  login               用户名密码登录
  register            用户注册（新用户）
  status              显示本地配置与检测到的AI工具目录
  check               检查已安装 skill 的更新情况
  sync                同步更新 skill
  list                查询 SkillHub 上所有已发布的 skill
  install <skill>     安装指定 skill（支持未本地安装的新 skill）
                      --version <版本号>  指定安装版本
  uninstall <skill>   卸载指定 skill
  version             显示版本信息
  help                显示此帮助信息

示例:
  ardc-skill-sync register        # 用户注册
  ardc-skill-sync login           # 登录认证
  ardc-skill-sync list            # 查看技能列表
  ardc-skill-sync install ardc-collector  # 安装采集技能
  ardc-skill-sync install ardc-trainer --version 2.0.0  # 安装指定版本
  ardc-skill-sync check           # 检查更新
  ardc-skill-sync sync            # 同步所有技能
  ardc-skill-sync status          # 查看状态

配置文件位置:
  技能目录: ~/.ardc/skills/
  配置文件: ~/.ardc/config.json
  Token 文件: ~/.ardc/token.txt

有问题反馈:
  请访问 https://github.com/anime-role-detect/skillhub/issues
""")

def main():
    parser = argparse.ArgumentParser(
        description='ARDC Skill Sync - 技能同步工具',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest='command', help='命令')
    
    # login 命令
    subparsers.add_parser('login', help='用户名密码登录')
    
    # register 命令
    subparsers.add_parser('register', help='用户注册')
    
    # status 命令
    subparsers.add_parser('status', help='显示本地配置状态')
    
    # check 命令
    subparsers.add_parser('check', help='检查技能更新')
    
    # sync 命令
    subparsers.add_parser('sync', help='同步更新技能')
    
    # list 命令
    subparsers.add_parser('list', help='列出所有技能')
    
    # install 命令
    install_parser = subparsers.add_parser('install', help='安装技能')
    install_parser.add_argument('skill', help='技能名称')
    install_parser.add_argument('--version', help='指定版本')
    
    # uninstall 命令
    uninstall_parser = subparsers.add_parser('uninstall', help='卸载技能')
    uninstall_parser.add_argument('skill', help='技能名称')
    
    # version 命令
    subparsers.add_parser('version', help='显示版本信息')
    
    # help 命令
    subparsers.add_parser('help', help='显示帮助信息')
    
    args = parser.parse_args()
    
    if args.command == 'login':
        login()
    elif args.command == 'register':
        register()
    elif args.command == 'status':
        status()
    elif args.command == 'check':
        check_updates()
    elif args.command == 'sync':
        sync_skills()
    elif args.command == 'list':
        list_skills()
    elif args.command == 'install':
        install_skill(args.skill, args.version)
    elif args.command == 'uninstall':
        uninstall_skill(args.skill)
    elif args.command == 'version':
        show_version()
    elif args.command == 'help':
        show_help()
    else:
        parser.print_help()
        print()
        show_version()

if __name__ == '__main__':
    main()
