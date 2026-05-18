#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Anime Role Detect 命令行工具
提供技能管理、版本控制和工作流编排功能
"""

import argparse
import sys
import json
from datetime import datetime

# 添加技能模块路径
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))

from skills.store.metadata import SkillMetadata
from skills.store.registry import SkillRegistry
from skills.store.index import SkillIndex
from skills.version.manager import VersionManager


class ARDCLI:
    """ARD命令行工具"""
    
    def __init__(self):
        self.registry = SkillRegistry()
        self.index = SkillIndex()
        self.version_manager = VersionManager()
        
        self.parser = argparse.ArgumentParser(
            prog='ardc',
            description='Anime Role Detect 命令行工具',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
示例:
  ardc skill install ardc-collector
  ardc skill search 采集
  ardc skill list --installed
  ardc version list ardc-collector
            """
        )
        
        self._setup_subparsers()
    
    def _setup_subparsers(self):
        """设置子命令"""
        subparsers = self.parser.add_subparsers(dest='command', help='可用命令')
        
        # skill 命令
        skill_parser = subparsers.add_parser('skill', help='技能管理')
        skill_subparsers = skill_parser.add_subparsers(dest='skill_command')
        
        # skill install
        install_parser = skill_subparsers.add_parser('install', help='安装技能')
        install_parser.add_argument('skill_id', help='技能ID')
        install_parser.add_argument('--version', '-v', help='指定版本')
        
        # skill uninstall
        uninstall_parser = skill_subparsers.add_parser('uninstall', help='卸载技能')
        uninstall_parser.add_argument('skill_id', help='技能ID')
        
        # skill list
        list_parser = skill_subparsers.add_parser('list', help='列出技能')
        list_parser.add_argument('--installed', '-i', action='store_true', help='仅显示已安装技能')
        list_parser.add_argument('--category', '-c', help='按分类筛选')
        
        # skill search
        search_parser = skill_subparsers.add_parser('search', help='搜索技能')
        search_parser.add_argument('keyword', help='搜索关键词')
        search_parser.add_argument('--category', '-c', help='按分类筛选')
        search_parser.add_argument('--status', '-s', help='按状态筛选')
        search_parser.add_argument('--limit', '-l', type=int, default=20, help='返回数量限制')
        
        # skill register
        register_parser = skill_subparsers.add_parser('register', help='注册技能')
        register_parser.add_argument('--name', required=True, help='技能名称')
        register_parser.add_argument('--id', required=True, help='技能ID')
        register_parser.add_argument('--version', required=True, help='版本号')
        register_parser.add_argument('--description', '-d', help='技能描述')
        register_parser.add_argument('--author', required=True, help='作者')
        register_parser.add_argument('--category', required=True, 
                                    choices=['collector', 'cleaner', 'classifier', 'trainer', 'search', 'analyzer', 'utility'],
                                    help='技能分类')
        register_parser.add_argument('--entry-point', required=True, help='入口文件路径')
        register_parser.add_argument('--tag', '-t', action='append', help='标签（可多次使用）')
        register_parser.add_argument('--release-notes', help='版本更新说明')
        
        # skill enable/disable
        enable_parser = skill_subparsers.add_parser('enable', help='启用技能')
        enable_parser.add_argument('skill_id', help='技能ID')
        
        disable_parser = skill_subparsers.add_parser('disable', help='禁用技能')
        disable_parser.add_argument('skill_id', help='技能ID')
        
        # version 命令
        version_parser = subparsers.add_parser('version', help='版本管理')
        version_subparsers = version_parser.add_subparsers(dest='version_command')
        
        # version list
        vlist_parser = version_subparsers.add_parser('list', help='列出版本')
        vlist_parser.add_argument('skill_id', help='技能ID')
        
        # version rollback
        rollback_parser = version_subparsers.add_parser('rollback', help='版本回滚')
        rollback_parser.add_argument('skill_id', help='技能ID')
        rollback_parser.add_argument('target_version', help='目标版本')
        
        # version history
        history_parser = version_subparsers.add_parser('history', help='版本历史')
        history_parser.add_argument('skill_id', help='技能ID')
        
        # version compare
        compare_parser = version_subparsers.add_parser('compare', help='比较版本')
        compare_parser.add_argument('version1', help='版本1')
        compare_parser.add_argument('version2', help='版本2')
        
        # info 命令
        info_parser = subparsers.add_parser('info', help='查看技能详情')
        info_parser.add_argument('skill_id', help='技能ID')
        info_parser.add_argument('--version', '-v', help='指定版本')
        
        # stats 命令
        stats_parser = subparsers.add_parser('stats', help='查看统计信息')
    
    def run(self):
        """运行CLI"""
        args = self.parser.parse_args()
        
        if args.command is None:
            self.parser.print_help()
            return
        
        try:
            self._execute_command(args)
        except Exception as e:
            print(f"错误: {e}", file=sys.stderr)
            sys.exit(1)
    
    def _execute_command(self, args):
        """执行命令"""
        if args.command == 'skill':
            self._handle_skill_command(args)
        elif args.command == 'version':
            self._handle_version_command(args)
        elif args.command == 'info':
            self._handle_info_command(args)
        elif args.command == 'stats':
            self._handle_stats_command(args)
        else:
            print(f"未知命令: {args.command}")
            self.parser.print_help()
    
    def _handle_skill_command(self, args):
        """处理技能命令"""
        if args.skill_command == 'install':
            self._install_skill(args.skill_id, args.version)
        elif args.skill_command == 'uninstall':
            self._uninstall_skill(args.skill_id)
        elif args.skill_command == 'list':
            self._list_skills(args.installed, args.category)
        elif args.skill_command == 'search':
            self._search_skills(args.keyword, args.category, args.status, args.limit)
        elif args.skill_command == 'register':
            self._register_skill(args)
        elif args.skill_command == 'enable':
            self._enable_skill(args.skill_id, True)
        elif args.skill_command == 'disable':
            self._enable_skill(args.skill_id, False)
        else:
            print(f"未知技能命令: {args.skill_command}")
    
    def _handle_version_command(self, args):
        """处理版本命令"""
        if args.version_command == 'list':
            self._list_versions(args.skill_id)
        elif args.version_command == 'rollback':
            self._rollback_version(args.skill_id, args.target_version)
        elif args.version_command == 'history':
            self._show_version_history(args.skill_id)
        elif args.version_command == 'compare':
            self._compare_versions(args.version1, args.version2)
        else:
            print(f"未知版本命令: {args.version_command}")
    
    def _handle_info_command(self, args):
        """处理信息命令"""
        self._show_skill_info(args.skill_id, args.version)
    
    def _handle_stats_command(self, args):
        """处理统计命令"""
        self._show_stats()
    
    # 技能操作方法
    def _install_skill(self, skill_id, version=None):
        """安装技能"""
        if self.registry.install_skill(skill_id, version):
            # 同时添加到索引
            metadata = self.registry.get_skill_by_version(skill_id, version)
            if metadata:
                self.index.add_skill(metadata)
    
    def _uninstall_skill(self, skill_id):
        """卸载技能"""
        if self.registry.uninstall_skill(skill_id):
            # 从索引中移除
            self.index.remove_skill(skill_id)
    
    def _list_skills(self, installed_only=False, category=None):
        """列出技能"""
        if installed_only:
            skills = self.registry.list_installed_skills()
            for installed in skills:
                status = "启用" if installed.enabled else "禁用"
                print(f"{installed.metadata.id} ({installed.metadata.version}) - {installed.metadata.name} [{status}]")
        else:
            skills = self.index.get_by_category(category) if category else self.index.get_all_skills()
            for skill in skills:
                print(f"{skill.id} ({skill.version}) - {skill.name}")
                if skill.description:
                    print(f"  {skill.description}")
                print(f"  分类: {skill.category} | 状态: {skill.status}")
    
    def _search_skills(self, keyword, category=None, status=None, limit=20):
        """搜索技能"""
        results = self.index.search(keyword, category, status, limit)
        if not results:
            print("未找到匹配的技能")
            return
        
        print(f"找到 {len(results)} 个匹配技能:")
        for i, skill in enumerate(results, 1):
            print(f"{i}. {skill.id} ({skill.version})")
            print(f"   名称: {skill.name}")
            if skill.description:
                print(f"   描述: {skill.description}")
            print(f"   分类: {skill.category} | 状态: {skill.status}")
            if skill.tags:
                print(f"   标签: {', '.join(skill.tags)}")
    
    def _register_skill(self, args):
        """注册技能"""
        metadata = SkillMetadata(
            id=args.id,
            name=args.name,
            version=args.version,
            description=args.description or "",
            author=args.author,
            category=args.category,
            entry_point=args.entry_point,
            tags=args.tag or []
        )
        
        if self.registry.register_skill(metadata, args.release_notes):
            self.index.add_skill(metadata)
            self.version_manager.release_version(metadata, args.release_notes)
            print(f"技能 {args.id} 注册成功")
    
    def _enable_skill(self, skill_id, enabled):
        """启用/禁用技能"""
        self.registry.enable_skill(skill_id, enabled)
    
    # 版本操作方法
    def _list_versions(self, skill_id):
        """列出版本"""
        versions = self.version_manager.list_versions(skill_id)
        if not versions:
            print(f"技能 {skill_id} 没有版本")
            return
        
        print(f"技能 {skill_id} 的版本列表:")
        for v in versions:
            print(f"  {v.version} ({v.metadata.status})")
            if v.release_notes:
                print(f"    更新说明: {v.release_notes}")
            print(f"    发布时间: {v.released_at.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"    下载次数: {v.download_count}")
    
    def _rollback_version(self, skill_id, target_version):
        """版本回滚"""
        self.version_manager.rollback(skill_id, target_version)
    
    def _show_version_history(self, skill_id):
        """显示版本历史"""
        history = self.version_manager.get_version_history(skill_id)
        if not history:
            print(f"技能 {skill_id} 没有版本历史")
            return
        
        print(f"技能 {skill_id} 的版本历史:")
        for record in history:
            if record["type"] == "release":
                print(f"  [发布] {record['version']} - {record['timestamp']}")
                if record.get('notes'):
                    print(f"    更新说明: {record['notes']}")
            elif record["type"] == "rollback":
                print(f"  [回滚] {record['from_version']} -> {record['to_version']} - {record['timestamp']}")
    
    def _compare_versions(self, version1, version2):
        """比较版本"""
        result = self.version_manager.compare_versions(version1, version2)
        if result < 0:
            print(f"{version1} < {version2}")
        elif result > 0:
            print(f"{version1} > {version2}")
        else:
            print(f"{version1} == {version2}")
    
    # 信息展示方法
    def _show_skill_info(self, skill_id, version=None):
        """显示技能详情"""
        metadata = self.registry.get_skill_by_version(skill_id, version)
        if not metadata:
            print(f"未找到技能 {skill_id}")
            return
        
        print(f"技能详情:")
        print(f"  ID: {metadata.id}")
        print(f"  名称: {metadata.name}")
        print(f"  版本: {metadata.version}")
        print(f"  描述: {metadata.description}")
        print(f"  作者: {metadata.author}")
        if metadata.author_url:
            print(f"  作者主页: {metadata.author_url}")
        print(f"  分类: {metadata.category}")
        print(f"  状态: {metadata.status}")
        print(f"  入口文件: {metadata.entry_point}")
        print(f"  运行时: {metadata.runtime}")
        print(f"  标签: {', '.join(metadata.tags)}")
        print(f"  创建时间: {metadata.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  更新时间: {metadata.updated_at.strftime('%Y-%m-%d %H:%M:%S')}")
        
        if metadata.dependencies:
            print(f"  依赖:")
            for dep in metadata.dependencies:
                opt = "(可选)" if dep.optional else ""
                print(f"    - {dep.skill_id} {dep.version} {opt}")
        
        # 检查是否已安装
        installed = self.registry.get_installed_skill(skill_id)
        if installed:
            print(f"  安装状态: 已安装")
            print(f"  安装路径: {installed.install_path}")
            print(f"  安装时间: {installed.installed_at.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"  启用状态: {'启用' if installed.enabled else '禁用'}")
    
    def _show_stats(self):
        """显示统计信息"""
        stats = self.index.get_statistics()
        
        print("技能仓库统计:")
        print(f"  技能总数: {stats['total_skills']}")
        print(f"  分类数量: {stats['total_categories']}")
        print(f"  标签数量: {stats['total_tags']}")
        print(f"  最后更新: {stats['last_updated']}")
        
        print("\n状态分布:")
        for status, count in stats.get("status_counts", {}).items():
            print(f"  {status}: {count}")
        
        print("\n分类分布:")
        for category, count in stats.get("categories", {}).items():
            print(f"  {category}: {count}")


def main():
    """主入口"""
    cli = ARDCLI()
    cli.run()


if __name__ == "__main__":
    main()