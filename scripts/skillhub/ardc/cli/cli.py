#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Anime Role Detect 命令行工具
提供技能管理、版本控制和工作流编排功能
"""

import argparse
import sys

from ardc.store.metadata import SkillMetadata
from ardc.store.registry import SkillRegistry
from ardc.store.index import SkillIndex
from ardc.version.manager import VersionManager


class ARDCLI:
    """ARD命令行工具"""

    def __init__(self):
        self.registry = SkillRegistry()
        self.index = SkillIndex()
        self.version_manager = VersionManager()

        self.parser = argparse.ArgumentParser(
            prog="ardc",
            description="Anime Role Detect 命令行工具",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
示例:
  ardc skill install ardc-collector
  ardc skill search 采集
  ardc skill list --installed
  ardc version list ardc-collector
            """,
        )

        self._setup_subparsers()

    def _setup_subparsers(self):
        subparsers = self.parser.add_subparsers(dest="command", help="可用命令")

        skill_parser = subparsers.add_parser("skill", help="技能管理")
        skill_subparsers = skill_parser.add_subparsers(dest="skill_command")

        install_parser = skill_subparsers.add_parser("install", help="安装技能")
        install_parser.add_argument("skill_id", help="技能ID")
        install_parser.add_argument("--version", "-v", help="指定版本")

        uninstall_parser = skill_subparsers.add_parser("uninstall", help="卸载技能")
        uninstall_parser.add_argument("skill_id", help="技能ID")

        list_parser = skill_subparsers.add_parser("list", help="列出技能")
        list_parser.add_argument("--installed", "-i", action="store_true", help="仅显示已安装技能")
        list_parser.add_argument("--category", "-c", help="按分类筛选")

        search_parser = skill_subparsers.add_parser("search", help="搜索技能")
        search_parser.add_argument("keyword", help="搜索关键词")
        search_parser.add_argument("--category", "-c", help="按分类筛选")
        search_parser.add_argument("--limit", "-l", type=int, default=20, help="返回数量限制")

        register_parser = skill_subparsers.add_parser("register", help="注册技能")
        register_parser.add_argument("--name", required=True, help="技能名称")
        register_parser.add_argument("--id", required=True, help="技能ID")
        register_parser.add_argument("--version", required=True, help="版本号")
        register_parser.add_argument("--description", "-d", help="技能描述")
        register_parser.add_argument("--author", required=True, help="作者")
        register_parser.add_argument(
            "--category",
            required=True,
            choices=[
                "collector",
                "cleaner",
                "classifier",
                "trainer",
                "search",
                "analyzer",
                "utility",
            ],
            help="技能分类",
        )
        register_parser.add_argument("--entry-point", required=True, help="入口文件路径")
        register_parser.add_argument("--tag", "-t", action="append", help="标签")
        register_parser.add_argument("--release-notes", help="版本更新说明")

        enable_parser = skill_subparsers.add_parser("enable", help="启用技能")
        enable_parser.add_argument("skill_id", help="技能ID")

        disable_parser = skill_subparsers.add_parser("disable", help="禁用技能")
        disable_parser.add_argument("skill_id", help="技能ID")

        version_parser = subparsers.add_parser("version", help="版本管理")
        version_subparsers = version_parser.add_subparsers(dest="version_command")

        vlist_parser = version_subparsers.add_parser("list", help="列出版本")
        vlist_parser.add_argument("skill_id", help="技能ID")

        rollback_parser = version_subparsers.add_parser("rollback", help="版本回滚")
        rollback_parser.add_argument("skill_id", help="技能ID")
        rollback_parser.add_argument("target_version", help="目标版本")

        history_parser = version_subparsers.add_parser("history", help="版本历史")
        history_parser.add_argument("skill_id", help="技能ID")

        info_parser = subparsers.add_parser("info", help="查看技能详情")
        info_parser.add_argument("skill_id", help="技能ID")
        info_parser.add_argument("--version", "-v", help="指定版本")

        stats_parser = subparsers.add_parser("stats", help="查看统计信息")

    def run(self):
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
        if args.command == "skill":
            self._handle_skill_command(args)
        elif args.command == "version":
            self._handle_version_command(args)
        elif args.command == "info":
            self._handle_info_command(args)
        elif args.command == "stats":
            self._handle_stats_command(args)

    def _handle_skill_command(self, args):
        if args.skill_command == "install":
            self._install_skill(args.skill_id, args.version)
        elif args.skill_command == "uninstall":
            self._uninstall_skill(args.skill_id)
        elif args.skill_command == "list":
            self._list_skills(args.installed, args.category)
        elif args.skill_command == "search":
            self._search_skills(args.keyword, args.category, args.limit)
        elif args.skill_command == "register":
            self._register_skill(args)
        elif args.skill_command == "enable":
            self._enable_skill(args.skill_id, True)
        elif args.skill_command == "disable":
            self._enable_skill(args.skill_id, False)

    def _handle_version_command(self, args):
        if args.version_command == "list":
            self._list_versions(args.skill_id)
        elif args.version_command == "rollback":
            self._rollback_version(args.skill_id, args.target_version)
        elif args.version_command == "history":
            self._show_version_history(args.skill_id)

    def _handle_info_command(self, args):
        self._show_skill_info(args.skill_id, args.version)

    def _handle_stats_command(self, args):
        self._show_stats()

    def _install_skill(self, skill_id, version=None):
        if self.registry.install_skill(skill_id, version):
            metadata = self.registry.get_skill_by_version(skill_id, version)
            if metadata:
                self.index.add_skill(metadata)

    def _uninstall_skill(self, skill_id):
        if self.registry.uninstall_skill(skill_id):
            self.index.remove_skill(skill_id)

    def _list_skills(self, installed_only=False, category=None):
        if installed_only:
            skills = self.registry.list_installed_skills()
            for installed in skills:
                status = "启用" if installed.enabled else "禁用"
                print(
                    f"{installed.metadata.id} ({installed.metadata.version}) - {installed.metadata.name} [{status}]"
                )
        else:
            skills = (
                self.index.get_by_category(category) if category else self.index.get_all_skills()
            )
            for skill in skills:
                print(f"{skill.id} ({skill.version}) - {skill.name}")
                if skill.description:
                    print(f"  {skill.description}")
                print(f"  分类: {skill.category} | 状态: {skill.status}")

    def _search_skills(self, keyword, category=None, limit=20):
        results = self.index.search(keyword, category, limit=limit)
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

    def _register_skill(self, args):
        metadata = SkillMetadata(
            id=args.id,
            name=args.name,
            version=args.version,
            description=args.description or "",
            author=args.author,
            category=args.category,
            entry_point=args.entry_point,
            tags=args.tag or [],
        )

        if self.registry.register_skill(metadata, args.release_notes):
            self.index.add_skill(metadata)
            self.version_manager.release_version(metadata, args.release_notes)
            print(f"技能 {args.id} 注册成功")

    def _enable_skill(self, skill_id, enabled):
        self.registry.enable_skill(skill_id, enabled)

    def _list_versions(self, skill_id):
        versions = self.version_manager.list_versions(skill_id)
        if not versions:
            print(f"技能 {skill_id} 没有版本")
            return

        print(f"技能 {skill_id} 的版本列表:")
        for v in versions:
            print(f"  {v.version} ({v.metadata.status})")
            if v.release_notes:
                print(f"    更新说明: {v.release_notes}")

    def _rollback_version(self, skill_id, target_version):
        self.version_manager.rollback(skill_id, target_version)

    def _show_version_history(self, skill_id):
        history = self.version_manager.get_version_history(skill_id)
        if not history:
            print(f"技能 {skill_id} 没有版本历史")
            return

        print(f"技能 {skill_id} 的版本历史:")
        for record in history:
            if record["type"] == "release":
                print(f"  [发布] {record['version']} - {record['timestamp']}")
            elif record["type"] == "rollback":
                print(f"  [回滚] {record['from_version']} -> {record['to_version']}")

    def _show_skill_info(self, skill_id, version=None):
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
        print(f"  分类: {metadata.category}")
        print(f"  状态: {metadata.status}")
        print(f"  入口文件: {metadata.entry_point}")
        print(f"  标签: {', '.join(metadata.tags)}")

        installed = self.registry.get_installed_skill(skill_id)
        if installed:
            print(f"  安装状态: 已安装")
            print(f"  安装路径: {installed.install_path}")

    def _show_stats(self):
        stats = self.index.get_statistics()

        print("技能仓库统计:")
        print(f"  技能总数: {stats['total_skills']}")
        print(f"  分类数量: {stats['total_categories']}")
        print(f"  标签数量: {stats['total_tags']}")

        print("\n状态分布:")
        for status, count in stats.get("status_counts", {}).items():
            print(f"  {status}: {count}")


def main():
    cli = ARDCLI()
    cli.run()


if __name__ == "__main__":
    main()
