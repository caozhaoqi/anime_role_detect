#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mac 系统存储空间清理工具
智能清理系统缓存、日志和临时文件
"""

import os
import shutil
import sys
import datetime
from pathlib import Path
from typing import Dict, List, Optional

# --- 配置清理目标 ---
# 警告：请仔细阅读路径后再运行
CLEAN_TARGETS: Dict[str, str] = {
    "User Caches": "~/Library/Caches",
    "System Logs": "~/Library/Logs",
    "Xcode DerivedData": "~/Library/Developer/Xcode/DerivedData",
    "Xcode iOS DeviceSupport": "~/Library/Developer/Xcode/iOS DeviceSupport",
    "Homebrew Cache": "~/Library/Caches/Homebrew",
    "CocoaPods Cache": "~/Library/Caches/CocoaPods",
    "Telegram Files": "~/Library/Group Containers/6N38VVP5K3.org.telegram.messenger/Context/Documents",
    "Spotify Cache": "~/Library/Application Support/Spotify/PersistentCache",
    "Trash": "~/.Trash",
    "npm Cache": "~/.npm",
    "pip Cache": "~/Library/Caches/pip",
    "Docker Images": "~/Library/Containers/com.docker.docker/Data/vms/0",
    "VSCode Cache": "~/Library/Application Support/Code/Cache",
    "VSCode CachedData": "~/Library/Application Support/Code/CachedData",
}

# 针对下载文件夹：删除超过指定天数未访问的文件
DOWNLOADS_DIR: str = "~/Downloads"
DOWNLOADS_MAX_AGE_DAYS: int = 30

# 危险操作确认提示
DANGEROUS_PATHS: List[str] = ["~/.Trash"]

class MacCleaner:
    def __init__(self, dry_run: bool = True):
        self.dry_run = dry_run
        self.total_freed = 0
        self.cleaned_items = []
        self.skipped_items = []
        
    def get_size(self, path: Path) -> int:
        """计算路径大小（字节）"""
        if not path.exists():
            return 0
        if path.is_file():
            return path.stat().st_size
        try:
            return sum(f.stat().st_size for f in path.glob('**/*') if f.is_file())
        except PermissionError:
            return 0
        
    def format_size(self, size: int) -> str:
        """格式化字节单位"""
        if size == 0:
            return "0 B"
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size < 1024:
                return f"{size:.2f} {unit}"
            size /= 1024
        return f"{size:.2f} PB"
    
    def confirm_action(self, message: str) -> bool:
        """交互式确认"""
        while True:
            response = input(f"{message} (y/N): ").strip().lower()
            if response in ['y', 'yes']:
                return True
            elif response in ['n', 'no', '']:
                return False
    
    def clean_path(self, name: str, path_str: str) -> None:
        """执行清理单个路径"""
        full_path = Path(os.path.expanduser(path_str))
        
        if not full_path.exists():
            self.skipped_items.append((name, "路径不存在"))
            return
        
        size = self.get_size(full_path)
        if size == 0:
            self.skipped_items.append((name, "为空"))
            return
        
        print(f"[*] 发现 {name}: {self.format_size(size)}")
        
        # 危险操作需要额外确认
        if path_str in DANGEROUS_PATHS:
            if not self.confirm_action(f"    ⚠️  确定要清空 {name} 吗？"):
                self.skipped_items.append((name, "用户取消"))
                return
        
        if not self.dry_run:
            try:
                if full_path.is_file():
                    full_path.unlink()
                else:
                    # 仅删除目录下的内容，保留目录本身
                    for item in full_path.iterdir():
                        try:
                            if item.is_file():
                                item.unlink()
                            elif item.is_dir():
                                shutil.rmtree(item)
                        except PermissionError:
                            print(f"    ⚠️  跳过 (权限不足): {item.name}")
                self.total_freed += size
                self.cleaned_items.append((name, size))
                print(f"    ✅ 已清理")
            except Exception as e:
                self.skipped_items.append((name, str(e)))
                print(f"    ❌ 清理失败: {e}")
        else:
            self.total_freed += size
            self.cleaned_items.append((name, size))
    
    def clean_downloads(self) -> None:
        """清理下载文件夹中老旧文件"""
        print(f"[*] 扫描下载文件夹 (>{DOWNLOADS_MAX_AGE_DAYS}天未访问)...")
        downloads = Path(os.path.expanduser(DOWNLOADS_DIR))
        
        if not downloads.exists():
            self.skipped_items.append(("Downloads", "路径不存在"))
            return
        
        now = datetime.datetime.now()
        count = 0
        total_size = 0
        
        for item in downloads.iterdir():
            if item.is_file():
                try:
                    mtime = datetime.datetime.fromtimestamp(item.stat().st_mtime)
                    if (now - mtime).days > DOWNLOADS_MAX_AGE_DAYS:
                        size = item.stat().st_size
                        total_size += size
                        count += 1
                        if not self.dry_run:
                            item.unlink()
                except (PermissionError, OSError):
                    pass
        
        if count > 0:
            self.total_freed += total_size
            self.cleaned_items.append((f"Downloads ({count}个文件)", total_size))
            print(f"    ✅ 已清理 {count} 个过期文件: {self.format_size(total_size)}")
        else:
            self.skipped_items.append(("Downloads", "无过期文件"))
            print(f"    无过期文件")
    
    def clean_logs(self) -> None:
        """清理系统日志文件"""
        print("[*] 清理系统日志文件...")
        log_paths = [
            "/var/log",
            "~/Library/Logs",
        ]
        
        total_cleaned = 0
        for path_str in log_paths:
            path = Path(os.path.expanduser(path_str))
            if not path.exists():
                continue
            
            for log_file in path.glob("**/*.log"):
                try:
                    size = log_file.stat().st_size
                    total_cleaned += size
                    if not self.dry_run:
                        log_file.unlink()
                except (PermissionError, OSError):
                    pass
        
        if total_cleaned > 0:
            self.total_freed += total_cleaned
            self.cleaned_items.append(("系统日志", total_cleaned))
            print(f"    ✅ 已清理日志: {self.format_size(total_cleaned)}")
    
    def generate_report(self) -> str:
        """生成清理报告"""
        report = ["\n" + "="*50]
        report.append("清理报告")
        report.append("="*50)
        
        report.append(f"\n📊 清理模式: {'预览模式' if self.dry_run else '强制清理模式'}")
        report.append(f"\n✅ 已清理项目 ({len(self.cleaned_items)}):")
        for name, size in self.cleaned_items:
            report.append(f"   • {name}: {self.format_size(size)}")
        
        if self.skipped_items:
            report.append(f"\n⚠️  跳过项目 ({len(self.skipped_items)}):")
            for name, reason in self.skipped_items:
                report.append(f"   • {name}: {reason}")
        
        status = "预计可释放" if self.dry_run else "总计已释放"
        report.append(f"\n{status} 空间: {self.format_size(self.total_freed)}")
        
        if self.dry_run:
            report.append("\n💡 提示：确认无误后，使用 --force 参数执行真实删除。")
        
        return "\n".join(report)
    
    def run(self) -> None:
        """主执行函数"""
        mode = "【预览模式 - 不会真的删除】" if self.dry_run else "【⚠️ 强制清理模式 ⚠️】"
        print(f"=== Mac 系统清理助手 {mode} ===")
        print("="*50)
        
        # 清理预定义目标
        for name, path in CLEAN_TARGETS.items():
            self.clean_path(name, path)
        
        # 清理下载文件夹
        self.clean_downloads()
        
        # 清理日志
        self.clean_logs()
        
        # 输出报告
        print(self.generate_report())

def main():
    # 解析命令行参数
    dry_run = True
    if len(sys.argv) > 1 and sys.argv[1] == "--force":
        dry_run = False
    
    # 创建清理器并运行
    cleaner = MacCleaner(dry_run=dry_run)
    cleaner.run()

if __name__ == "__main__":
    main()
