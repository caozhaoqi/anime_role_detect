#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日志监控脚本 - 实时监控日志文件，发现异常及时告警
"""

import os
import sys
import re
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from pathlib import Path
from collections import defaultdict, deque
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.logging.global_logger import get_logger

logger = get_logger("log_monitor")


class LogPattern:
    """日志模式定义"""

    def __init__(
        self,
        name: str,
        pattern: str,
        severity: str = "medium",
        description: str = "",
        cooldown: int = 300  # 冷却时间（秒）
    ):
        self.name = name
        self.pattern = re.compile(pattern)
        self.severity = severity
        self.description = description
        self.cooldown = cooldown


class LogMonitor:
    """日志监控器"""

    # 默认日志模式
    DEFAULT_PATTERNS = [
        LogPattern(
            name="error",
            pattern=r"ERROR|Exception|Traceback",
            severity="high",
            description="发现错误日志",
            cooldown=60
        ),
        LogPattern(
            name="warning",
            pattern=r"WARNING|WARN",
            severity="medium",
            description="发现警告日志",
            cooldown=300
        ),
        LogPattern(
            name="timeout",
            pattern=r"timeout|超时|TimeoutError",
            severity="high",
            description="发现超时错误",
            cooldown=120
        ),
        LogPattern(
            name="connection_error",
            pattern=r"ConnectionError|连接失败|连接超时",
            severity="high",
            description="发现连接错误",
            cooldown=180
        ),
        LogPattern(
            name="memory_error",
            pattern=r"MemoryError|内存不足|OutOfMemory",
            severity="critical",
            description="发现内存错误",
            cooldown=60
        ),
        LogPattern(
            name="database_error",
            pattern=r"DatabaseError|数据库错误|SQL error",
            severity="high",
            description="发现数据库错误",
            cooldown=120
        ),
        LogPattern(
            name="authentication_error",
            pattern=r"AuthenticationError|认证失败|Unauthorized",
            severity="medium",
            description="发现认证错误",
            cooldown=300
        ),
        LogPattern(
            name="service_unavailable",
            pattern=r"ServiceUnavailable|服务不可用|503",
            severity="critical",
            description="发现服务不可用",
            cooldown=60
        ),
    ]

    def __init__(
        self,
        log_dir: Optional[Path] = None,
        patterns: Optional[List[LogPattern]] = None,
        alert_callback: Optional[Callable] = None
    ):
        """
        初始化日志监控器

        Args:
            log_dir: 日志目录
            patterns: 自定义日志模式
            alert_callback: 告警回调函数
        """
        self.log_dir = log_dir or (project_root / "logs")
        self.patterns = patterns or self.DEFAULT_PATTERNS
        self.alert_callback = alert_callback or self.default_alert_callback

        # 文件位置追踪
        self.file_positions = {}

        # 告警历史
        self.alert_history = defaultdict(lambda: datetime.min)

        # 统计信息
        self.stats = defaultdict(lambda: defaultdict(int))

        # 最近日志缓存（用于分析趋势）
        self.recent_logs = deque(maxlen=1000)

        logger.info(f"日志监控器初始化完成，监控目录: {self.log_dir}")

    def get_log_files(self) -> List[Path]:
        """获取需要监控的日志文件"""
        log_files = []

        # 监控所有.log文件
        for log_file in self.log_dir.glob("*.log"):
            if log_file.is_file():
                log_files.append(log_file)

        return sorted(log_files)

    def read_new_lines(self, log_file: Path) -> List[str]:
        """
        读取日志文件的新增内容

        Args:
            log_file: 日志文件路径

        Returns:
            新增的日志行列表
        """
        new_lines = []

        try:
            # 获取文件当前位置
            current_position = self.file_positions.get(str(log_file), 0)

            # 读取文件
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                # 移动到上次读取的位置
                f.seek(current_position)

                # 读取新内容
                for line in f:
                    new_lines.append(line.rstrip('\n'))

                # 更新位置
                self.file_positions[str(log_file)] = f.tell()

        except Exception as e:
            logger.error(f"读取日志文件 {log_file} 失败: {e}")

        return new_lines

    def analyze_log_line(self, line: str) -> List[Dict[str, any]]:
        """
        分析单行日志

        Args:
            line: 日志行

        Returns:
            匹配的模式列表
        """
        matches = []

        for pattern in self.patterns:
            if pattern.pattern.search(line):
                matches.append({
                    "pattern": pattern.name,
                    "severity": pattern.severity,
                    "description": pattern.description,
                    "line": line,
                    "timestamp": datetime.now().isoformat()
                })

        return matches

    def check_alert_cooldown(self, pattern_name: str) -> bool:
        """
        检查告警冷却时间

        Args:
            pattern_name: 模式名称

        Returns:
            是否可以发送告警
        """
        last_alert = self.alert_history[pattern_name]
        cooldown = next(
            (p.cooldown for p in self.patterns if p.name == pattern_name),
            300
        )

        return datetime.now() - last_alert > timedelta(seconds=cooldown)

    def send_alert(self, match: Dict[str, any]):
        """
        发送告警

        Args:
            match: 匹配结果
        """
        pattern_name = match["pattern"]

        if not self.check_alert_cooldown(pattern_name):
            return

        # 更新告警历史
        self.alert_history[pattern_name] = datetime.now()

        # 调用告警回调
        try:
            self.alert_callback(match)
        except Exception as e:
            logger.error(f"发送告警失败: {e}")

    def default_alert_callback(self, match: Dict[str, any]):
        """
        默认告警回调函数

        Args:
            match: 匹配结果
        """
        severity = match["severity"]
        pattern_name = match["pattern"]
        description = match["description"]
        line = match["line"]

        # 根据严重程度记录不同级别的日志
        if severity == "critical":
            logger.critical(f"[{pattern_name}] {description}: {line}")
        elif severity == "high":
            logger.error(f"[{pattern_name}] {description}: {line}")
        elif severity == "medium":
            logger.warning(f"[{pattern_name}] {description}: {line}")
        else:
            logger.info(f"[{pattern_name}] {description}: {line}")

    def monitor_once(self) -> Dict[str, any]:
        """
        执行一次监控检查

        Returns:
            监控结果
        """
        result = {
            "timestamp": datetime.now().isoformat(),
            "files_monitored": 0,
            "new_lines": 0,
            "patterns_matched": 0,
            "alerts_sent": 0,
            "matches": []
        }

        log_files = self.get_log_files()
        result["files_monitored"] = len(log_files)

        for log_file in log_files:
            new_lines = self.read_new_lines(log_file)
            result["new_lines"] += len(new_lines)

            for line in new_lines:
                # 添加到缓存
                self.recent_logs.append({
                    "file": str(log_file.name),
                    "line": line,
                    "timestamp": datetime.now().isoformat()
                })

                # 分析日志行
                matches = self.analyze_log_line(line)

                for match in matches:
                    result["patterns_matched"] += 1
                    result["matches"].append(match)

                    # 更新统计
                    self.stats[match["pattern"]][match["severity"]] += 1

                    # 发送告警
                    self.send_alert(match)
                    result["alerts_sent"] += 1

        return result

    def monitor_continuously(self, interval: int = 5):
        """
        持续监控日志

        Args:
            interval: 检查间隔（秒）
        """
        logger.info(f"开始持续监控日志，检查间隔: {interval}秒")

        try:
            while True:
                try:
                    result = self.monitor_once()

                    if result["alerts_sent"] > 0:
                        logger.info(
                            f"监控周期完成: 文件={result['files_monitored']}, "
                            f"新行={result['new_lines']}, "
                            f"匹配={result['patterns_matched']}, "
                            f"告警={result['alerts_sent']}"
                        )

                    time.sleep(interval)

                except KeyboardInterrupt:
                    logger.info("日志监控已停止")
                    break
                except Exception as e:
                    logger.error(f"监控过程中出错: {e}")
                    time.sleep(interval)

        except Exception as e:
            logger.error(f"持续监控失败: {e}")

    def get_statistics(self) -> Dict[str, any]:
        """
        获取统计信息

        Returns:
            统计信息
        """
        return {
            "timestamp": datetime.now().isoformat(),
            "patterns": dict(self.stats),
            "recent_logs_count": len(self.recent_logs),
            "alert_history": {
                pattern: last_alert.isoformat()
                for pattern, last_alert in self.alert_history.items()
            }
        }

    def save_statistics(self, output_file: Optional[Path] = None):
        """
        保存统计信息

        Args:
            output_file: 输出文件路径
        """
        if output_file is None:
            output_file = self.log_dir / "log_monitor_stats.json"

        try:
            stats = self.get_statistics()

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)

            logger.info(f"统计信息已保存: {output_file}")

        except Exception as e:
            logger.error(f"保存统计信息失败: {e}")

    def analyze_trends(self) -> Dict[str, any]:
        """
        分析日志趋势

        Returns:
            趋势分析结果
        """
        trends = {
            "timestamp": datetime.now().isoformat(),
            "error_rate": 0.0,
            "warning_rate": 0.0,
            "top_errors": [],
            "recent_activity": []
        }

        if not self.recent_logs:
            return trends

        # 计算错误率
        total_logs = len(self.recent_logs)
        error_count = sum(1 for log in self.recent_logs if "ERROR" in log["line"])
        warning_count = sum(1 for log in self.recent_logs if "WARNING" in log["line"])

        trends["error_rate"] = (error_count / total_logs) * 100 if total_logs > 0 else 0
        trends["warning_rate"] = (warning_count / total_logs) * 100 if total_logs > 0 else 0

        # 统计最常见的错误
        error_patterns = defaultdict(int)
        for log in self.recent_logs:
            if "ERROR" in log["line"]:
                # 提取错误类型
                match = re.search(r'ERROR\s+\|?\s*(\w+)', log["line"])
                if match:
                    error_type = match.group(1)
                    error_patterns[error_type] += 1

        trends["top_errors"] = [
            {"error": error, "count": count}
            for error, count in sorted(error_patterns.items(), key=lambda x: x[1], reverse=True)[:10]
        ]

        # 最近的活动
        trends["recent_activity"] = list(self.recent_logs)[-10:]

        return trends


class EmailAlertSender:
    """邮件告警发送器"""

    def __init__(
        self,
        smtp_server: str,
        smtp_port: int,
        username: str,
        password: str,
        from_addr: str,
        to_addrs: List[str]
    ):
        """
        初始化邮件告警发送器

        Args:
            smtp_server: SMTP服务器地址
            smtp_port: SMTP端口
            username: 用户名
            password: 密码
            from_addr: 发件人地址
            to_addrs: 收件人地址列表
        """
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_addr = from_addr
        self.to_addrs = to_addrs

    def send_alert(self, match: Dict[str, any]):
        """
        发送邮件告警

        Args:
            match: 匹配结果
        """
        try:
            # 创建邮件
            msg = MIMEMultipart()
            msg['From'] = self.from_addr
            msg['To'] = ', '.join(self.to_addrs)
            msg['Subject'] = f"[{match['severity'].upper()}] {match['description']}"

            # 邮件正文
            body = f"""
告警详情:
- 严重程度: {match['severity']}
- 模式: {match['pattern']}
- 描述: {match['description']}
- 时间: {match['timestamp']}
- 日志内容: {match['line']}
"""

            msg.attach(MIMEText(body, 'plain', 'utf-8'))

            # 发送邮件
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.send_message(msg)

            logger.info(f"邮件告警已发送: {match['description']}")

        except Exception as e:
            logger.error(f"发送邮件告警失败: {e}")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="日志监控脚本")
    parser.add_argument("--once", action="store_true", help="执行一次监控检查")
    parser.add_argument("--daemon", action="store_true", help="以守护进程模式运行")
    parser.add_argument("--interval", type=int, default=5, help="检查间隔（秒），默认5秒")
    parser.add_argument("--stats", action="store_true", help="显示统计信息")
    parser.add_argument("--trends", action="store_true", help="分析日志趋势")

    args = parser.parse_args()

    monitor = LogMonitor()

    if args.once:
        result = monitor.monitor_once()
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.stats:
        stats = monitor.get_statistics()
        print(json.dumps(stats, ensure_ascii=False, indent=2))

    elif args.trends:
        trends = monitor.analyze_trends()
        print(json.dumps(trends, ensure_ascii=False, indent=2))

    elif args.daemon:
        monitor.monitor_continuously(interval=args.interval)

    else:
        # 默认行为：执行一次监控检查
        result = monitor.monitor_once()

        print("\n=== 日志监控报告 ===")
        print(f"检查时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"监控文件数: {result['files_monitored']}")
        print(f"新增日志行: {result['new_lines']}")
        print(f"模式匹配数: {result['patterns_matched']}")
        print(f"发送告警数: {result['alerts_sent']}")

        if result["matches"]:
            print("\n=== 匹配的日志模式 ===")
            for match in result["matches"]:
                print(f"[{match['severity'].upper()}] {match['description']}")
                print(f"  {match['line'][:100]}...")


if __name__ == "__main__":
    main()