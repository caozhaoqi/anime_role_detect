#!/usr/bin/env python3
"""
日志分析脚本
用于分析分类日志，生成统计报告
"""
import os
import json
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter

class LogAnalyzer:
    """日志分析器"""
    
    def __init__(self, log_dir='./logs'):
        """初始化日志分析器
        
        Args:
            log_dir: 日志目录
        """
        self.log_dir = log_dir
        self.logs = []
    
    def load_logs(self):
        """加载日志文件
        
        Returns:
            list: 日志数据列表
        """
        print(f"正在加载日志目录: {self.log_dir}")
        
        logs = []
        
        try:
            log_files = [f for f in os.listdir(self.log_dir) if f.endswith('.json')]
            print(f"找到 {len(log_files)} 个日志文件")
            
            for log_file in log_files:
                log_path = os.path.join(self.log_dir, log_file)
                try:
                    with open(log_path, 'r', encoding='utf-8') as f:
                        log_data = json.load(f)
                        # 添加文件名信息
                        log_data['log_file'] = log_file
                        logs.append(log_data)
                except Exception as e:
                    print(f"加载日志文件失败 {log_file}: {e}")
            
            self.logs = logs
            print(f"成功加载 {len(logs)} 条日志")
            return logs
        except Exception as e:
            print(f"加载日志失败: {e}")
            return []
    
    def analyze_similarity(self):
        """分析相似度分布
        
        Returns:
            dict: 相似度统计信息
        """
        if not self.logs:
            return {}
        
        similarities = [log.get('similarity', 0) for log in self.logs]
        
        stats = {
            'mean': np.mean(similarities),
            'median': np.median(similarities),
            'min': np.min(similarities),
            'max': np.max(similarities),
            'std': np.std(similarities)
        }
        
        print("\n=== 相似度分析 ===")
        print(f"平均相似度: {stats['mean']:.4f}")
        print(f"中位数相似度: {stats['median']:.4f}")
        print(f"最小相似度: {stats['min']:.4f}")
        print(f"最大相似度: {stats['max']:.4f}")
        print(f"标准差: {stats['std']:.4f}")
        
        return stats
    
    def analyze_role_distribution(self):
        """分析角色分布
        
        Returns:
            dict: 角色分布统计
        """
        if not self.logs:
            return {}
        
        roles = [log.get('role', '未知') for log in self.logs]
        role_counter = Counter(roles)
        
        print("\n=== 角色分布分析 ===")
        print(f"总角色数: {len(role_counter)}")
        print("\nTop 10 最常见角色:")
        
        for role, count in role_counter.most_common(10):
            percentage = (count / len(self.logs)) * 100
            print(f"{role}: {count}次 ({percentage:.2f}%)")
        
        return dict(role_counter)
    
    def analyze_temporal_trend(self):
        """分析时间趋势
        
        Returns:
            dict: 时间趋势数据
        """
        if not self.logs:
            return {}
        
        # 提取时间戳并按日期分组
        date_counts = Counter()
        
        for log in self.logs:
            timestamp = log.get('timestamp')
            if timestamp:
                try:
                    date = datetime.fromisoformat(timestamp).strftime('%Y-%m-%d')
                    date_counts[date] += 1
                except Exception:
                    pass
        
        print("\n=== 时间趋势分析 ===")
        print("每日分类次数:")
        
        for date, count in sorted(date_counts.items()):
            print(f"{date}: {count}次")
        
        return dict(date_counts)
    
    def analyze_model_usage(self):
        """分析模型使用情况
        
        Returns:
            dict: 模型使用统计
        """
        if not self.logs:
            return {}
        
        model_counts = Counter()
        
        for log in self.logs:
            metadata = log.get('metadata', {})
            mode = metadata.get('mode', '未知')
            model_counts[mode] += 1
        
        print("\n=== 模型使用分析 ===")
        print("各模型使用次数:")
        
        for mode, count in model_counts.items():
            percentage = (count / len(self.logs)) * 100
            print(f"{mode}: {count}次 ({percentage:.2f}%)")
        
        return dict(model_counts)
    
    def generate_report(self, output_file='log_analysis_report.txt'):
        """生成分析报告
        
        Args:
            output_file: 输出文件路径
        """
        if not self.logs:
            print("没有日志数据可分析")
            return
        
        report_lines = []
        report_lines.append("# 日志分析报告")
        report_lines.append(f"\n生成时间: {datetime.now().isoformat()}")
        report_lines.append(f"分析日志数: {len(self.logs)}")
        report_lines.append(f"日志目录: {self.log_dir}")
        
        # 相似度分析
        report_lines.append("\n## 相似度分析")
        similarity_stats = self.analyze_similarity()
        for key, value in similarity_stats.items():
            report_lines.append(f"{key}: {value:.4f}")
        
        # 角色分布
        report_lines.append("\n## 角色分布")
        role_dist = self.analyze_role_distribution()
        report_lines.append(f"总角色数: {len(role_dist)}")
        report_lines.append("\nTop 10 最常见角色:")
        role_counter = Counter(role_dist)
        for role, count in role_counter.most_common(10):
            percentage = (count / len(self.logs)) * 100
            report_lines.append(f"{role}: {count}次 ({percentage:.2f}%)")
        
        # 时间趋势
        report_lines.append("\n## 时间趋势")
        temporal_trend = self.analyze_temporal_trend()
        for date, count in sorted(temporal_trend.items()):
            report_lines.append(f"{date}: {count}次")
        
        # 模型使用
        report_lines.append("\n## 模型使用")
        model_usage = self.analyze_model_usage()
        for mode, count in model_usage.items():
            percentage = (count / len(self.logs)) * 100
            report_lines.append(f"{mode}: {count}次 ({percentage:.2f}%)")
        
        # 保存报告
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(report_lines))
            print(f"\n分析报告生成成功: {output_file}")
        except Exception as e:
            print(f"生成报告失败: {e}")
    
    def plot_similarity_distribution(self, output_file='similarity_distribution.png'):
        """绘制相似度分布图
        
        Args:
            output_file: 输出文件路径
        """
        if not self.logs:
            return
        
        similarities = [log.get('similarity', 0) for log in self.logs]
        
        plt.figure(figsize=(10, 6))
        sns.histplot(similarities, bins=20, kde=True)
        plt.title('相似度分布')
        plt.xlabel('相似度')
        plt.ylabel('频率')
        plt.grid(True, alpha=0.3)
        
        try:
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"相似度分布图生成成功: {output_file}")
        except Exception as e:
            print(f"生成相似度分布图失败: {e}")
        finally:
            plt.close()
    
    def plot_role_distribution(self, output_file='role_distribution.png', top_n=20):
        """绘制角色分布图
        
        Args:
            output_file: 输出文件路径
            top_n: 显示前N个角色
        """
        if not self.logs:
            return
        
        roles = [log.get('role', '未知') for log in self.logs]
        role_counter = Counter(roles)
        top_roles = role_counter.most_common(top_n)
        
        roles = [role for role, count in top_roles]
        counts = [count for role, count in top_roles]
        
        plt.figure(figsize=(12, 8))
        plt.barh(roles, counts)
        plt.title(f'Top {top_n} 角色分布')
        plt.xlabel('出现次数')
        plt.ylabel('角色')
        plt.tight_layout()
        
        try:
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"角色分布图生成成功: {output_file}")
        except Exception as e:
            print(f"生成角色分布图失败: {e}")
        finally:
            plt.close()

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='日志分析脚本')
    parser.add_argument('--log_dir', default='./logs', help='日志目录')
    parser.add_argument('--output', default='log_analysis_report.txt', help='分析报告输出文件')
    parser.add_argument('--plot', action='store_true', help='生成图表')
    
    args = parser.parse_args()
    
    # 初始化分析器
    analyzer = LogAnalyzer(args.log_dir)
    
    # 加载日志
    analyzer.load_logs()
    
    # 生成报告
    analyzer.generate_report(args.output)
    
    # 生成图表
    if args.plot:
        analyzer.plot_similarity_distribution()
        analyzer.plot_role_distribution()

if __name__ == '__main__':
    main()
