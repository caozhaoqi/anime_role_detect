#!/usr/bin/env python3
"""
脚本功能：检查项目中的荣誉代码（重复创建的对象或资源）
作者：Project Team
创建日期：2026-03-15
"""

import os
import re
import argparse
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HonorCodeChecker:
    """荣誉代码检查器"""
    
    def __init__(self, project_root):
        """初始化检查器"""
        self.project_root = project_root
        self.results = {
            'classifier_creation': [],
            'model_loading': [],
            'cache_management': [],
            'resource_leaks': []
        }
    
    def scan_files(self):
        """扫描项目文件"""
        logger.info(f'开始扫描项目：{self.project_root}')
        
        # 定义要扫描的文件类型
        extensions = ['.py']
        
        # 遍历项目目录
        for root, dirs, files in os.walk(self.project_root):
            # 跳过一些目录
            dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', 'node_modules', '.next']]
            
            for file in files:
                if any(file.endswith(ext) for ext in extensions):
                    file_path = os.path.join(root, file)
                    self.check_file(file_path)
    
    def check_file(self, file_path):
        """检查单个文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            logger.error(f'读取文件失败：{file_path} - {e}')
            return
        
        # 检查分类器实例创建
        self.check_classifier_creation(file_path, content)
        
        # 检查模型加载
        self.check_model_loading(file_path, content)
        
        # 检查缓存管理
        self.check_cache_management(file_path, content)
        
        # 检查资源泄漏
        self.check_resource_leaks(file_path, content)
    
    def check_classifier_creation(self, file_path, content):
        """检查分类器实例创建"""
        # 查找分类器实例创建的模式
        patterns = [
            r'GeneralClassification\(',
            r'Classification\(',
            r'classifier\s*=.*\(',
            r'new.*Classifier\(',
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                line_num = content[:match.start()].count('\n') + 1
                self.results['classifier_creation'].append({
                    'file': file_path,
                    'line': line_num,
                    'pattern': pattern,
                    'context': self.get_context(content, match.start(), 200)
                })
    
    def check_model_loading(self, file_path, content):
        """检查模型加载"""
        # 查找模型加载的模式
        patterns = [
            r'load_model\(',
            r'load.*model',
            r'model\.load\(',
            r'load.*weights',
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                line_num = content[:match.start()].count('\n') + 1
                self.results['model_loading'].append({
                    'file': file_path,
                    'line': line_num,
                    'pattern': pattern,
                    'context': self.get_context(content, match.start(), 200)
                })
    
    def check_cache_management(self, file_path, content):
        """检查缓存管理"""
        # 查找缓存管理的模式
        patterns = [
            r'cache\s*=.*\{',
            r'cache\s*\[',
            r'LRU.*cache',
            r'max_size',
            r'cache.*limit',
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                line_num = content[:match.start()].count('\n') + 1
                self.results['cache_management'].append({
                    'file': file_path,
                    'line': line_num,
                    'pattern': pattern,
                    'context': self.get_context(content, match.start(), 200)
                })
    
    def check_resource_leaks(self, file_path, content):
        """检查资源泄漏"""
        # 查找可能的资源泄漏模式
        patterns = [
            r'open\(',
            r'connect\(',
            r'socket\.',
            r'file\.',
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                line_num = content[:match.start()].count('\n') + 1
                # 检查是否有对应的关闭操作
                context = self.get_context(content, match.start(), 500)
                if 'close()' not in context and 'with' not in context:
                    self.results['resource_leaks'].append({
                        'file': file_path,
                        'line': line_num,
                        'pattern': pattern,
                        'context': context
                    })
    
    def get_context(self, content, start, length):
        """获取匹配上下文"""
        end = min(start + length, len(content))
        return content[max(0, start - 50):end].strip()
    
    def generate_report(self):
        """生成检查报告"""
        logger.info('生成荣誉代码检查报告')
        
        report = []
        report.append('=' * 80)
        report.append('荣誉代码检查报告')
        report.append('=' * 80)
        
        # 分类器实例创建检查
        if self.results['classifier_creation']:
            report.append('\n1. 分类器实例创建检查:')
            report.append('-' * 60)
            for item in self.results['classifier_creation']:
                report.append(f"文件: {item['file']}:{item['line']}")
                report.append(f"模式: {item['pattern']}")
                report.append(f"上下文: {item['context']}")
                report.append('-' * 60)
        else:
            report.append('\n1. 分类器实例创建检查: 未发现问题')
        
        # 模型加载检查
        if self.results['model_loading']:
            report.append('\n2. 模型加载检查:')
            report.append('-' * 60)
            for item in self.results['model_loading']:
                report.append(f"文件: {item['file']}:{item['line']}")
                report.append(f"模式: {item['pattern']}")
                report.append(f"上下文: {item['context']}")
                report.append('-' * 60)
        else:
            report.append('\n2. 模型加载检查: 未发现问题')
        
        # 缓存管理检查
        if self.results['cache_management']:
            report.append('\n3. 缓存管理检查:')
            report.append('-' * 60)
            for item in self.results['cache_management']:
                report.append(f"文件: {item['file']}:{item['line']}")
                report.append(f"模式: {item['pattern']}")
                report.append(f"上下文: {item['context']}")
                report.append('-' * 60)
        else:
            report.append('\n3. 缓存管理检查: 未发现问题')
        
        # 资源泄漏检查
        if self.results['resource_leaks']:
            report.append('\n4. 资源泄漏检查:')
            report.append('-' * 60)
            for item in self.results['resource_leaks']:
                report.append(f"文件: {item['file']}:{item['line']}")
                report.append(f"模式: {item['pattern']}")
                report.append(f"上下文: {item['context']}")
                report.append('-' * 60)
        else:
            report.append('\n4. 资源泄漏检查: 未发现问题')
        
        report.append('=' * 80)
        report.append('检查完成')
        report.append('=' * 80)
        
        return '\n'.join(report)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='检查项目中的荣誉代码')
    parser.add_argument('--project_root', type=str, default='.', help='项目根目录')
    parser.add_argument('--output', type=str, default='honor_code_report.txt', help='报告输出文件')
    args = parser.parse_args()
    
    # 确保项目根目录是绝对路径
    project_root = os.path.abspath(args.project_root)
    
    logger.info(f'开始检查项目：{project_root}')
    
    # 创建检查器实例
    checker = HonorCodeChecker(project_root)
    
    # 扫描文件
    checker.scan_files()
    
    # 生成报告
    report = checker.generate_report()
    
    # 输出报告
    print(report)
    
    # 保存报告到文件
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f'报告已保存到：{args.output}')

if __name__ == '__main__':
    main()