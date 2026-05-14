#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型任务
用于异步处理模型相关任务
"""

from src.core.celery_config import app
from src.core.prometheus_metrics import MetricsCollector
import subprocess
import os
import time

@app.task(bind=True, queue='model_queue')
def convert_model_to_onnx(self, model_path: str, output_path: str, quantize: bool = False):
    """
    将模型转换为 ONNX 格式
    
    Args:
        model_path: 源模型路径
        output_path: 输出路径
        quantize: 是否量化
    """
    start_time = time.time()
    
    try:
        # 执行转换脚本
        cmd = ['python3', 'scripts/optimization/convert_to_onnx.py',
               '--weights', model_path,
               '--output', output_path]
        
        if quantize:
            cmd.append('--quantize')
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        )
        
        # 计算耗时
        duration = time.time() - start_time
        
        return {
            'status': 'success' if result.returncode == 0 else 'error',
            'model_path': model_path,
            'output_path': output_path,
            'quantize': quantize,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'return_code': result.returncode,
            'conversion_time_ms': duration * 1000
        }
    
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e)
        }

@app.task(bind=True, queue='model_queue')
def benchmark_model(self, model_name: str, iterations: int = 100):
    """
    模型性能基准测试
    
    Args:
        model_name: 模型名称
        iterations: 测试迭代次数
    """
    start_time = time.time()
    
    try:
        from src.core.onnx_engine import get_engine
        
        # 加载模型
        engine = get_engine(model_name)
        
        # 运行基准测试
        import numpy as np
        dummy_input = np.random.randn(1, 3, engine.input_size, engine.input_size).astype(np.float32)
        
        # 预热
        for _ in range(10):
            engine.session.run([engine.output_name], {engine.input_name: dummy_input})
        
        # 正式测试
        test_start = time.time()
        for _ in range(iterations):
            engine.session.run([engine.output_name], {engine.input_name: dummy_input})
        test_duration = time.time() - test_start
        
        avg_time_ms = (test_duration / iterations) * 1000
        fps = iterations / test_duration
        
        # 记录指标
        MetricsCollector.record_inference_time(model_name=model_name, duration=test_duration)
        
        return {
            'status': 'success',
            'model_name': model_name,
            'iterations': iterations,
            'input_size': engine.input_size,
            'avg_inference_time_ms': avg_time_ms,
            'fps': fps,
            'total_time_ms': (time.time() - start_time) * 1000
        }
    
    except Exception as e:
        return {
            'status': 'error',
            'model_name': model_name,
            'error': str(e)
        }

@app.task(bind=True, queue='model_queue')
def update_model(self, model_name: str, source_url: str):
    """
    更新模型文件
    
    Args:
        model_name: 模型名称
        source_url: 模型下载地址
    """
    try:
        import requests
        
        # 下载模型
        response = requests.get(source_url, stream=True)
        response.raise_for_status()
        
        # 保存模型
        model_dir = 'models/onnx'
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f'{model_name}.onnx')
        
        with open(model_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return {
            'status': 'success',
            'model_name': model_name,
            'model_path': model_path,
            'source_url': source_url
        }
    
    except Exception as e:
        return {
            'status': 'error',
            'model_name': model_name,
            'error': str(e)
        }
