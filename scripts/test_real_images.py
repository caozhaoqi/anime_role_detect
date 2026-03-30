#!/usr/bin/env python3
"""
使用真实角色图像测试API
"""

import sys
import os
import time
import requests
from pathlib import Path

project_root = Path(__file__).parent.parent
logger = None


def log(message, level="INFO"):
    """简单的日志函数"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp} | {level} | {message}")


def test_real_images():
    """测试真实角色图像"""
    log("=" * 60)
    log("开始测试真实角色图像")
    log("=" * 60)
    
    # API地址
    api_url = "http://localhost:8000/api/classify"
    
    # 测试图像列表
    test_images = [
        ("data/train/日奈/日奈_000.jpg", "日奈"),
        ("data/train/伊织/伊织_000.jpg", "伊织"),
        ("data/train/阿罗娜/阿罗娜_000.jpg", "阿罗娜"),
        ("data/train/普拉娜/普拉娜_000.jpg", "普拉娜"),
        ("data/train/亚子/亚子_000.jpg", "亚子"),
    ]
    
    results = []
    
    for i, (image_path, expected_role) in enumerate(test_images):
        log(f"\n测试 {i+1}/{len(test_images)}: {expected_role}")
        
        # 检查文件是否存在
        full_path = project_root / image_path
        if not full_path.exists():
            log(f"文件不存在: {full_path}", "ERROR")
            continue
        
        # 准备请求
        files = {'file': (full_path.name, open(full_path, 'rb'), 'image/jpeg')}
        data = {
            'use_model': 'false',
            'use_attributes': 'true',
            'model_name': 'default'
        }
        
        # 发送请求
        start_time = time.time()
        try:
            response = requests.post(api_url, files=files, data=data, timeout=60)
            elapsed_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                predicted_role = result.get('role', 'unknown')
                similarity = result.get('similarity', 0.0)
                
                # 判断是否正确
                is_correct = predicted_role == expected_role
                status = "✅ 正确" if is_correct else "❌ 错误"
                
                log(f"  预期: {expected_role}, 实际: {predicted_role}, 相似度: {similarity:.4f}")
                log(f"  状态: {status}, 响应时间: {elapsed_time:.2f}秒")
                
                results.append({
                    'expected': expected_role,
                    'predicted': predicted_role,
                    'similarity': similarity,
                    'is_correct': is_correct,
                    'response_time': elapsed_time
                })
            else:
                log(f"  请求失败: 状态码 {response.status_code}", "ERROR")
                results.append({
                    'expected': expected_role,
                    'predicted': 'ERROR',
                    'similarity': 0.0,
                    'is_correct': False,
                    'response_time': time.time() - start_time
                })
        except Exception as e:
            log(f"  请求异常: {e}", "ERROR")
            results.append({
                'expected': expected_role,
                'predicted': 'EXCEPTION',
                'similarity': 0.0,
                'is_correct': False,
                'response_time': time.time() - start_time
            })
        finally:
            # 关闭文件
            if 'file' in files:
                files['file'][1].close()
    
    # 统计结果
    log("\n" + "=" * 60)
    log("测试结果汇总")
    log("=" * 60)
    
    correct_count = sum(1 for r in results if r['is_correct'])
    total_count = len(results)
    accuracy = correct_count / total_count * 100 if total_count > 0 else 0
    
    avg_similarity = sum(r['similarity'] for r in results) / total_count if total_count > 0 else 0
    avg_response_time = sum(r['response_time'] for r in results) / total_count if total_count > 0 else 0
    
    log(f"总测试数: {total_count}")
    log(f"正确识别: {correct_count}")
    log(f"错误识别: {total_count - correct_count}")
    log(f"准确率: {accuracy:.1f}%")
    log(f"平均相似度: {avg_similarity:.4f}")
    log(f"平均响应时间: {avg_response_time:.2f}秒")
    
    # 详细结果
    log("\n详细结果:")
    for i, result in enumerate(results):
        status = "✅" if result['is_correct'] else "❌"
        log(f"{i+1}. {status} {result['expected']} -> {result['predicted']} (相似度: {result['similarity']:.4f}, 时间: {result['response_time']:.2f}s)")
    
    log("\n" + "=" * 60)
    log("测试完成")
    log("=" * 60)
    
    return results


if __name__ == "__main__":
    test_real_images()
