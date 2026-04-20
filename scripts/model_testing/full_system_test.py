#!/usr/bin/env python3
"""
完整系统测试套件
测试所有优化功能
"""

import sys
import os
import time
import json
import requests
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from utils.diagnostics import CrossPlatformDiagnostics
from core.logging.global_logger import get_logger

logger = get_logger("full_system_test")

API_BASE_URL = "http://localhost:8000"
TEST_RESULTS = {}


def test_api_health():
    """测试API健康检查"""
    logger.info("=" * 60)
    logger.info("测试1: API健康检查")
    logger.info("=" * 60)
    
    try:
        response = requests.get(f"{API_BASE_URL}/api/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            logger.info(f"✓ API健康检查通过: {data}")
            return True, data
        else:
            logger.error(f"✗ API健康检查失败: 状态码 {response.status_code}")
            return False, None
    except Exception as e:
        logger.error(f"✗ API健康检查异常: {e}")
        return False, None


def test_api_info():
    """测试API信息端点"""
    logger.info("\n" + "=" * 60)
    logger.info("测试2: API信息")
    logger.info("=" * 60)
    
    try:
        response = requests.get(f"{API_BASE_URL}/api/info", timeout=10)
        if response.status_code == 200:
            data = response.json()
            logger.info(f"✓ API信息获取成功")
            logger.info(f"  服务名称: {data.get('name', 'N/A')}")
            logger.info(f"  版本: {data.get('version', 'N/A')}")
            logger.info(f"  状态: {data.get('status', 'N/A')}")
            return True, data
        else:
            logger.error(f"✗ API信息获取失败: 状态码 {response.status_code}")
            return False, None
    except Exception as e:
        logger.error(f"✗ API信息获取异常: {e}")
        return False, None


def test_monitoring_endpoints():
    """测试监控端点"""
    logger.info("\n" + "=" * 60)
    logger.info("测试3: 监控端点")
    logger.info("=" * 60)
    
    endpoints = [
        "/api/monitoring/status",
        "/api/monitoring/memory",
        "/api/monitoring/network"
    ]
    
    results = {}
    for endpoint in endpoints:
        try:
            response = requests.get(f"{API_BASE_URL}{endpoint}", timeout=10)
            if response.status_code == 200:
                data = response.json()
                logger.info(f"✓ {endpoint}: 正常")
                results[endpoint] = True
            else:
                logger.warning(f"⚠ {endpoint}: 状态码 {response.status_code}")
                results[endpoint] = False
        except Exception as e:
            logger.warning(f"⚠ {endpoint}: 异常 - {e}")
            results[endpoint] = False
    
    return all(results.values()), results


def test_diagnostics_integration():
    """测试诊断系统集成"""
    logger.info("\n" + "=" * 60)
    logger.info("测试4: 诊断系统集成")
    logger.info("=" * 60)
    
    try:
        # 测试设备检测
        device = CrossPlatformDiagnostics.get_device_info()
        logger.info(f"✓ 设备检测: {device}")
        
        # 测试内存快照
        snapshot = CrossPlatformDiagnostics.dump_memory_snapshot()
        logger.info(f"✓ 内存快照生成成功")
        logger.info(f"  平台: {snapshot.get('platform', 'N/A')}")
        logger.info(f"  CPU使用率: {snapshot.get('cpu_percent', 'N/A')}%")
        logger.info(f"  内存使用: {snapshot.get('ram_used_gb', 'N/A'):.2f} GB")
        
        # 测试内存阈值检查
        is_high = CrossPlatformDiagnostics.check_memory_threshold(95.0)
        logger.info(f"✓ 内存阈值检查: {'超过阈值' if is_high else '正常'}")
        
        # 测试缓存清理
        CrossPlatformDiagnostics.clear_cache()
        logger.info(f"✓ 缓存清理成功")
        
        return True, snapshot
    except Exception as e:
        logger.error(f"✗ 诊断系统集成测试失败: {e}")
        return False, None


def test_image_classification():
    """测试图像分类功能"""
    logger.info("\n" + "=" * 60)
    logger.info("测试5: 图像分类功能")
    logger.info("=" * 60)
    
    # 创建测试图像
    try:
        from PIL import Image
        import io
        
        # 创建测试图像
        img = Image.new('RGB', (224, 224), color='red')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        # 发送分类请求
        files = {'file': ('test.png', img_bytes, 'image/png')}
        data = {'use_model': 'false', 'use_attributes': 'true', 'model_name': 'default'}
        
        start_time = time.time()
        response = requests.post(
            f"{API_BASE_URL}/api/classify",
            files=files,
            data=data,
            timeout=60
        )
        elapsed_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            logger.info(f"✓ 图像分类成功")
            logger.info(f"  响应时间: {elapsed_time:.2f}秒")
            logger.info(f"  识别角色: {result.get('role', 'N/A')}")
            logger.info(f"  相似度: {result.get('similarity', 'N/A')}")
            return True, result
        else:
            logger.error(f"✗ 图像分类失败: 状态码 {response.status_code}")
            logger.error(f"  响应: {response.text}")
            return False, None
    except Exception as e:
        logger.error(f"✗ 图像分类测试异常: {e}")
        return False, None


def test_batch_classification():
    """测试批量分类功能"""
    logger.info("\n" + "=" * 60)
    logger.info("测试6: 批量分类功能")
    logger.info("=" * 60)
    
    try:
        from PIL import Image
        import io
        
        # 创建多个测试图像
        files = []
        for i in range(3):
            img = Image.new('RGB', (224, 224), color=['red', 'green', 'blue'][i])
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='PNG')
            img_bytes.seek(0)
            files.append(('files', (f'test_{i}.png', img_bytes, 'image/png')))
        
        # 发送批量分类请求
        start_time = time.time()
        response = requests.post(
            f"{API_BASE_URL}/api/classify/batch",
            files=files,
            data={'model_name': 'default'},
            timeout=120
        )
        elapsed_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            results = result.get('results', [])
            logger.info(f"✓ 批量分类成功")
            logger.info(f"  响应时间: {elapsed_time:.2f}秒")
            logger.info(f"  处理图像数: {len(results)}")
            return True, result
        else:
            logger.error(f"✗ 批量分类失败: 状态码 {response.status_code}")
            return False, None
    except Exception as e:
        logger.error(f"✗ 批量分类测试异常: {e}")
        return False, None


def test_performance():
    """测试性能"""
    logger.info("\n" + "=" * 60)
    logger.info("测试7: 性能测试")
    logger.info("=" * 60)
    
    try:
        from PIL import Image
        import io
        
        # 创建测试图像
        img = Image.new('RGB', (224, 224), color='red')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        
        # 进行多次请求测试
        response_times = []
        for i in range(5):
            img_bytes.seek(0)
            files = {'file': ('test.png', img_bytes, 'image/png')}
            data = {'use_model': 'false', 'use_attributes': 'true', 'model_name': 'default'}
            
            start_time = time.time()
            response = requests.post(
                f"{API_BASE_URL}/api/classify",
                files=files,
                data=data,
                timeout=60
            )
            elapsed_time = time.time() - start_time
            
            if response.status_code == 200:
                response_times.append(elapsed_time)
                logger.info(f"  请求 {i+1}: {elapsed_time:.2f}秒")
            else:
                logger.warning(f"  请求 {i+1}: 失败")
        
        if response_times:
            avg_time = sum(response_times) / len(response_times)
            min_time = min(response_times)
            max_time = max(response_times)
            
            logger.info(f"✓ 性能测试完成")
            logger.info(f"  平均响应时间: {avg_time:.2f}秒")
            logger.info(f"  最小响应时间: {min_time:.2f}秒")
            logger.info(f"  最大响应时间: {max_time:.2f}秒")
            
            return True, {
                'avg_time': avg_time,
                'min_time': min_time,
                'max_time': max_time
            }
        else:
            logger.error(f"✗ 性能测试失败: 无成功请求")
            return False, None
    except Exception as e:
        logger.error(f"✗ 性能测试异常: {e}")
        return False, None


def run_all_tests():
    """运行所有测试"""
    logger.info("\n" + "=" * 60)
    logger.info("开始完整系统测试")
    logger.info("=" * 60)
    logger.info(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"API地址: {API_BASE_URL}")
    logger.info("")
    
    tests = [
        ("API健康检查", test_api_health),
        ("API信息", test_api_info),
        ("监控端点", test_monitoring_endpoints),
        ("诊断系统集成", test_diagnostics_integration),
        ("图像分类功能", test_image_classification),
        ("批量分类功能", test_batch_classification),
        ("性能测试", test_performance)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            success, data = test_func()
            results[test_name] = {
                'success': success,
                'data': data
            }
        except Exception as e:
            logger.error(f"测试 '{test_name}' 执行失败: {e}")
            results[test_name] = {
                'success': False,
                'error': str(e)
            }
    
    # 生成测试报告
    logger.info("\n" + "=" * 60)
    logger.info("测试结果汇总")
    logger.info("=" * 60)
    
    passed = 0
    failed = 0
    
    for test_name, result in results.items():
        status = "✓ 通过" if result['success'] else "✗ 失败"
        logger.info(f"{test_name}: {status}")
        if result['success']:
            passed += 1
        else:
            failed += 1
    
    logger.info(f"\n总计: {passed}/{len(results)} 测试通过")
    
    if failed == 0:
        logger.info("🎉 所有测试通过！系统运行正常！")
    else:
        logger.warning(f"⚠️ {failed} 个测试失败")
    
    # 保存测试结果
    test_report = {
        'test_time': datetime.now().isoformat(),
        'api_url': API_BASE_URL,
        'results': results,
        'summary': {
            'total': len(results),
            'passed': passed,
            'failed': failed,
            'pass_rate': f"{passed/len(results)*100:.1f}%"
        }
    }
    
    report_file = project_root / "logs" / "full_system_test_report.json"
    report_file.parent.mkdir(exist_ok=True)
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(test_report, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n测试报告已保存: {report_file}")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
