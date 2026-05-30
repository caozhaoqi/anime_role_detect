import sys
import os
import time
import traceback
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from utils.diagnostics import CrossPlatformDiagnostics
from core.logging.global_logger import get_logger

logger = get_logger("diagnostics_test")


def test_device_detection():
    logger.info("=" * 60)
    logger.info("测试1: 设备检测")
    logger.info("=" * 60)

    device = CrossPlatformDiagnostics.get_device_info()
    logger.info(f"检测到设备: {device}")

    if device == "cuda":
        logger.info("✓ CUDA设备可用")
    elif device == "mps":
        logger.info("✓ MPS设备可用 (Apple Silicon)")
    else:
        logger.info("✓ 使用CPU设备")

    return device


def test_system_info():
    logger.info("\n" + "=" * 60)
    logger.info("测试2: 系统信息获取")
    logger.info("=" * 60)

    system_info = CrossPlatformDiagnostics.get_system_info()

    logger.info("系统信息:")
    for key, value in system_info.items():
        logger.info(f"  {key}: {value}")

    return system_info


def test_memory_usage():
    logger.info("\n" + "=" * 60)
    logger.info("测试3: 内存使用监控")
    logger.info("=" * 60)

    memory_info = CrossPlatformDiagnostics.get_memory_usage()

    logger.info("内存使用情况:")
    for key, value in memory_info.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.2f}")
        else:
            logger.info(f"  {key}: {value}")

    return memory_info


def test_memory_threshold():
    logger.info("\n" + "=" * 60)
    logger.info("测试4: 内存阈值检查")
    logger.info("=" * 60)

    thresholds = [50.0, 70.0, 85.0, 95.0]

    for threshold in thresholds:
        is_high = CrossPlatformDiagnostics.check_memory_threshold(threshold)
        status = "⚠️ 超过阈值" if is_high else "✓ 正常"
        logger.info(f"阈值 {threshold}%: {status}")

    return True


def test_memory_snapshot():
    logger.info("\n" + "=" * 60)
    logger.info("测试5: 内存快照生成")
    logger.info("=" * 60)

    snapshot = CrossPlatformDiagnostics.dump_memory_snapshot()

    logger.info("内存快照已生成，包含以下信息:")
    for key, value in snapshot.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.2f}")
        elif isinstance(value, str) and len(value) > 100:
            logger.info(f"  {key}: [长文本，已省略]")
        else:
            logger.info(f"  {key}: {value}")

    return snapshot


def test_cache_clear():
    logger.info("\n" + "=" * 60)
    logger.info("测试6: 缓存清理")
    logger.info("=" * 60)

    logger.info("执行缓存清理...")
    CrossPlatformDiagnostics.clear_cache()
    logger.info("✓ 缓存清理完成")

    return True


def test_oom_diagnosis():
    logger.info("\n" + "=" * 60)
    logger.info("测试7: OOM诊断")
    logger.info("=" * 60)

    test_errors = [
        RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"),
        RuntimeError("RuntimeError: CUDA error: out of memory"),
        RuntimeError("allotted memory has been exhausted"),
        RuntimeError("Some other error"),
        ValueError("Invalid argument"),
    ]

    for error in test_errors:
        diagnosis = CrossPlatformDiagnostics.diagnose_oom_error(error)
        logger.info(f"\n错误: {str(error)[:50]}")
        logger.info(f"  是否OOM: {diagnosis['is_oom']}")
        logger.info(f"  设备类型: {diagnosis['device']}")

    return True


def test_feature_extraction_oom():
    logger.info("\n" + "=" * 60)
    logger.info("测试8: 特征提取OOM处理")
    logger.info("=" * 60)

    try:
        from core.feature_extraction.feature_extraction import FeatureExtraction
        from PIL import Image
        import numpy as np

        logger.info("初始化特征提取器...")
        extractor = FeatureExtraction(quantize=True)

        logger.info("创建测试图像...")
        test_img = Image.new("RGB", (224, 224), color="red")

        logger.info("提取特征...")
        feature = extractor.extract_features(test_img)

        logger.info(f"✓ 特征提取成功，维度: {feature.shape}")
        logger.info(f"✓ 特征向量前5个元素: {feature[:5]}")

        return True
    except Exception as e:
        logger.error(f"✗ 特征提取测试失败: {e}")
        logger.error(traceback.format_exc())
        return False


def test_large_image_handling():
    logger.info("\n" + "=" * 60)
    logger.info("测试9: 大图像处理")
    logger.info("=" * 60)

    try:
        from core.feature_extraction.feature_extraction import FeatureExtraction
        from PIL import Image

        logger.info("初始化特征提取器...")
        extractor = FeatureExtraction(quantize=True)

        logger.info("创建大图像 (3000x3000 = 900万像素)...")
        large_img = Image.new("RGB", (3000, 3000), color="blue")

        logger.info("提取特征（应该触发警告）...")
        feature = extractor.extract_features(large_img)

        logger.info(f"✓ 大图像处理成功，特征维度: {feature.shape}")

        return True
    except Exception as e:
        logger.error(f"✗ 大图像处理测试失败: {e}")
        logger.error(traceback.format_exc())
        return False


def run_all_tests():
    logger.info("\n" + "=" * 60)
    logger.info("开始跨平台诊断系统测试")
    logger.info("=" * 60)

    results = {}

    tests = [
        ("设备检测", test_device_detection),
        ("系统信息获取", test_system_info),
        ("内存使用监控", test_memory_usage),
        ("内存阈值检查", test_memory_threshold),
        ("内存快照生成", test_memory_snapshot),
        ("缓存清理", test_cache_clear),
        ("OOM诊断", test_oom_diagnosis),
        ("特征提取OOM处理", test_feature_extraction_oom),
        ("大图像处理", test_large_image_handling),
    ]

    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = "✓ 通过" if result else "✗ 失败"
        except Exception as e:
            logger.error(f"测试 '{test_name}' 执行失败: {e}")
            logger.error(traceback.format_exc())
            results[test_name] = f"✗ 错误: {str(e)[:50]}"

    logger.info("\n" + "=" * 60)
    logger.info("测试结果汇总")
    logger.info("=" * 60)

    for test_name, result in results.items():
        logger.info(f"{test_name}: {result}")

    passed = sum(1 for r in results.values() if "✓" in r)
    total = len(results)

    logger.info(f"\n总计: {passed}/{total} 测试通过")

    if passed == total:
        logger.info("🎉 所有测试通过！")
    else:
        logger.warning(f"⚠️ {total - passed} 个测试失败")


if __name__ == "__main__":
    run_all_tests()
