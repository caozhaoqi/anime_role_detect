# 全局日志系统使用指南

## 1. 系统概述

全局日志系统是一个统一的日志管理模块，用于记录系统运行状态、模型推理结果、模型训练结果和错误日志。它基于 loguru 库实现，提供了按类型分目录存储的结构和日志轮转功能。

## 2. 日志目录结构

全局日志系统按类型分目录存储日志文件：

```
logs/
├── system/         # 系统运行状态日志
├── inference/      # 模型推理结果日志
├── training/       # 模型训练结果日志
└── error/          # 错误日志
```

## 3. 日志轮转配置

全局日志系统配置了以下日志轮转策略：

| 日志类型 | 轮转策略 | 保留时间 | 压缩方式 |
|---------|---------|---------|--------|
| 系统日志 | 100 MB | 7 天 | zip |
| 推理日志 | 100 MB | 14 天 | zip |
| 训练日志 | 200 MB | 30 天 | zip |
| 错误日志 | 50 MB | 30 天 | zip |

## 4. 使用方法

### 4.1 基本使用

在需要使用日志的模块中，导入全局日志系统并使用：

```python
from src.core.logging.global_logger import (
    get_logger, log_system, log_inference, log_training, log_error
)

# 使用便捷函数记录日志
log_system("系统启动成功")
log_inference("模型推理完成，识别结果: 角色A, 相似度: 0.95")
log_training("模型训练完成，准确率: 98.5%")
log_error("文件上传失败: 文件太大")

# 使用自定义logger对象
logger = get_logger("module_name")
logger.info("模块初始化完成")
logger.error("模块错误: 参数无效")
```

### 4.2 日志级别

全局日志系统支持以下日志级别：

- DEBUG: 详细的调试信息
- INFO: 一般信息
- WARNING: 警告信息
- ERROR: 错误信息
- CRITICAL: 严重错误信息

使用示例：

```python
# 记录不同级别的日志
log_system("调试信息", level="debug")
log_system("警告信息", level="warning")
log_error("严重错误", level="critical")

# 使用logger对象记录不同级别的日志
logger = get_logger("module_name")
logger.debug("调试信息")
logger.info("一般信息")
logger.warning("警告信息")
logger.error("错误信息")
logger.critical("严重错误信息")
```

## 5. 集成到现有模块

### 5.1 API路由模块

在API路由模块中，使用全局日志系统记录请求处理和错误信息：

```python
from src.core.logging.global_logger import get_logger, log_inference, log_error

logger = get_logger("api_routes")

@app.route('/api/classify', methods=['POST'])
def api_classify():
    try:
        # 处理请求...
        
        # 记录推理结果
        log_inference(f"图像分类成功: {filename}, 角色: {role}, 相似度: {similarity:.4f}")
        
        return json.dumps(response), 200
    except Exception as e:
        # 记录错误
        error_msg = f"分类失败: {str(e)}"
        log_error(error_msg)
        logger.error(error_msg)
        return json.dumps({'error': error_msg}), 500
```

### 5.2 模型训练模块

在模型训练模块中，使用全局日志系统记录训练过程和结果：

```python
from src.core.logging.global_logger import get_logger, log_training

logger = get_logger("train_model")

def train_model():
    log_training("开始训练模型...")
    
    # 训练过程...
    
    log_training(f"训练完成！最佳验证准确率: {best_acc:.2f}%")
```

## 6. 日志格式

全局日志系统使用以下日志格式：

```
<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | <level>{message}</level>
```

示例输出：

```
2026-02-12 17:30:22.742 | INFO     | src.core.logging.global_logger:log_system:135 | 系统启动成功
2026-02-12 17:30:22.742 | ERROR    | api_routes:api_classify:150 | 分类失败: 文件格式不支持
```

## 7. 自定义配置

如果需要自定义日志系统的配置，可以修改 `src/core/logging/global_logger.py` 文件中的相关参数：

- `log_dir`: 日志根目录
- `rotation`: 日志轮转策略
- `retention`: 日志保留时间
- `compression`: 日志压缩方式
- `level`: 日志级别
- `format`: 日志格式

## 8. 故障排查

### 8.1 日志文件不存在

如果日志文件不存在，可能是以下原因：

1. 日志目录权限不足
2. 日志系统未正确初始化
3. 日志级别设置过高，没有触发日志记录

### 8.2 日志内容为空

如果日志文件存在但内容为空，可能是以下原因：

1. 日志级别设置过高，没有触发日志记录
2. 日志记录语句没有执行到
3. 日志系统配置错误

### 8.3 日志轮转不生效

如果日志轮转不生效，可能是以下原因：

1. 日志文件大小未达到轮转阈值
2. 日志系统配置错误
3. 磁盘空间不足

## 9. 最佳实践

1. **按类型记录日志**：使用不同类型的日志函数记录对应类型的日志，便于后续分析和管理。

2. **记录关键信息**：在日志中记录足够的关键信息，便于故障排查和系统监控。

3. **避免过度日志**：不要记录过多的调试信息，以免影响系统性能和日志可读性。

4. **使用结构化日志**：在日志中包含结构化信息，便于后续分析和处理。

5. **定期清理日志**：虽然系统配置了日志保留时间，但建议定期检查和清理日志文件，避免磁盘空间不足。

## 10. 总结

全局日志系统是一个功能强大、配置灵活的日志管理模块，它提供了按类型分目录存储的结构和日志轮转功能，便于系统运行状态的监控、故障排查和结果分析。通过统一的日志管理，可以提高系统的可维护性和可靠性。
