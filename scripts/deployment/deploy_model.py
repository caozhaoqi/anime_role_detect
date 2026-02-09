#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型部署脚本

将训练好的模型部署到生产环境
"""

import os
import argparse
import logging
import shutil
from pathlib import Path

from src.utils.config_utils import (
    get_model_dir,
    get_onnx_dir,
    get_checkpoint_dir,
    get_src_dir
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('model_deployer')


def parse_args():
    """
    解析命令行参数
    
    Returns:
        argparse.Namespace: 命令行参数
    """
    parser = argparse.ArgumentParser(description='模型部署脚本')
    parser.add_argument('--model-path', type=str, required=True, help='模型文件路径')
    parser.add_argument('--output-dir', type=str, default='deployment', help='部署输出目录')
    parser.add_argument('--format', type=str, default='onnx', choices=['pth', 'onnx'], help='模型格式')
    parser.add_argument('--include-dependencies', action='store_true', help='包含依赖文件')
    parser.add_argument('--include-config', action='store_true', help='包含配置文件')
    
    return parser.parse_args()


def prepare_deployment_directory(output_dir):
    """
    准备部署目录
    
    Args:
        output_dir: 部署输出目录
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # 创建子目录
    (output_path / 'models').mkdir(exist_ok=True)
    (output_path / 'config').mkdir(exist_ok=True)
    (output_path / 'scripts').mkdir(exist_ok=True)
    (output_path / 'logs').mkdir(exist_ok=True)
    
    logger.info(f"创建部署目录: {output_path}")


def copy_model(model_path, output_dir, model_format):
    """
    复制模型文件到部署目录
    
    Args:
        model_path: 模型文件路径
        output_dir: 部署输出目录
        model_format: 模型格式
    """
    model_path = Path(model_path)
    output_path = Path(output_dir) / 'models'
    
    if not model_path.exists():
        logger.error(f"模型文件不存在: {model_path}")
        return False
    
    # 复制模型文件
    dest_path = output_path / model_path.name
    shutil.copy2(model_path, dest_path)
    logger.info(f"复制模型文件: {model_path} -> {dest_path}")
    
    return True


def copy_dependencies(output_dir):
    """
    复制依赖文件到部署目录
    
    Args:
        output_dir: 部署输出目录
    """
    output_path = Path(output_dir)
    
    # 复制配置文件
    config_path = Path(__file__).parent.parent.parent / 'config'
    if config_path.exists():
        dest_config_path = output_path / 'config'
        for config_file in config_path.glob('*.py'):
            shutil.copy2(config_file, dest_config_path)
            logger.info(f"复制配置文件: {config_file} -> {dest_config_path}")
    
    # 复制源代码
    src_path = get_src_dir()
    src_path = Path(src_path)
    if src_path.exists():
        dest_src_path = output_path / 'src'
        dest_src_path.mkdir(exist_ok=True)
        
        # 复制必要的模块
        modules_to_copy = ['utils', 'model_training']
        for module in modules_to_copy:
            module_path = src_path / module
            if module_path.exists():
                dest_module_path = dest_src_path / module
                shutil.copytree(module_path, dest_module_path, dirs_exist_ok=True)
                logger.info(f"复制模块: {module_path} -> {dest_module_path}")


def create_requirements_file(output_dir):
    """
    创建依赖文件
    
    Args:
        output_dir: 部署输出目录
    """
    output_path = Path(output_dir)
    requirements_path = output_path / 'requirements.txt'
    
    with open(requirements_path, 'w') as f:
        f.write("""# 部署依赖
numpy
pandas
Pillow
requests
PyTorch
torchvision
onnx
onnxruntime
Flask
""")
    
    logger.info(f"创建依赖文件: {requirements_path}")


def create_deployment_script(output_dir):
    """
    创建部署脚本
    
    Args:
        output_dir: 部署输出目录
    """
    output_path = Path(output_dir)
    deploy_script_path = output_path / 'scripts' / 'run_deployment.py'
    
    with open(deploy_script_path, 'w') as f:
        f.write("""
""")
    
    # 设置脚本执行权限
    deploy_script_path.chmod(0o755)
    logger.info(f"创建部署脚本: {deploy_script_path}")


def create_readme(output_dir):
    """
    创建部署说明文件
    
    Args:
        output_dir: 部署输出目录
    """
    output_path = Path(output_dir)
    readme_path = output_path / 'DEPLOYMENT.md'
    
    with open(readme_path, 'w') as f:
        f.write("""# 模型部署说明

## 目录结构

```
deployment/
├── models/         # 模型文件
├── config/         # 配置文件
├── scripts/        # 脚本文件
├── logs/           # 日志文件
└── DEPLOYMENT.md   # 部署说明
```

## 部署步骤

### 1. 准备环境

```bash
# 创建虚拟环境
python3 -m venv venv

# 激活虚拟环境
# Linux/Mac
source venv/bin/activate
# Windows
venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 启动服务

```bash
# 运行部署脚本
python scripts/run_deployment.py

# 启动Flask服务（如果需要）
python -m flask run --host=0.0.0.0 --port=5000
```

## 模型使用

### 推理API

```python
from scripts.run_deployment import load_model, run_inference

# 加载模型
model = load_model('models/model.onnx')

# 准备输入数据
input_data = ...  # 准备模型输入数据

# 运行推理
result = run_inference(model, input_data)
print(result)
```

### 批量推理

```bash
# 运行批量推理脚本
python scripts/batch_inference.py --input-dir input_images --output-dir output_results
```

## 配置说明

### 模型配置

- **models/**: 存储模型文件
- **config/**: 存储配置文件

### 环境变量

| 环境变量 | 描述 | 默认值 |
|---------|------|--------|
| MODEL_PATH | 模型文件路径 | models/model.onnx |
| PORT | 服务端口 | 5000 |
| HOST | 服务主机 | 0.0.0.0 |
| LOG_LEVEL | 日志级别 | INFO |

## 故障排除

### 模型加载失败

- 检查模型文件路径是否正确
- 检查模型文件是否损坏
- 检查依赖是否安装正确

### 推理失败

- 检查输入数据格式是否正确
- 检查模型输入维度是否匹配
- 检查模型是否支持当前输入类型

### 服务启动失败

- 检查端口是否被占用
- 检查依赖是否安装正确
- 检查配置文件是否正确

## 性能优化

### 模型优化

- 使用ONNX格式模型获得更好的推理性能
- 对模型进行量化处理
- 对模型进行裁剪和压缩

### 部署优化

- 使用GPU进行推理（如果可用）
- 启用批处理推理
- 使用异步处理提高并发性能

## 版本说明

- **v1.0**: 初始部署版本
- **v1.1**: 添加ONNX格式支持
- **v1.2**: 添加批量推理功能
- **v1.3**: 优化部署脚本
""")
    
    logger.info(f"创建部署说明文件: {readme_path}")


def main():
    """
    主函数
    """
    args = parse_args()
    
    # 准备部署目录
    prepare_deployment_directory(args.output_dir)
    
    # 复制模型文件
    success = copy_model(args.model_path, args.output_dir, args.format)
    if not success:
        return
    
    # 包含依赖文件
    if args.include_dependencies:
        copy_dependencies(args.output_dir)
        create_requirements_file(args.output_dir)
    
    # 包含配置文件
    if args.include_config:
        # 复制配置文件
        config_path = Path(__file__).parent.parent.parent / 'config'
        if config_path.exists():
            dest_config_path = Path(args.output_dir) / 'config'
            for config_file in config_path.glob('*.py'):
                shutil.copy2(config_file, dest_config_path)
                logger.info(f"复制配置文件: {config_file} -> {dest_config_path}")
    
    # 创建部署脚本
    create_deployment_script(args.output_dir)
    
    # 创建部署说明文件
    create_readme(args.output_dir)
    
    logger.info("模型部署准备完成！")
    logger.info(f"部署目录: {args.output_dir}")
    logger.info("请按照DEPLOYMENT.md中的说明进行部署")


if __name__ == '__main__':
    main()
