#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
下载并部署WD Vit V3 Tagger模型
"""

import os
import argparse
import logging
from huggingface_hub import hf_hub_download

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('deploy_wd_vit_v3')


def download_model(model_id, model_dir):
    """从Hugging Face下载WD Vit V3 Tagger模型
    
    Args:
        model_id: Hugging Face模型ID
        model_dir: 模型保存目录
    """
    os.makedirs(model_dir, exist_ok=True)
    
    # 下载模型文件
    files_to_download = [
        'model.safetensors',
        'config.json',
        'preprocessor_config.json',
        'tokenizer.json',
        'tokenizer_config.json'
    ]
    
    for file_name in files_to_download:
        try:
            file_path = hf_hub_download(
                repo_id=model_id,
                filename=file_name,
                local_dir=model_dir
            )
            logger.info(f"下载完成: {file_path}")
        except Exception as e:
            logger.warning(f"下载 {file_name} 失败: {e}")
            # 继续下载其他文件
            continue


def main():
    parser = argparse.ArgumentParser(description='下载并部署WD Vit V3 Tagger模型')
    parser.add_argument('--model-id', type=str, default='SmilingWolf/wd-vit-v3', help='Hugging Face模型ID')
    parser.add_argument('--model-dir', type=str, default='models/wd-vit-v3', help='模型保存目录')
    
    args = parser.parse_args()
    
    logger.info(f"开始下载WD Vit V3 Tagger模型: {args.model_id}")
    logger.info(f"保存目录: {args.model_dir}")
    
    download_model(args.model_id, args.model_dir)
    
    logger.info("模型下载完成！")
    logger.info(f"模型已保存到: {args.model_dir}")
    logger.info("现在可以使用本地模型进行推理了")


if __name__ == '__main__':
    main()