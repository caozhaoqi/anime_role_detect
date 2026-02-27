#!/usr/bin/env python3
"""
模型部署脚本 - 使用FastAPI提供角色识别API服务
"""
import os
import sys
import argparse
import logging
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
import uvicorn
import io
import hashlib
from functools import lru_cache

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('model_deployment')

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.training.train_simple import SimpleCharacterClassifier
from models.training.train_resnet import ResNetCharacterClassifier

class ModelManager:
    """模型版本管理类"""
    
    def __init__(self, model_configs):
        """初始化模型管理器
        
        Args:
            model_configs: 模型配置列表，每个配置包含name, path, type
        """
        self.models = {}
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        
        # 数据预处理
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # 缓存配置
        self.cache = {}
        self.cache_size = 1000  # 缓存大小
        self.cache_hits = 0
        self.cache_misses = 0
        
        # 加载所有模型
        for config in model_configs:
            model_name = config['name']
            model_path = config['path']
            model_type = config['type']
            try:
                model = self._load_model(model_path, model_type)
                self.models[model_name] = {
                    'model': model,
                    'class_to_idx': model['class_to_idx'],
                    'idx_to_class': model['idx_to_class'],
                    'path': model_path,
                    'type': model_type
                }
                logger.info(f"模型 {model_name} 加载成功")
            except Exception as e:
                logger.error(f"模型 {model_name} 加载失败: {e}")
    
    def _load_model(self, model_path, model_type):
        """加载单个模型
        
        Args:
            model_path: 模型路径
            model_type: 模型类型
            
        Returns:
            dict: 模型信息
        """
        checkpoint = torch.load(model_path, map_location=self.device)
        class_to_idx = checkpoint.get('class_to_idx', {})
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        num_classes = len(class_to_idx)
        
        # 初始化模型
        if model_type == 'simple':
            model = SimpleCharacterClassifier(num_classes=num_classes)
        else:
            model = ResNetCharacterClassifier(num_classes=num_classes)
        
        # 加载模型权重
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        return {
            'model': model,
            'class_to_idx': class_to_idx,
            'idx_to_class': idx_to_class
        }
    
    def get_model(self, model_name):
        """获取指定模型
        
        Args:
            model_name: 模型名称
            
        Returns:
            dict: 模型信息
        """
        if model_name not in self.models:
            raise ValueError(f"模型 {model_name} 不存在")
        return self.models[model_name]
    
    def list_models(self):
        """列出所有可用模型
        
        Returns:
            list: 模型列表
        """
        return [
            {
                'name': name,
                'path': info['path'],
                'type': info['type'],
                'classes': len(info['model']['class_to_idx'])
            }
            for name, info in self.models.items()
        ]
    
    def _get_cache_key(self, image_content, model_name):
        """生成缓存键
        
        Args:
            image_content: 图像内容（字节流）
            model_name: 模型名称
            
        Returns:
            str: 缓存键
        """
        # 计算图像内容的哈希值
        hash_obj = hashlib.md5()
        hash_obj.update(image_content)
        image_hash = hash_obj.hexdigest()
        return f"{model_name}:{image_hash}"
    
    def _clean_cache(self):
        """清理缓存，保持缓存大小在限制范围内"""
        if len(self.cache) > self.cache_size:
            # 删除最旧的缓存项
            oldest_keys = list(self.cache.keys())[:len(self.cache) - self.cache_size]
            for key in oldest_keys:
                del self.cache[key]
            logger.info(f"清理缓存，删除了 {len(oldest_keys)} 个旧缓存项")
    
    def predict(self, image_content, model_name='default'):
        """预测图像类别
        
        Args:
            image_content: 图像内容（字节流）
            model_name: 模型名称
            
        Returns:
            dict: 预测结果
        """
        # 生成缓存键
        cache_key = self._get_cache_key(image_content, model_name)
        
        # 检查缓存
        if cache_key in self.cache:
            self.cache_hits += 1
            logger.info(f"缓存命中: {cache_key}")
            # 更新缓存状态为hit
            cached_result = self.cache[cache_key].copy()
            cached_result['cache_status'] = 'hit'
            return cached_result
        
        # 缓存未命中，执行预测
        self.cache_misses += 1
        
        model_info = self.get_model(model_name)
        model = model_info['model']['model']  # 正确获取模型对象
        idx_to_class = model_info['model']['idx_to_class']
        
        try:
            # 加载图像
            image = Image.open(io.BytesIO(image_content)).convert('RGB')
            
            # 预处理
            image = self.transform(image)
            image = image.unsqueeze(0)  # 添加批次维度
            image = image.to(self.device)
            
            # 预测
            with torch.no_grad():
                outputs = model(image)
                _, preds = torch.max(outputs, 1)
                probabilities = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()[0]
            
            # 获取预测结果
            predicted_idx = preds.item()
            predicted_class = idx_to_class.get(predicted_idx, "未知")
            # 提取角色名，去掉前缀（如 sdv50_ 或 原神_）
            if '_' in predicted_class:
                predicted_class = predicted_class.split('_')[-1]
            confidence = float(probabilities[predicted_idx])
            
            # 获取前5个预测结果
            top5_indices = np.argsort(probabilities)[-5:][::-1]
            top5_results = [
                {
                    "class": idx_to_class.get(idx, "未知").split('_')[-1] if '_' in idx_to_class.get(idx, "未知") else idx_to_class.get(idx, "未知"),
                    "confidence": float(probabilities[idx])
                }
                for idx in top5_indices
            ]
            
            result = {
                "predicted_class": predicted_class,
                "confidence": confidence,
                "top5": top5_results,
                "model_used": model_name,
                "cache_status": "miss"
            }
            
            # 存储到缓存
            self.cache[cache_key] = result
            
            # 清理缓存
            self._clean_cache()
            
            # 定期记录缓存统计
            if (self.cache_hits + self.cache_misses) % 100 == 0:
                hit_rate = self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0
                logger.info(f"缓存统计: 命中 {self.cache_hits}, 未命中 {self.cache_misses}, 命中率 {hit_rate:.2f}")
            
            return result
        except Exception as e:
            logger.error(f"预测失败: {e}")
            raise
    
def create_app(model_manager):
    """创建FastAPI应用
    
    Args:
        model_manager: 模型管理器实例
        
    Returns:
        FastAPI: FastAPI应用实例
    """
    app = FastAPI(
        title="角色识别API",
        description="使用深度学习模型识别动漫角色",
        version="1.0.0"
    )
    
    @app.get("/")
    async def root():
        """根路径"""
        return {
            "message": "角色识别API服务运行中",
            "version": "1.0.0",
            "endpoints": {
                "/predict": "上传图像进行角色识别",
                "/predict_batch": "批量上传图像进行角色识别",
                "/models": "列出所有可用模型",
                "/health": "健康检查"
            }
        }
    
    @app.get("/health")
    async def health_check():
        """健康检查"""
        return {"status": "healthy"}
    
    @app.get("/models")
    async def list_models():
        """列出所有可用模型"""
        try:
            models = model_manager.list_models()
            return {
                "total": len(models),
                "models": models
            }
        except Exception as e:
            logger.error(f"获取模型列表失败: {e}")
            raise HTTPException(status_code=500, detail=f"获取模型列表失败: {str(e)}")
    
    @app.post("/predict")
    async def predict(file: UploadFile = File(...), model: str = Form("default")):
        """预测图像类别
        
        Args:
            file: 上传的图像文件
            model: 模型名称，默认为default
            
        Returns:
            dict: 预测结果
        """
        try:
            # 读取文件内容
            content = await file.read()
            
            # 检查文件类型
            if not file.content_type or not file.content_type.startswith('image/'):
                raise HTTPException(status_code=400, detail="只支持图像文件")
            
            # 预测
            result = model_manager.predict(content, model_name=model)
            
            return result
        except HTTPException:
            raise
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"预测请求失败: {e}")
            raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")
    
    @app.post("/predict_batch")
    async def predict_batch(files: list[UploadFile] = File(...), model: str = Form("default")):
        """批量预测图像类别
        
        Args:
            files: 上传的图像文件列表
            model: 模型名称，默认为default
            
        Returns:
            list: 预测结果列表
        """
        try:
            if len(files) > 10:
                raise HTTPException(status_code=400, detail="批量预测最多支持10张图片")
            
            results = []
            for file in files:
                # 检查文件类型
                if not file.content_type or not file.content_type.startswith('image/'):
                    results.append({
                        "file_name": file.filename,
                        "error": "只支持图像文件"
                    })
                    continue
                
                # 读取文件内容
                content = await file.read()
                
                # 预测
                try:
                    result = model_manager.predict(content, model_name=model)
                    results.append({
                        "file_name": file.filename,
                        "result": result
                    })
                except ValueError as e:
                    results.append({
                        "file_name": file.filename,
                        "error": str(e)
                    })
                except Exception as e:
                    results.append({
                        "file_name": file.filename,
                        "error": f"预测失败: {str(e)}"
                    })
            
            return {
                "total": len(files),
                "results": results
            }
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"批量预测请求失败: {e}")
            raise HTTPException(status_code=500, detail=f"批量预测失败: {str(e)}")
    
    return app

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='模型部署脚本 - 提供角色识别API服务')
    
    parser.add_argument('--host', type=str, default='0.0.0.0', help='服务主机')
    parser.add_argument('--port', type=int, default=8000, help='服务端口')
    
    args = parser.parse_args()
    
    # 模型配置
    model_configs = [
        {
            'name': 'default',
            'path': 'models_simple/character_classifier_simple_best.pth',
            'type': 'simple'
        }
    ]
    
    logger.info('开始部署模型...')
    logger.info(f'服务地址: {args.host}:{args.port}')
    logger.info(f'模型配置: {model_configs}')
    
    try:
        # 初始化模型管理器
        model_manager = ModelManager(model_configs)
        
        # 创建FastAPI应用
        app = create_app(model_manager)
        
        # 启动服务
        logger.info('启动API服务...')
        uvicorn.run(app, host=args.host, port=args.port)
    except Exception as e:
        logger.error(f"部署失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
