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
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import io

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('model_deployment')

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.model_training.train_simple import SimpleCharacterClassifier
from src.model_training.train_resnet import ResNetCharacterClassifier

class ModelDeployer:
    """模型部署类"""
    
    def __init__(self, model_path, model_type='simple'):
        """初始化模型部署器
        
        Args:
            model_path: 模型路径
            model_type: 模型类型 ('simple' 或 'resnet')
        """
        self.model_path = model_path
        self.model_type = model_type
        self.model = None
        self.class_to_idx = None
        self.idx_to_class = None
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
        
        # 加载模型
        self.load_model()
    
    def load_model(self):
        """加载模型"""
        try:
            logger.info(f"加载模型: {self.model_path}")
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # 获取类别映射
            self.class_to_idx = checkpoint.get('class_to_idx', {})
            self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
            
            # 初始化模型
            num_classes = len(self.class_to_idx)
            if self.model_type == 'simple':
                self.model = SimpleCharacterClassifier(num_classes=num_classes)
            else:
                self.model = ResNetCharacterClassifier(num_classes=num_classes)
            
            # 加载模型权重
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"模型加载成功，包含 {num_classes} 个类别")
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
    
    def predict(self, image_content):
        """预测图像类别
        
        Args:
            image_content: 图像内容（字节流）
            
        Returns:
            dict: 预测结果
        """
        try:
            # 加载图像
            image = Image.open(io.BytesIO(image_content)).convert('RGB')
            
            # 预处理
            image = self.transform(image)
            image = image.unsqueeze(0)  # 添加批次维度
            image = image.to(self.device)
            
            # 预测
            with torch.no_grad():
                outputs = self.model(image)
                _, preds = torch.max(outputs, 1)
                probabilities = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()[0]
            
            # 获取预测结果
            predicted_idx = preds.item()
            predicted_class = self.idx_to_class.get(predicted_idx, "未知")
            confidence = float(probabilities[predicted_idx])
            
            # 获取前5个预测结果
            top5_indices = np.argsort(probabilities)[-5:][::-1]
            top5_results = [
                {
                    "class": self.idx_to_class.get(idx, "未知"),
                    "confidence": float(probabilities[idx])
                }
                for idx in top5_indices
            ]
            
            return {
                "predicted_class": predicted_class,
                "confidence": confidence,
                "top5": top5_results
            }
        except Exception as e:
            logger.error(f"预测失败: {e}")
            raise

def create_app(model_deployer):
    """创建FastAPI应用
    
    Args:
        model_deployer: 模型部署器实例
        
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
                "/health": "健康检查"
            }
        }
    
    @app.get("/health")
    async def health_check():
        """健康检查"""
        return {"status": "healthy"}
    
    @app.post("/predict")
    async def predict(file: UploadFile = File(...)):
        """预测图像类别
        
        Args:
            file: 上传的图像文件
            
        Returns:
            dict: 预测结果
        """
        try:
            # 读取文件内容
            content = await file.read()
            
            # 检查文件类型
            if not file.content_type.startswith('image/'):
                raise HTTPException(status_code=400, detail="只支持图像文件")
            
            # 预测
            result = model_deployer.predict(content)
            
            return JSONResponse(content=result)
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"预测请求失败: {e}")
            raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")
    
    return app

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='模型部署脚本 - 提供角色识别API服务')
    
    parser.add_argument('--model_path', type=str, 
                       default='models_simple/character_classifier_simple_best.pth', 
                       help='模型路径')
    parser.add_argument('--model_type', type=str, default='simple', 
                       choices=['simple', 'resnet'], 
                       help='模型类型')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='服务主机')
    parser.add_argument('--port', type=int, default=8000, help='服务端口')
    
    args = parser.parse_args()
    
    logger.info('开始部署模型...')
    logger.info(f'模型路径: {args.model_path}')
    logger.info(f'模型类型: {args.model_type}')
    logger.info(f'服务地址: {args.host}:{args.port}')
    
    try:
        # 初始化模型部署器
        deployer = ModelDeployer(args.model_path, args.model_type)
        
        # 创建FastAPI应用
        app = create_app(deployer)
        
        # 启动服务
        logger.info('启动API服务...')
        uvicorn.run(app, host=args.host, port=args.port)
    except Exception as e:
        logger.error(f"部署失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
