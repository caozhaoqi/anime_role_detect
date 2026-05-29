#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试新训练模型的API服务
"""

import os
import sys
import json
import torch
from torchvision import models, transforms
from PIL import Image
from fastapi import FastAPI, UploadFile, File

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

app = FastAPI(title="New Model Test API", version="1.0.0")

# 全局变量
model = None
class_names = None
transform = None
device = None

def load_model(model_name="efficientnet_b3_loli_optimized_v2_20260529_133654"):
    """加载新训练的模型"""
    global model, class_names, transform, device
    
    # 获取设备
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    print(f"📦 使用设备: {device}")
    
    # 模型路径
    model_dir = os.path.join(project_root, 'models', model_name)
    config_path = os.path.join(model_dir, 'training_results.json')
    
    # 加载配置
    config = {}
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    
    num_classes = config.get('num_classes', 76)
    image_size = config.get('image_size', 224)
    
    # 加载模型
    model_path = os.path.join(model_dir, 'model_full.pth')
    if not os.path.exists(model_path):
        model_path = os.path.join(model_dir, 'model_best.pth')
    
    print(f"🔄 加载模型: {model_path}")
    
    # 创建模型骨架
    model = models.efficientnet_b3(num_classes=num_classes)
    
    # 加载模型权重
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    if isinstance(checkpoint, torch.nn.Module):
        model = checkpoint
    else:
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict, strict=False)
    
    model = model.to(device)
    model.eval()
    
    # 获取类别名称
    class_names = config.get('class_names', [f"class_{i}" for i in range(num_classes)])
    
    # 定义变换
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    print("✅ 模型加载成功")
    print(f"📋 类别数量: {len(class_names)}")

@app.on_event("startup")
async def startup_event():
    """启动时加载模型"""
    load_model()

@app.get("/api/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy", "model": "efficientnet_b3_loli_optimized_v2"}

@app.post("/api/classify")
async def classify_image(file: UploadFile = File(...)):
    """分类图像"""
    global model, class_names, transform, device
    
    try:
        # 读取文件内容
        content = await file.read()
        
        # 从内存创建PIL图像
        from PIL import Image
        import io
        image = Image.open(io.BytesIO(content)).convert('RGB')
        
        # 预处理图像
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # 推理
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            
            # 获取Top-5结果
            top5_prob, top5_idx = torch.topk(probabilities, 5)
            
            results = []
            for i in range(5):
                class_idx = top5_idx[i].item()
                class_name = class_names[class_idx] if class_idx < len(class_names) else f"class_{class_idx}"
                results.append({
                    "role": class_name,
                    "similarity": round(top5_prob[i].item() * 100, 2)
                })
        
        return {
            "success": True,
            "results": results,
            "top_role": results[0]["role"],
            "top_similarity": results[0]["similarity"]
        }
    
    except Exception as e:
        print(f"❌ 分类失败: {e}")
        return {
            "success": False,
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)
