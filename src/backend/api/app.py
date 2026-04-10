from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import hashlib
import time
import torch
import torch.nn as nn
import torchvision.models as models # 新增：建议放到全局
import torchvision.transforms as transforms
from PIL import Image
import aiohttp
import asyncio
from pathlib import Path

# 从环境变量中读取配置
USE_MODEL_SERVICE = os.environ.get('USE_MODEL_SERVICE', 'false').lower() == 'true'
MODEL_SERVICE_URL = os.environ.get('MODEL_SERVICE_URL', 'http://localhost:8001')

# 创建FastAPI应用实例
app = FastAPI(
    title="Anime Role Detect API",
    description="Anime role detection API",
    version="1.0.0"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该设置为具体的前端域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 导入其他模块
from src.core.logging.global_logger import get_logger
from src.utils.image_utils import ImageUtils
from src.core.preprocessing.preprocessing import Preprocessing
from src.core.keypoint.mediapipe_keypoint_detector import MediaPipeKeypointDetector
from src.core.tagging.wd_vit_v3_tagger import WDViTV3Tagger
from src.scripts.ai_role_prediction import AIRolePredictor
from src.utils.cache_manager import CacheManager
from src.core.classification.models import get_model
import os
# 初始化预处理、关键点检测、标签生成和角色预测实例
preprocessor = Preprocessing()
keypoint_detector = MediaPipeKeypointDetector()
tagger = WDViTV3Tagger()
tagger.load_model()
role_predictor = AIRolePredictor()

# 初始化日志记录器和缓存管理器
logger = get_logger("api")
cache_manager = CacheManager()

# 类名列表
class_names = [
    "unknown", "plana", "other"
]

def load_model(model_name):
    """
    加载模型
    
    Args:
        model_name: 模型名称
    
    Returns:
        加载的模型或None
    """
    try:
        logger.info(f"加载模型: {model_name}")
        # 直接创建一个简单的分类模型
        # 
        # .nn as nn
        # vision.models as models
        
        # 创建一个mobilenet_v2模型
        model = models.mobilenet_v2(pretrained=True)
        # 修改分类层，使其输出3个类别
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 3)
        model.eval()
        logger.info(f"使用默认预训练模型: mobilenet_v2")
        return model
    except Exception as e:
        logger.error(f"加载模型失败: {e}")
        import traceback
        logger.error(f"异常堆栈: {traceback.format_exc()}")
        return None

def load_trained_model(model_name):
    """
    加载训练好的模型
    
    Args:
        model_name: 模型名称
    
    Returns:
        模型和类别映射
    """
    
    try:
        logger.info(f"加载训练好的模型: {model_name}")
        
        # 构建模型路径
        model_dir = os.path.join("models", model_name)
        model_path = os.path.join(model_dir, "model_best.pth")
        
        if not os.path.exists(model_path):
            logger.error(f"模型文件不存在: {model_path}")
            # 检查模型目录是否存在
            if not os.path.exists(model_dir):
                logger.warning(f"模型目录不存在: {model_dir}")
            return None
        
        # 确定模型类型
        if model_name == "incremental":
            model_type = "mobilenet_v2_incremental"
        else:
            model_type = model_name
        
        # 加载模型权重
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        
        # 获取类别映射
        class_to_idx = checkpoint.get('class_to_idx', {})
        num_classes = len(class_to_idx)
        
        # 创建模型
        model = get_model(model_type, num_classes)
        
        # 加载权重，使用strict=False以忽略不匹配的键
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model.eval()
        
        logger.info(f"模型加载成功: {model_name}, 类别数: {num_classes}")
        return model, class_to_idx
    except Exception as e:
        logger.error(f"加载训练好的模型失败: {e}")
        import traceback
        logger.error(f"异常堆栈: {traceback.format_exc()}")
        return None

async def process_single_image(file: UploadFile, model_name: str, cache_bypass: bool = False):
    """
    处理单个图像
    
    Args:
        file: 上传的文件
        model_name: 模型名称
        cache_bypass: 是否绕过缓存
    
    Returns:
        处理结果
    """
    import os
    
    temp_path = None
    start_time = time.time()
    
    try:
        # 读取文件内容
        content = await file.read()
        process_time = time.time() - start_time
        logger.debug(f"读取文件耗时: {process_time:.4f}秒")
        
        # 生成文件哈希作为缓存键
        file_hash = hashlib.md5(content).hexdigest()
        cache_key = f"image_processing_{file_hash}_{model_name}"
        
        # 尝试从缓存获取结果
        if not cache_bypass:
            cached_result = cache_manager.get(cache_key)
            if cached_result:
                logger.info(f"缓存命中，直接返回结果: {file.filename}")
                # 添加文件名
                cached_result["filename"] = file.filename
                cached_result["processing_time"] = time.time() - start_time
                return cached_result
        else:
            logger.info(f"缓存绕过，重新处理: {file.filename}")
        
        # 初始化变量
        text_detections = []
        keypoints = []
        ai_predicted_role = None
        
        # 检查是否使用模型服务
        use_model_service = USE_MODEL_SERVICE
        if use_model_service:
            logger.info(f"使用模型服务: {MODEL_SERVICE_URL}")
            
            # 确定文件类型
            content_type = file.content_type
            if content_type is None:
                # import os
                ext = os.path.splitext(file.filename)[1].lower()
                ext_to_content_type = {
                    '.jpg': 'image/jpeg',
                    '.jpeg': 'image/jpeg',
                    '.png': 'image/png',
                    '.gif': 'image/gif',
                    '.bmp': 'image/bmp',
                    '.svg': 'image/svg+xml'
                }
                content_type = ext_to_content_type.get(ext, 'application/octet-stream')
            
            # 构建请求
            files = {'file': (file.filename, content, content_type)}
            data = {
                'model_name': model_name,
                'use_attributes': 'true'
            }
            
            # 发送请求到模型服务
            try:
                logger.info(f"开始调用模型服务: {MODEL_SERVICE_URL}/api/model/predict")
                logger.info(f"请求文件: {file.filename}, 大小: {len(content)}字节, 类型: {content_type}")
                
                # 使用aiohttp进行异步HTTP调用
                async with aiohttp.ClientSession() as session:
                    # 构建multipart/form-data请求
                    form = aiohttp.FormData()
                    form.add_field('file', content, filename=file.filename, content_type=content_type)
                    form.add_field('model_name', model_name)
                    form.add_field('use_attributes', 'true')
                    
                    logger.info(f"准备发送请求到模型服务")
                    async with session.post(
                        f"{MODEL_SERVICE_URL}/api/model/predict",
                        data=form,
                        timeout=30
                    ) as response:
                        logger.info(f"模型服务响应状态码: {response.status}")
                        response.raise_for_status()
                        model_result = await response.json()
                        logger.info(f"模型服务返回数据: {model_result}")
                
                # 处理结果
                role = model_result.get('role', 'unknown')
                similarity = model_result.get('similarity', 0.0)
                attributes = model_result.get('attributes', [])
                feature = model_result.get('feature', None)
                
                logger.info(f"模型服务返回结果: role={role}, similarity={similarity}, has_feature={feature is not None}")
                
                # 保存临时文件用于其他处理
                temp_path = f"temp_{int(time.time())}_{file.filename}"
                with open(temp_path, "wb") as f:
                    f.write(content)
                
                # 如果模型服务返回unknown且提供了特征向量，使用本地模型进行分类
                if role == 'unknown' and feature is not None:
                    logger.info(f"模型服务返回unknown且提供了特征向量，role={role}, feature长度={len(feature) if feature else 'None'}")
                    # 加载训练好的模型
                    
                    model_info = load_trained_model(model_name)
                    logger.info(f"load_trained_model返回: {model_info}")
                    if model_info is not None:
                        model, class_to_idx = model_info
                        idx_to_class = {v: k for k, v in class_to_idx.items()}
                        
                        # 图像预处理
                        transform = transforms.Compose([
                            transforms.Resize((256, 256)),
                            transforms.CenterCrop((224, 224)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        ])
                        
                        # 加载图像
                        img = Image.open(temp_path).convert('RGB')
                        img = transform(img)
                        img = img.unsqueeze(0)  # 添加批次维度
                        
                        # 预测
                        with torch.no_grad():
                            outputs = model(img)
                            _, predicted = torch.max(outputs, 1)
                            confidence = torch.nn.functional.softmax(outputs, dim=1)[0][predicted.item()].item()
                        
                        # 获取预测结果
                        role = idx_to_class.get(predicted.item(), "unknown")
                        similarity = float(confidence)
                        logger.info(f"本地模型分类结果: {role}, 相似度: {similarity:.4f}")
                
                # 检测文本（非SVG文件）
                if file.content_type != "image/svg+xml":
                    text_detections = preprocessor.detect_text(temp_path)
                else:
                    logger.info("跳过SVG文件的文本检测")
                    text_detections = []
                
                # 处理关键点检测和角色预测
                if file.content_type != "image/svg+xml":
                    keypoints = keypoint_detector.detect_keypoints(temp_path)
                    ai_predicted_role = role_predictor.predict_role(attributes)
                else:
                    logger.info("跳过SVG文件的关键点检测")
                    ai_predicted_role = role_predictor.predict_role(attributes)
            except Exception as e:
                logger.error(f"调用模型服务失败: {e}")
                logger.error(f"异常类型: {type(e).__name__}")
                import traceback
                logger.error(f"异常堆栈: {traceback.format_exc()}")
                # 回退到本地处理
                use_model_service = False
        
        # 如果不使用模型服务或调用失败，使用本地处理
        if not use_model_service:
            # 验证图像
            validate_start = time.time()
            is_valid = ImageUtils.validate_image(content)
            validate_time = time.time() - validate_start
            logger.debug(f"验证图像耗时: {validate_time:.4f}秒, 结果: {is_valid}")
            
            # 保存临时文件
            temp_path = f"temp_{int(time.time())}_{file.filename}"
            with open(temp_path, "wb") as f:
                f.write(content)
            
            # 检查是否使用新训练的模型
            trained_model_names = ["mobilenet_v2", "efficientnet_b0", "efficientnet_b3", "resnet50", "incremental"]
            
            if model_name in trained_model_names:
                logger.info(f"使用新训练的模型: {model_name}")
                # 加载训练好的模型
                
                model_info = load_trained_model(model_name)
                if model_info is None:
                    result = {"role": "unknown", "similarity": 0.0, "attributes": []}
                    # 缓存结果
                    cache_manager.set(result, cache_key, ttl=3600)
                    result["processing_time"] = time.time() - start_time
                    return result
                
                model, class_to_idx = model_info
                idx_to_class = {v: k for k, v in class_to_idx.items()}
                
                # 图像预处理
                transform = transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.CenterCrop((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                
                # 加载图像
                try:
                    logger.info(f"加载图像: {temp_path}")
                    img = Image.open(temp_path).convert('RGB')
                    logger.info(f"图像加载成功，大小: {img.size}")
                    
                    # 应用变换
                    img = transform(img)
                    logger.info(f"图像变换成功，形状: {img.shape}")
                    
                    img = img.unsqueeze(0)  # 添加批次维度
                    logger.info(f"添加批次维度后，形状: {img.shape}")
                    
                    # 预测
                    with torch.no_grad():
                        logger.info("开始模型预测...")
                        outputs = model(img)
                        logger.info(f"模型输出形状: {outputs.shape}")
                        _, predicted = torch.max(outputs, 1)
                        confidence = torch.nn.functional.softmax(outputs, dim=1)[0][predicted.item()].item()
                    logger.info(f"预测完成，预测类别: {predicted.item()}, 置信度: {confidence}")
                    
                    # 获取预测结果
                    role = idx_to_class.get(predicted.item(), "unknown")
                    similarity = float(confidence)
                    logger.info(f"预测角色: {role}, 相似度: {similarity}")
                    
                    # 处理文本检测、标签生成
                    
                    # 文本检测
                    try:
                        if file.content_type != "image/svg+xml":
                            logger.info("开始文本检测...")
                            text_detections = preprocessor.detect_text(temp_path)
                            logger.info(f"文本检测完成，检测到 {len(text_detections)} 个文本")
                        else:
                            text_detections = []
                            logger.info("SVG图像，跳过文本检测")
                    except Exception as e:
                        logger.error(f"文本检测失败: {e}")
                        text_detections = []
                    
                    # 标签生成
                    try:
                        logger.info("开始标签生成...")
                        attributes = tagger.generate_tags(temp_path)
                        logger.info(f"标签生成完成，生成 {len(attributes)} 个标签")
                    except Exception as e:
                        logger.error(f"标签生成失败: {e}")
                        attributes = []
                    
                    # 关键点检测
                    try:
                        if file.content_type != "image/svg+xml":
                            logger.info("开始关键点检测...")
                            keypoints = keypoint_detector.detect_keypoints(temp_path)
                            logger.info(f"关键点检测完成，检测到 {len(keypoints)} 个关键点")
                        else:
                            keypoints = []
                            logger.info("SVG图像，跳过关键点检测")
                    except Exception as e:
                        logger.error(f"关键点检测失败: {e}")
                        keypoints = []
                    
                    # 角色预测
                    try:
                        logger.info("开始角色预测...")
                        ai_predicted_role = role_predictor.predict_role([])
                        logger.info(f"角色预测完成，预测角色: {ai_predicted_role}")
                    except Exception as e:
                        logger.error(f"角色预测失败: {e}")
                        ai_predicted_role = None
                except Exception as e:
                    logger.error(f"图像处理失败: {e}")
                    import traceback
                    logger.error(f"异常堆栈: {traceback.format_exc()}")
                    raise
            else:
                # 使用传统模型
                logger.info(f"使用传统模型: {model_name}")
                
                # 检查模型是否存在
                model_path = f"models/{model_name}"
                if not os.path.exists(model_path):
                    logger.warning(f"模型目录不存在: {model_path}，使用默认预训练模型")
                
                # 直接创建模型
                try:
                    logger.info("创建默认预训练模型: mobilenet_v2")
                    
                    # .nn as nn
                    # vision.models as models
                    
                    # 创建一个mobilenet_v2模型
                    model = models.mobilenet_v2(pretrained=True)
                    # 修改分类层，使其输出3个类别
                    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 3)
                    model.eval()
                    logger.info("默认预训练模型创建成功")
                except Exception as e:
                    logger.error(f"创建模型失败: {e}")
                    import traceback
                    logger.error(f"异常堆栈: {traceback.format_exc()}")
                    result = {"role": "unknown", "similarity": 0.0, "attributes": []}
                    # 缓存结果
                    cache_manager.set(result, cache_key, ttl=3600)
                    result["processing_time"] = time.time() - start_time
                    return result
                
                # 图像预处理
                transform = transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.CenterCrop((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                
                # 加载图像
                img = Image.open(temp_path).convert('RGB')
                img = transform(img)
                img = img.unsqueeze(0)  # 添加批次维度
                
                # 预测
                with torch.no_grad():
                    outputs = model(img)
                    _, predicted = torch.max(outputs, 1)
                    confidence = torch.nn.functional.softmax(outputs, dim=1)[0][predicted.item()].item()
                
                # 获取预测结果
                role = class_names[predicted.item()]
                similarity = float(confidence)
                
                # 处理文本检测、标签生成
                
                # 文本检测
                if file.content_type != "image/svg+xml":
                    text_detections = preprocessor.detect_text(temp_path)
                else:
                    text_detections = []
                
                # 标签生成
                attributes = tagger.generate_tags(temp_path)
                
                # 关键点检测
                if file.content_type != "image/svg+xml":
                    keypoints = keypoint_detector.detect_keypoints(temp_path)
                else:
                    keypoints = []
                
                # 角色预测
                ai_predicted_role = role_predictor.predict_role([])
        
        # 构建结果
        result = {
            "role": role,
            "similarity": similarity,
            "attributes": attributes,
            "text_detections": text_detections,
            "keypoints": keypoints,
            "ai_predicted_role": ai_predicted_role
        }
        
        # 缓存结果
        cache_manager.set(result, cache_key, ttl=1800)
        
        # 清理临时文件
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as e:
                logger.error(f"删除临时文件失败: {e}")
        
        # 记录处理时间
        result["processing_time"] = time.time() - start_time
        
        return result
    except Exception as e:
        logger.error(f"处理图像失败: {e}")
        # 清理临时文件
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as e:
                logger.error(f"删除临时文件失败: {e}")
        return {"role": "unknown", "similarity": 0.0, "attributes": [], "error": str(e)}

@app.get("/api/health")
async def health_check():
    """
    健康检查端点
    """
    return {"status": "healthy", "service": "Anime Role Detect API"}

@app.get("/api/models")
async def list_models():
    """返回可用模型列表"""
    models_dir = Path("models")

    default_models = [
        {"name": "default", "path": "", "description": "默认分类模型", "available": True},
        {"name": "mobilenet_v2", "path": "models/mobilenet_v2", "description": "MobileNetV2 模型", "available": False},
        {"name": "efficientnet_b0", "path": "models/efficientnet_b0", "description": "EfficientNet-B0 模型", "available": False},
        {"name": "efficientnet_b3", "path": "models/efficientnet_b3", "description": "EfficientNet-B3 模型", "available": False},
        {"name": "resnet50", "path": "models/resnet50", "description": "ResNet50 模型", "available": False},
    ]

    available_names = set()
    if models_dir.exists() and models_dir.is_dir():
        for model_dir in models_dir.iterdir():
            if model_dir.is_dir():
                available_names.add(model_dir.name)

    models = []
    for model in default_models:
        model_copy = dict(model)
        if model_copy["name"] != "default":
            model_copy["available"] = model_copy["name"] in available_names
        models.append(model_copy)

    for extra_name in sorted(available_names):
        if extra_name not in {m["name"] for m in models}:
            models.append(
                {
                    "name": extra_name,
                    "path": f"models/{extra_name}",
                    "description": f"{extra_name} 模型",
                    "available": True,
                }
            )

    return {"models": models}

@app.post("/api/classify")
async def classify(
    file: UploadFile = File(...),
    model_name: str = Form(...),
    cache_bypass: bool = Form(False)
):
    """
    图像分类端点
    
    Args:
        file: 上传的文件
        model_name: 模型名称
        cache_bypass: 是否绕过缓存
    
    Returns:
        分类结果
    """
    try:
        result = await process_single_image(file, model_name, cache_bypass)
        return result
    except Exception as e:
        logger.error(f"分类失败: {e}")
        raise HTTPException(status_code=500, detail=f"分类失败: {str(e)}")

async def process_multiple_characters(file: UploadFile, model_name: str, max_characters: int = 5):
    """
    处理图像中的多个角色
    
    Args:
        file: 上传的文件
        model_name: 模型名称
        max_characters: 最大检测角色数
    
    Returns:
        多角色检测结果
    """
    import os
    temp_path = None
    start_time = time.time()
    
    try:
        # 读取文件内容
        content = await file.read()
        process_time = time.time() - start_time
        logger.debug(f"读取文件耗时: {process_time:.4f}秒")
        
        # 检查是否使用模型服务
        use_model_service = USE_MODEL_SERVICE
        if use_model_service:
            logger.info(f"使用模型服务进行多角色检测: {MODEL_SERVICE_URL}")
            
            # 确定文件类型
            content_type = file.content_type
            if content_type is None:
                import os
                ext = os.path.splitext(file.filename)[1].lower()
                ext_to_content_type = {
                    '.jpg': 'image/jpeg',
                    '.jpeg': 'image/jpeg',
                    '.png': 'image/png',
                    '.gif': 'image/gif',
                    '.bmp': 'image/bmp',
                    '.svg': 'image/svg+xml'
                }
                content_type = ext_to_content_type.get(ext, 'application/octet-stream')
            
            # 发送请求到模型服务的多角色检测API
            try:
                logger.info(f"开始调用模型服务的多角色检测API: {MODEL_SERVICE_URL}/api/model/detect-multiple")
                
                # 使用aiohttp进行异步HTTP调用
                async with aiohttp.ClientSession() as session:
                    # 构建multipart/form-data请求
                    form = aiohttp.FormData()
                    form.add_field('file', content, filename=file.filename, content_type=content_type)
                    form.add_field('max_characters', str(max_characters))
                    
                    async with session.post(
                        f"{MODEL_SERVICE_URL}/api/model/detect-multiple",
                        data=form,
                        timeout=30
                    ) as response:
                        logger.info(f"模型服务响应状态码: {response.status}")
                        response.raise_for_status()
                        model_result = await response.json()
                        logger.info(f"模型服务返回数据: {model_result}")
                
                # 处理结果
                total_characters = model_result.get('total_characters', 0)
                characters = model_result.get('characters', [])
                
                logger.info(f"多角色检测完成，检测到 {total_characters} 个角色")
                
                # 保存临时文件用于其他处理
                temp_path = f"temp_{int(time.time())}_{file.filename}"
                with open(temp_path, "wb") as f:
                    f.write(content)
                
                # 对每个角色进行进一步处理
                processed_characters = []
                for i, char in enumerate(characters):
                    # 检测文本
                    text_detections = []
                    if content_type != "image/svg+xml":
                        # 这里可以为每个角色单独检测文本，但需要裁剪图像
                        # 暂时使用整个图像的文本检测
                        text_detections = preprocessor.detect_text(temp_path)
                    
                    # 构建处理后的角色信息
                    processed_char = {
                        "id": char.get("id", i + 1),
                        "box": char.get("box"),
                        "confidence": char.get("confidence"),
                        "attributes": char.get("attributes", []),
                        "text_detections": text_detections
                    }
                    processed_characters.append(processed_char)
                
                # 构建最终结果
                result = {
                    "total_characters": total_characters,
                    "characters": processed_characters,
                    "filename": file.filename,
                    "processing_time": time.time() - start_time
                }
                
                return result
            except Exception as e:
                logger.error(f"调用模型服务失败: {e}")
                # 回退到本地处理
                use_model_service = False
        
        # 如果不使用模型服务或调用失败，使用本地处理
        if not use_model_service:
            logger.info("使用本地处理进行多角色检测")
            
            # 保存临时文件
            temp_path = f"temp_{int(time.time())}_{file.filename}"
            with open(temp_path, "wb") as f:
                f.write(content)
            
            # 使用本地预处理器检测多个角色
            characters = preprocessor.process_multiple_characters(temp_path, max_characters=max_characters)
            
            # 处理结果
            processed_characters = []
            for i, char in enumerate(characters):
                # 检测文本
                text_detections = []
                if file.content_type != "image/svg+xml":
                    text_detections = preprocessor.detect_text(temp_path)
                
                # 构建处理后的角色信息
                processed_char = {
                    "id": i + 1,
                    "box": char.get("box"),
                    "confidence": char.get("confidence"),
                    "attributes": [],  # 本地处理暂时不生成属性
                    "text_detections": text_detections
                }
                processed_characters.append(processed_char)
            
            # 构建最终结果
            result = {
                "total_characters": len(processed_characters),
                "characters": processed_characters,
                "filename": file.filename,
                "processing_time": time.time() - start_time
            }
            
            return result
    except Exception as e:
        logger.error(f"多角色处理失败: {e}")
        raise HTTPException(status_code=500, detail=f"多角色处理失败: {str(e)}")
    finally:
        # 清理临时文件
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as e:
                logger.error(f"删除临时文件失败: {e}")

@app.post("/api/classify-multiple")
async def classify_multiple(
    file: UploadFile = File(...),
    model_name: str = Form(...),
    max_characters: int = Form(5)
):
    """
    多角色检测端点
    
    Args:
        file: 上传的文件
        model_name: 模型名称
        max_characters: 最大检测角色数
    
    Returns:
        多角色检测结果
    """
    try:
        result = await process_multiple_characters(file, model_name, max_characters)
        return result
    except Exception as e:
        logger.error(f"多角色检测失败: {e}")
        raise HTTPException(status_code=500, detail=f"多角色检测失败: {str(e)}")
