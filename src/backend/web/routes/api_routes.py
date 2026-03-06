import json
import os
import sys
from flask import request

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from backend.web.utils.file_utils import allowed_file, allowed_video_file
from backend.web.services.classification_service import classify_image
from backend.web.config.config import DEFAULT_FRAME_SKIP
from backend.web.models.coreml_model import coreml_model

# 使用全局日志系统
from core.logging.global_logger import get_logger, log_system, log_inference, log_error
logger = get_logger("api_routes")


def setup_api_routes(app):
    """设置 API 路由"""
    
    @app.route('/api/models', methods=['GET'])
    def api_get_models():
        """获取可用模型列表"""
        try:
            # 定义模型目录
            models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'models')
            
            # 可用模型列表
            available_models = [
                {"name": "default", "path": "", "description": "默认分类模型"},
                {"name": "augmented_training", "path": "models/augmented_training", "description": "增强训练模型"},
                {"name": "arona_plana", "path": "models/arona_plana", "description": "阿罗娜普拉娜模型"},
                {"name": "arona_plana_efficientnet", "path": "models/arona_plana_efficientnet", "description": "EfficientNet模型"},
                {"name": "arona_plana_resnet18", "path": "models/arona_plana_resnet18", "description": "ResNet18模型"},
                {"name": "optimized", "path": "models/optimized", "description": "优化模型"}
            ]
            
            # 检查模型文件是否存在
            for model in available_models:
                if model["path"]:
                    model_path = os.path.join(models_dir, os.path.basename(model["path"]))
                    if os.path.exists(model_path):
                        model["available"] = True
                    else:
                        model["available"] = False
                else:
                    model["available"] = True
            
            logger.debug(f"获取模型列表成功: {available_models}")
            return json.dumps({"models": available_models}, ensure_ascii=False), 200, {'Content-Type': 'application/json'}
        except Exception as e:
            logger.error(f"获取模型列表失败: {e}")
            return json.dumps({'error': f'获取模型列表失败: {str(e)}'}), 500
    
    @app.route('/api/classify', methods=['GET', 'POST'])
    def api_classify():
        """API分类端点（支持图片和视频）"""
        import json

        if request.method == 'GET':
            # GET 请求返回 API 文档
            api_doc = {
                'endpoint': '/api/classify',
                'method': 'POST',
                'description': '角色分类API（支持图片和视频）',
                'parameters': {
                    'file': '媒体文件（必填，支持图片和视频）',
                    'use_model': '是否使用专用模型 (true/false, 默认false)',
                    'use_deepdanbooru': '是否使用集成DeepDanbooru的分类方法 (true/false, 默认false)',
                    'frame_skip': '视频帧跳过间隔 (默认5)'
                },
                'response': {
                    'filename': '文件名',
                    'role': '识别的角色',
                    'similarity': '相似度',
                    'boxes': '边界框信息',
                    'fileType': '文件类型 (image/video)',
                    'videoResults': '视频帧检测结果（仅视频文件）'
                },
                'example_image': 'curl -X POST -F "file=@image.jpg" -F "use_model=true" http://localhost:5001/api/classify',
                'example_video': 'curl -X POST -F "file=@video.mp4" -F "frame_skip=10" http://localhost:5001/api/classify'
            }
            return json.dumps(api_doc, ensure_ascii=False), 200, {'Content-Type': 'application/json'}

        logger.debug("\n" + "="*80)
        logger.debug("🚀 收到API分类请求")
        logger.debug("="*80)
        
        # POST 请求处理媒体文件分类
        logger.debug("📋 请求方法:", request.method)
        logger.debug("📋 请求头:", dict(request.headers))
        logger.debug("📋 表单数据:", dict(request.form))
        
        if 'file' not in request.files:
            logger.debug("❌ 请求中没有文件")
            return json.dumps({'error': '没有文件部分'}), 400

        file = request.files['file']
        logger.debug("📋 收到文件:", file.filename)
        logger.debug("📋 文件类型:", file.content_type)
        logger.debug("📋 文件大小:", file.content_length)
        
        use_model = request.form.get('use_model') == 'true'
        use_coreml = request.form.get('use_coreml') == 'true'
        use_deepdanbooru = request.form.get('use_deepdanbooru') == 'true'
        use_attributes = request.form.get('use_attributes') == 'true'
        model_name = request.form.get('model_name', 'default')
        frame_skip = int(request.form.get('frame_skip', str(DEFAULT_FRAME_SKIP)))
        
        logger.debug("📋 参数:", {
            'use_model': use_model,
            'use_coreml': use_coreml,
            'use_deepdanbooru': use_deepdanbooru,
            'use_attributes': use_attributes,
            'model_name': model_name,
            'frame_skip': frame_skip
        })

        # 检查Core ML模型是否可用
        if use_coreml and coreml_model is None:
            logger.debug("❌ Core ML模型不可用")
            return json.dumps({'error': 'Core ML模型不可用'}), 400

        if file.filename == '':
            logger.debug("❌ 没有选择文件")
            return json.dumps({'error': '没有选择文件'}), 400

        # 检查文件类型
        is_video = allowed_video_file(file.filename)
        is_image = allowed_file(file.filename)
        
        logger.debug("📋 文件类型检查:", {
            'is_video': is_video,
            'is_image': is_image
        })

        if not (is_image or is_video):
            logger.debug("❌ 不支持的文件类型")
            return json.dumps({'error': '不支持的文件类型'}), 400

        # 保存文件到临时位置
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        logger.debug("📁 保存文件到:", temp_path)
        file.save(temp_path)
        logger.debug("✅ 文件保存成功")

        try:
            if is_image:
                # 处理图像文件
                logger.debug("🖼️ 开始处理图像文件")
                if use_coreml:
                    # 使用Core ML模型
                    logger.debug("🤖 使用Core ML模型")
                    role, similarity, boxes, mode, attributes = classify_image(temp_path, use_coreml=True, use_model=False, use_deepdanbooru=False, use_attributes=False)
                elif use_deepdanbooru:
                    # 使用集成DeepDanbooru的分类方法
                    logger.debug("🤖 使用集成DeepDanbooru的分类方法")
                    role, similarity, boxes, mode, attributes = classify_image(temp_path, use_coreml=False, use_model=False, use_deepdanbooru=True, use_attributes=False)
                else:
                    # 使用默认模型
                    logger.debug("🤖 使用默认模型")
                    role, similarity, boxes, mode, attributes = classify_image(temp_path, use_coreml=False, use_model=use_model, use_deepdanbooru=False, use_attributes=use_attributes)
                
                # 构建响应
                response = {
                    'filename': file.filename,
                    'role': role if role else '未知',
                    'similarity': similarity,
                    'boxes': boxes,
                    'fileType': 'image',
                    'mode': mode
                }
                
                # 添加属性标签
                if attributes:
                    response['attributes'] = attributes
                
                # 记录推理结果
                log_inference(f"✅ 图像分类成功: {file.filename}, 角色: {role}, 相似度: {similarity:.4f}, 模式: {mode}, 属性: {len(attributes)}个")
                logger.debug(f"✅ 图像分类成功: {response}")
                return json.dumps(response, ensure_ascii=False), 200, {'Content-Type': 'application/json'}
                
            elif is_video:
                # 处理视频文件
                logger.debug("🎥 开始处理视频文件")
                video_results, overall_role, overall_similarity = process_video(temp_path, frame_skip=frame_skip)
                
                # 构建响应
                response = {
                    'filename': file.filename,
                    'role': overall_role if overall_role else '未知',
                    'similarity': overall_similarity,
                    'boxes': [],  # 视频处理不返回边界框
                    'fileType': 'video',
                    'videoResults': video_results
                }
                
                # 记录推理结果
                log_inference(f"✅ 视频分类成功: {file.filename}, 角色: {overall_role}, 相似度: {overall_similarity:.4f}, 帧处理数: {len(video_results)}")
                logger.debug(f"✅ 视频分类成功: {response}")
                return json.dumps(response, ensure_ascii=False), 200, {'Content-Type': 'application/json'}
                
        except Exception as e:
            # 记录错误
            error_msg = f"❌ 分类失败: {str(e)}"
            log_error(error_msg)
            logger.error(error_msg)
            return json.dumps({'error': f'分类失败: {str(e)}'}), 500

import os
