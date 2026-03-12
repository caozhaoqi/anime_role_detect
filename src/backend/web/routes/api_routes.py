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
            # 可用模型列表
            available_models = [
                {"name": "default", "path": "", "description": "默认分类模型", "available": True},
                {"name": "efficientnet", "path": "models/efficientnet", "description": "EfficientNet-B0模型", "available": True},
                {"name": "efficientnet_b3", "path": "models/efficientnet_b3", "description": "EfficientNet-B3模型", "available": True}
            ]
            
            return json.dumps({"models": available_models}, ensure_ascii=False), 200, {'Content-Type': 'application/json'}
        except Exception as e:
            logger.error(f"获取模型列表失败: {e}")
            return json.dumps({'error': f'获取模型列表失败: {str(e)}'}), 500
    
    @app.route('/api/classify', methods=['POST'])
    def api_classify():
        """API分类端点"""
        try:
            if 'file' not in request.files:
                return json.dumps({'error': '没有文件部分'}), 400

            file = request.files['file']
            if file.filename == '':
                return json.dumps({'error': '没有选择文件'}), 400

            if not allowed_file(file.filename):
                return json.dumps({'error': '不支持的文件类型'}), 400

            # 保存文件到临时位置
            temp_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(temp_path)

            # 获取参数
            use_model = request.form.get('use_model') == 'true'
            model_name = request.form.get('model_name', 'default')

            # 分类图像
            role, similarity, boxes, mode, attributes, text_detections = classify_image(temp_path, use_model=use_model, model_name=model_name)

            # 构建响应
            response = {
                'filename': file.filename,
                'role': role if role else '未知',
                'similarity': similarity,
                'boxes': boxes,
                'fileType': 'image',
                'mode': mode
            }

            if attributes:
                response['attributes'] = attributes
            
            if text_detections:
                response['text_detections'] = text_detections

            return json.dumps(response, ensure_ascii=False), 200, {'Content-Type': 'application/json'}
        except Exception as e:
            logger.error(f"分类失败: {e}")
            return json.dumps({'error': f'分类失败: {str(e)}'}), 500