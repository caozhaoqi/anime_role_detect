from flask_restx import Api, Resource, fields
from flask import Blueprint, redirect, request
from loguru import logger
import os

# 创建一个蓝图用于文档
api_bp = Blueprint('api', __name__, url_prefix='/api')

# 创建 API 实例，配置文档信息
api = Api(
    api_bp,
    version='1.0',
    title='动漫角色识别 API',
    description='基于机器学习的动漫角色识别系统 API 文档',
    doc='/docs/',  # 文档访问路径
    # 移除 prefix，确保路径与实际后端 API 匹配
    # 优化 Swagger UI 配置
    ui_params={
        'docExpansion': 'list',  # 展开所有操作
        'defaultModelRendering': 'model',  # 默认渲染为模型
        'defaultModelsExpandDepth': 2,  # 默认展开深度
        'displayOperationId': True,  # 显示操作 ID
        'displayRequestDuration': True,  # 显示请求持续时间
        'tryItOutEnabled': True,  # 启用 "Try it out" 功能
        'filter': True,  # 启用过滤功能
        'showExtensions': True,  # 显示扩展
        'showCommonExtensions': True,  # 显示常见扩展
    }
)

# 创建分类命名空间（不需要前缀，因为我们直接映射到 /api/classify）
classify_ns = api.namespace('', description='角色分类相关操作')

# 定义响应模型
classify_response_model = api.model('ClassifyResponse', {
    'filename': fields.String(description='文件名'),
    'role': fields.String(description='识别的角色'),
    'similarity': fields.Float(description='相似度'),
    'boxes': fields.List(fields.Raw, description='边界框信息'),
    'fileType': fields.String(description='文件类型 (image/video)'),
    'videoResults': fields.List(fields.Raw, description='视频帧检测结果（仅视频文件）')
})

# 定义分类端点，直接映射到 /api/classify
@classify_ns.route('/classify')
class ClassifyResource(Resource):
    """角色分类 API"""
    
    @classify_ns.doc('classify', 
                    responses={
                        200: '成功',
                        400: '请求错误',
                        500: '服务器错误'
                    },
                    consumes=['multipart/form-data'],  # 支持文件上传
                    params={
                        'file': {'description': '媒体文件（必填，支持图片和视频）', 'in': 'formData', 'required': True, 'type': 'file'},
                        'use_model': {'description': '是否使用专用模型 (true/false, 默认false)', 'in': 'formData', 'type': 'boolean'},
                        'frame_skip': {'description': '视频帧跳过间隔 (默认5)', 'in': 'formData', 'type': 'integer'}
                    }
    )
    @classify_ns.marshal_with(classify_response_model)
    def post(self):
        """上传媒体文件进行角色分类"""
        """
        上传媒体文件（图片或视频）进行角色分类。
        
        - **file**: 媒体文件（必填，支持图片和视频）
        - **use_model**: 是否使用专用模型 (true/false, 默认false)
        - **frame_skip**: 视频帧跳过间隔 (默认5)
        
        返回识别结果，包括角色名称、相似度、边界框等信息。
        """
        # 处理实际的文件上传
        if 'file' in request.files:
            file = request.files['file']
            if file and file.filename:
                # 这里可以添加实际的文件处理逻辑
                # 例如保存文件并调用分类服务
                return {
                    'filename': file.filename,
                    'role': 'genshin_impact_甘雨',
                    'similarity': 0.92,
                    'boxes': [],
                    'fileType': 'image'
                }, 200
        # 如果没有文件，返回错误
        return {'error': '没有文件部分'}, 400

    @classify_ns.doc('get_classify_docs')
    def get(self):
        """获取分类 API 文档"""
        """
        获取分类 API 的详细文档，包括参数说明、响应格式等。
        """
        return {
            'endpoint': '/api/classify',
            'method': 'POST',
            'description': '角色分类API（支持图片和视频）',
            'parameters': {
                'file': '媒体文件（必填，支持图片和视频）',
                'use_model': '是否使用专用模型 (true/false, 默认false)',
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
            'example_image': 'curl -X POST -F "file=@image.jpg" -F "use_model=true" http://localhost:5002/api/classify',
            'example_video': 'curl -X POST -F "file=@video.mp4" -F "frame_skip=10" http://localhost:5002/api/classify'
        }, 200


def setup_docs_routes(app):
    """设置文档路由"""
    # 注册 API 蓝图
    app.register_blueprint(api_bp)
    
    # 添加一个路由，让用户可以通过 /docs 直接重定向到 Swagger UI 文档
    @app.route('/docs')
    def docs():
        """API 文档首页"""
        # 重定向到 Swagger UI 文档
        return redirect('/api/docs/', code=302)
    
    logger.debug("📚 API文档路由已设置，访问路径: http://localhost:5002/docs 和 http://localhost:5002/api/docs/")
