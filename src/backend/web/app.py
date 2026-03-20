#!/usr/bin/env python3
"""
Flask网页应用
实现图片上传和角色分类功能
"""
import os
import sys
from flask import Flask
from flask_cors import CORS
from loguru import logger

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# 导入配置
from src.backend.web.config.config import SECRET_KEY, UPLOAD_FOLDER, MAX_CONTENT_LENGTH

# 导入模块
from src.backend.web.models.coreml_model import load_coreml_model
from src.backend.web.routes.web_routes import setup_web_routes
from src.backend.web.routes.api_routes import setup_api_routes
from src.backend.web.routes.docs_routes import setup_docs_routes
from src.backend.web.services.classification_service import initialize_system

# 初始化Flask应用
app = Flask(__name__)
app.config['SECRET_KEY'] = SECRET_KEY
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# 启用CORS，允许跨域请求
CORS(app)
logger.debug("🌐 CORS已启用，允许跨域请求")

# 确保上传目录存在
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# 加载Core ML模型
coreml_model = load_coreml_model()

# 初始化分类系统
initialize_system()

# 设置路由
setup_web_routes(app)
setup_api_routes(app)
setup_docs_routes(app)

if __name__ == '__main__':
    # 启动应用
    logger.info("🚀 启动Flask应用...")
    app.run(debug=True, host='0.0.0.0', port=5002)
