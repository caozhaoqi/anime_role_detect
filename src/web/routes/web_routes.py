from flask import render_template, redirect, url_for, flash, request
import os
from src.web.config.config import IS_DARWIN
from src.web.utils.file_utils import allowed_file
from src.web.services.classification_service import classify_image, get_image_info
from src.web.models.coreml_model import coreml_model

# 使用全局日志系统
from src.core.logging.global_logger import get_logger, log_system, log_inference, log_error
logger = get_logger("web_routes")


def setup_web_routes(app):
    """设置 Web 路由"""
    
    @app.route('/', methods=['GET', 'POST'])
    def index():
        """首页"""
        if request.method == 'POST':
            logger.info("收到首页POST请求")
            
            # 检查是否有文件部分
            if 'file' not in request.files:
                logger.warning("没有文件部分")
                flash('没有文件部分')
                return redirect(request.url)

            file = request.files['file']
            logger.info(f"收到文件: {file.filename}")
            
            # 在Mac平台默认使用Core ML模型
            if IS_DARWIN and coreml_model is not None:
                use_coreml = True
            else:
                use_coreml = 'use_coreml' in request.form and request.form['use_coreml'] == 'true'

            use_model = 'use_model' in request.form and request.form['use_model'] == 'true'
            
            logger.info(f"分类参数: use_coreml={use_coreml}, use_model={use_model}")

            # 检查Core ML模型是否可用
            if use_coreml and coreml_model is None:
                logger.warning('Core ML模型不可用，将使用默认模型')
                flash('Core ML模型不可用，将使用默认模型')
                use_coreml = False

            # 检查用户是否选择了文件
            if file.filename == '':
                logger.warning('没有选择文件')
                flash('没有选择文件')
                return redirect(request.url)

            # 检查文件是否允许
            if file and allowed_file(file.filename):
                # 保存文件到临时位置
                temp_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                logger.info(f"保存文件到: {temp_path}")
                file.save(temp_path)

                try:
                    # 分类图像
                    logger.info("开始分类图像")
                    role, similarity, boxes, mode = classify_image(temp_path, use_coreml=use_coreml, use_model=use_model)
                    
                    # 记录推理结果
                    log_inference(f"图像分类成功: {file.filename}, 角色: {role}, 相似度: {similarity:.4f}, 模式: {mode}")

                    # 转换相似度为百分比
                    similarity_percent = similarity * 100

                    # 获取图像信息
                    img_width, img_height = get_image_info(temp_path)

                    # 准备结果
                    result = {
                        'filename': file.filename,
                        'role': role if role else '未知',
                        'similarity': similarity_percent,
                        'image_path': file.filename,  # 只使用文件名
                        'image_width': img_width,
                        'image_height': img_height,
                        'boxes': boxes,
                        'mode': mode
                    }
                    
                    logger.info(f"分类完成，结果: {result}")
                    return render_template('result.html', result=result)
                except ValueError as e:
                    logger.error(f'系统错误: {str(e)}')
                    flash(f'系统错误: {str(e)}')
                    return redirect(request.url)
                except Exception as e:
                    logger.error(f'分类失败: {str(e)}')
                    log_error(f'分类失败: {str(e)}')
                    flash(f'分类失败: {str(e)}')
                    return redirect(request.url)

        # GET请求，显示上传表单
        logger.debug("收到首页GET请求")
        return render_template('index.html')

    @app.route('/about')
    def about():
        """关于页面"""
        logger.debug("收到关于页面请求")
        return render_template('about.html')

    @app.route('/monitoring')
    def monitoring():
        """性能监控页面"""
        logger.debug("收到性能监控页面请求")
        return render_template('monitoring.html')
