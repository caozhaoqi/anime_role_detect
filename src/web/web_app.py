#!/usr/bin/env python3
"""
Flask网页应用
实现图片上传和角色分类功能
"""
import os
import sys
import tempfile
from flask import Flask, request, render_template, redirect, url_for, flash
from PIL import Image

# 解决OpenMP冲突问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# 导入通用分类模块
from src.core.general_classification import GeneralClassification, get_classifier

# 初始化Flask应用
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# 确保上传目录存在
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# 允许的文件扩展名
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

def allowed_file(filename):
    """检查文件扩展名是否允许"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def initialize_system():
    """初始化分类系统"""
    print("初始化分类系统...")
    classifier = get_classifier()
    
    # 从所有角色数据构建索引
    data_dir = "data/all_characters"
    if os.path.exists(data_dir):
        success = classifier.build_index_from_directory(data_dir)
        if success:
            print("系统初始化成功!")
        else:
            print("系统初始化失败，使用备用数据...")
            # 尝试使用备用数据
            backup_dir = "data/blue_archive_optimized_v2"
            if os.path.exists(backup_dir):
                classifier.build_index_from_directory(backup_dir)
            else:
                backup_dir = "data/blue_archive_optimized"
                if os.path.exists(backup_dir):
                    classifier.build_index_from_directory(backup_dir)
    else:
        print("数据目录不存在，系统初始化失败")

@app.route('/', methods=['GET', 'POST'])
def index():
    """首页"""
    if request.method == 'POST':
        # 检查是否有文件部分
        if 'file' not in request.files:
            flash('没有文件部分')
            return redirect(request.url)
        
        file = request.files['file']
        
        # 检查用户是否选择了文件
        if file.filename == '':
            flash('没有选择文件')
            return redirect(request.url)
        
        # 检查文件是否允许
        if file and allowed_file(file.filename):
            # 保存文件到临时位置
            temp_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(temp_path)
            
            try:
                # 分类图像
                classifier = get_classifier()
                role, similarity, boxes = classifier.classify_image(temp_path)
                
                # 转换相似度为百分比
                similarity_percent = similarity * 100
                
                # 获取图像信息
                img = Image.open(temp_path)
                img_width, img_height = img.size
                
                # 准备结果
                result = {
                    'filename': file.filename,
                    'role': role if role else '未知',
                    'similarity': similarity_percent,
                    'image_path': file.filename,  # 只使用文件名
                    'image_width': img_width,
                    'image_height': img_height,
                    'boxes': boxes
                }
                
                return render_template('result.html', result=result)
            except Exception as e:
                flash(f'分类失败: {str(e)}')
                return redirect(request.url)
            # 不清理临时文件，以便在结果页面中显示
            # finally:
            #     # 清理临时文件
            #     if os.path.exists(temp_path):
            #         os.remove(temp_path)
    
    # GET请求，显示上传表单
    return render_template('index.html')

@app.route('/about')
def about():
    """关于页面"""
    return render_template('about.html')

@app.route('/api/classify', methods=['POST'])
def api_classify():
    """API分类端点"""
    import json
    
    if 'file' not in request.files:
        return json.dumps({'error': '没有文件部分'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return json.dumps({'error': '没有选择文件'}), 400
    
    if file and allowed_file(file.filename):
        # 保存文件到临时位置
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(temp_path)
        
        try:
            # 分类图像
            classifier = get_classifier()
            role, similarity, boxes = classifier.classify_image(temp_path)
            
            # 准备结果
            result = {
                'filename': file.filename,
                'role': role if role else '未知',
                'similarity': float(similarity),
                'boxes': boxes
            }
            
            return json.dumps(result), 200, {'Content-Type': 'application/json'}
        except Exception as e:
            return json.dumps({'error': str(e)}), 500
        finally:
            # 清理临时文件
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    return json.dumps({'error': '不支持的文件类型'}), 400

# 创建HTML模板
@app.template_filter('format_similarity')
def format_similarity(value):
    """格式化相似度"""
    return f"{value:.2f}%"

if __name__ == '__main__':
    # 初始化系统
    initialize_system()
    
    # 运行应用
    port = 5001
    print("启动Flask应用...")
    print(f"访问地址: http://127.0.0.1:{port}")
    app.run(debug=True, host='0.0.0.0', port=port)
