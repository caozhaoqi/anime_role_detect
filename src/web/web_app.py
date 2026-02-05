#!/usr/bin/env python3
"""
Flask网页应用
实现图片上传和角色分类功能
"""
import os
import sys
import tempfile
import platform
import numpy as np
from flask import Flask, request, render_template, redirect, url_for, flash
from PIL import Image

# 解决OpenMP冲突问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# 导入通用分类模块
from src.core.general_classification import GeneralClassification, get_classifier
# 导入日志记录模块
from src.core.log_fusion.log_recorder import record_classification_log

# Core ML模型加载（仅在macOS上）
coreml_model = None
if platform.system() == 'Darwin':
    try:
        import coremltools

        coreml_model_path = os.path.join('models', 'character_classifier_best_improved.mlpackage')
        if os.path.exists(coreml_model_path):
            coreml_model = coremltools.models.MLModel(coreml_model_path)
            print(f"Core ML模型已加载: {coreml_model_path}")
    except ImportError:
        print("coremltools未安装，Core ML功能不可用")
    except Exception as e:
        print(f"Core ML模型加载失败: {e}")

# 初始化Flask应用
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = os.path.join('src', 'web', 'static', 'uploads')  # 修正上传路径
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# 确保上传目录存在
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# 允许的文件扩展名
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}


def allowed_file(filename):
    """检查文件扩展名是否允许"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def classify_with_coreml(image_path):
    """使用Core ML模型进行分类
    
    Args:
        image_path: 图像路径
    
    Returns:
        (role, similarity, boxes): 角色名称、相似度、边界框
    """
    import json
    import numpy as np

    # 加载类别映射
    mapping_path = os.path.join('models', 'character_classifier_best_improved_class_mapping.json')
    idx_to_class = None
    if os.path.exists(mapping_path):
        with open(mapping_path, 'r') as f:
            mapping = json.load(f)
            idx_to_class = mapping['idx_to_class']

    # 加载并预处理图像
    image = Image.open(image_path).convert('RGB')
    image = image.resize((224, 224))

    # Core ML推理
    try:
        output = coreml_model.predict({'input': image})

        # 获取预测结果
        if 'var_874' in output:
            predictions = output['var_874']
        elif 'output' in output:
            predictions = output['output']
        else:
            # 尝试找到输出键
            output_keys = [k for k in output.keys() if k != 'input']
            if output_keys:
                predictions = output[output_keys[0]]
            else:
                raise ValueError("无法找到Core ML模型输出")

        # 获取最高概率的类别
        if len(predictions.shape) == 2:
            predictions = predictions[0]

        # 应用softmax转换为概率
        exp_predictions = np.exp(predictions - np.max(predictions))  # 数值稳定
        probabilities = exp_predictions / np.sum(exp_predictions)

        predicted_idx = int(np.argmax(probabilities))
        similarity = float(probabilities[predicted_idx])

        # 转换为角色名称
        if idx_to_class:
            # 尝试将predicted_idx转换为字符串查找
            if str(predicted_idx) in idx_to_class:
                role = idx_to_class[str(predicted_idx)]
            else:
                role = f"类别_{predicted_idx}"
        else:
            role = f"类别_{predicted_idx}"

        # Core ML模型不提供边界框信息
        boxes = []

        return role, similarity, boxes

    except Exception as e:
        raise ValueError(f"Core ML推理失败: {e}")


def initialize_system():
    """初始化分类系统"""
    print("初始化分类系统...")
    # 这里只负责初始化，具体的索引加载由 GeneralClassification 内部处理
    # 默认加载 'role_index'
    classifier = get_classifier(index_path="role_index")
    classifier.initialize()


@app.route('/', methods=['GET', 'POST'])
def index():
    """首页"""
    if request.method == 'POST':
        # 检查是否有文件部分
        if 'file' not in request.files:
            flash('没有文件部分')
            return redirect(request.url)

        file = request.files['file']
        # 在Mac平台默认使用Core ML模型
        if platform.system() == 'Darwin' and coreml_model is not None:
            use_coreml = True
        else:
            use_coreml = 'use_coreml' in request.form and request.form['use_coreml'] == 'true'

        use_model = 'use_model' in request.form and request.form['use_model'] == 'true'

        # 检查Core ML模型是否可用
        if use_coreml and coreml_model is None:
            flash('Core ML模型不可用，将使用默认模型')
            use_coreml = False

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
                if use_coreml:
                    # 使用Core ML模型
                    role, similarity, boxes = classify_with_coreml(temp_path)
                    mode = 'Core ML模型 (Apple设备)'
                    # 记录分类日志
                    record_classification_log(
                        image_path=temp_path,
                        role=role,
                        similarity=similarity,
                        feature=[],  # Core ML模型不提供特征向量
                        boxes=boxes,
                        metadata={'mode': mode, 'use_coreml': True}
                    )
                else:
                    # 使用默认模型
                    classifier = get_classifier()
                    role, similarity, boxes = classifier.classify_image(temp_path, use_model=use_model)
                    mode = '专用模型 (EfficientNet)' if use_model else '通用索引 (CLIP)'
                    # 记录分类日志
                    record_classification_log(
                        image_path=temp_path,
                        role=role,
                        similarity=similarity,
                        feature=[],  # 简化处理，不记录特征向量
                        boxes=boxes,
                        metadata={'mode': mode, 'use_model': use_model}
                    )

                # 安全检查：处理无穷大或无效值
                if similarity is None or not isinstance(similarity, (int, float)):
                    similarity = 0.0
                elif np.isinf(similarity) or np.isnan(similarity):
                    similarity = 0.0

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
                    'boxes': boxes,
                    'mode': mode
                }

                return render_template('result.html', result=result)
            except ValueError as e:
                flash(f'系统错误: {str(e)}')
                return redirect(request.url)
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


@app.route('/monitoring')
def monitoring():
    """性能监控页面"""
    return render_template('monitoring.html')


@app.route('/api/classify', methods=['GET', 'POST'])
def api_classify():
    """API分类端点"""
    import json

    if request.method == 'GET':
        # GET 请求返回 API 文档
        api_doc = {
            'endpoint': '/api/classify',
            'method': 'POST',
            'description': '角色分类API',
            'parameters': {
                'file': '图像文件（必填）',
                'use_model': '是否使用专用模型 (true/false, 默认false)'
            },
            'response': {
                'filename': '文件名',
                'role': '识别的角色',
                'similarity': '相似度',
                'boxes': '边界框信息'
            },
            'example': 'curl -X POST -F "file=@image.jpg" -F "use_model=true" http://localhost:5001/api/classify'
        }
        return json.dumps(api_doc, ensure_ascii=False), 200, {'Content-Type': 'application/json'}

    # POST 请求处理图像分类
    if 'file' not in request.files:
        return json.dumps({'error': '没有文件部分'}), 400

    file = request.files['file']
    use_model = request.form.get('use_model') == 'true'
    use_coreml = request.form.get('use_coreml') == 'true'

    # 检查Core ML模型是否可用
    if use_coreml and coreml_model is None:
        return json.dumps({'error': 'Core ML模型不可用'}), 400

    if file.filename == '':
        return json.dumps({'error': '没有选择文件'}), 400

    if file and allowed_file(file.filename):
        # 保存文件到临时位置
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(temp_path)

        try:
            # 分类图像
            if use_coreml:
                # 使用Core ML模型
                role, similarity, boxes = classify_with_coreml(temp_path)
                mode = 'Core ML'
                # 记录分类日志
                record_classification_log(
                    image_path=temp_path,
                    role=role,
                    similarity=similarity,
                    feature=[],
                    boxes=boxes,
                    metadata={'mode': mode, 'use_coreml': True, 'api': True}
                )
            else:
                # 使用默认模型
                classifier = get_classifier()
                role, similarity, boxes = classifier.classify_image(temp_path, use_model=use_model)
                mode = 'EfficientNet' if use_model else 'CLIP'
                # 记录分类日志
                record_classification_log(
                    image_path=temp_path,
                    role=role,
                    similarity=similarity,
                    feature=[],
                    boxes=boxes,
                    metadata={'mode': mode, 'use_model': use_model, 'api': True}
                )

            # 准备结果
            result = {
                'filename': file.filename,
                'role': role if role else '未知',
                'similarity': float(similarity),
                'boxes': boxes,
                'mode': mode
            }

            return json.dumps(result), 200, {'Content-Type': 'application/json'}
        except Exception as e:
            return json.dumps({'error': str(e)}), 500

        finally:
            # 清理临时文件
            if os.path.exists(temp_path):
                os.remove(temp_path)

    return json.dumps({'error': '不支持的文件类型'}), 400


@app.route('/api/track_inference', methods=['POST'])
def track_inference():
    """接收用户体验数据"""
    import json
    from datetime import datetime

    try:
        data = request.json

        # 创建日志目录
        log_dir = os.path.join('logs', 'user_experience')
        os.makedirs(log_dir, exist_ok=True)

        # 保存到日志文件
        log_file = os.path.join(log_dir, f'inference_{datetime.now().strftime("%Y%m%d")}.log')

        with open(log_file, 'a') as f:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')

        return json.dumps({'status': 'success'}), 200
    except Exception as e:
        return json.dumps({'error': str(e)}), 500


@app.route('/workflow', methods=['GET', 'POST'])
def workflow():
    """角色检测工作流"""
    if request.method == 'POST':
        try:
            # 获取表单数据
            characters = request.form.get('characters')
            test_image = request.form.get('test_image')
            max_images = int(request.form.get('max_images', 50))
            batch_size = int(request.form.get('batch_size', 16))
            num_epochs = int(request.form.get('num_epochs', 50))
            learning_rate = float(request.form.get('learning_rate', 5e-5))
            num_workers = int(request.form.get('num_workers', 4))
            threshold = float(request.form.get('threshold', 0.5))
            multiple = 'multiple' in request.form
            grid_size = int(request.form.get('grid_size', 3))

            # 验证输入
            if not characters or not test_image:
                flash('角色信息和测试图像路径不能为空')
                return redirect(request.url)
            
            # 验证并处理角色信息JSON格式
            try:
                import json
                # 清理JSON字符串（去除多余空格和换行符）
                characters = characters.strip()
                print(f"原始角色信息: '{characters}'")
                
                # 尝试自动修复常见的JSON格式问题
                # 1. 替换单引号为双引号
                characters = characters.replace("'", '"')
                # 2. 去除可能的首尾引号
                if characters.startswith('"') and characters.endswith('"'):
                    characters = characters[1:-1]
                # 3. 再次清理
                characters = characters.strip()
                
                print(f"修复后角色信息: '{characters}'")
                
                # 解析验证JSON格式
                parsed_characters = json.loads(characters)
                
                # 验证角色信息格式
                if not isinstance(parsed_characters, list):
                    # 如果不是列表，包装成列表
                    parsed_characters = [parsed_characters]
                
                # 验证每个角色的格式
                for i, char in enumerate(parsed_characters):
                    if not isinstance(char, dict) or 'name' not in char or 'series' not in char:
                        flash(f'角色 {i+1} 格式错误，需要包含 name 和 series 字段')
                        return redirect(request.url)
                
                # 重新序列化确保格式正确
                characters_json = json.dumps(parsed_characters)
                print(f"最终JSON: '{characters_json}'")
            except json.JSONDecodeError as e:
                flash(f'角色信息JSON格式错误: {str(e)}')
                print(f'JSON解析错误: {e}，输入: {characters}')
                return redirect(request.url)
            except Exception as e:
                flash(f'处理角色信息时出错: {str(e)}')
                print(f'处理角色信息错误: {e}')
                return redirect(request.url)

            # 检查测试图像是否存在
            # 转换为绝对路径
            test_image_abs = os.path.abspath(test_image)
            if not os.path.exists(test_image_abs):
                flash(f'测试图像不存在: {test_image}')
                return redirect(request.url)

            # 构建命令列表，避免shell转义问题
            cmd_list = [
                sys.executable,
                'scripts/workflow/character_detection_workflow.py',
                '--characters', characters_json,
                '--test_image', test_image_abs,
                '--max_images', str(max_images),
                '--batch_size', str(batch_size),
                '--num_epochs', str(num_epochs),
                '--learning_rate', str(learning_rate),
                '--num_workers', str(num_workers),
                '--threshold', str(threshold),
                '--grid_size', str(grid_size)
            ]
            
            if multiple:
                cmd_list.append('--multiple')

            # 执行工作流
            import subprocess
            import shlex
            # 使用项目根目录作为工作目录
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
            
            # 构建命令列表，避免shell转义问题
            cmd_list = [
                sys.executable,
                'scripts/workflow/character_detection_workflow.py',
                '--characters', characters_json,
                '--test_image', test_image_abs,
                '--max_images', str(max_images),
                '--batch_size', str(batch_size),
                '--num_epochs', str(num_epochs),
                '--learning_rate', str(learning_rate),
                '--num_workers', str(num_workers),
                '--threshold', str(threshold),
                '--grid_size', str(grid_size)
            ]
            
            if multiple:
                cmd_list.append('--multiple')
            
            print(f"执行命令: {cmd_list}")
            subprocess.Popen(cmd_list, cwd=project_root)

            flash('工作流已启动！请查看终端输出了解进度。')
            return redirect(url_for('workflow'))
        except Exception as e:
            flash(f'工作流启动失败: {str(e)}')
            return redirect(request.url)

    # GET请求，显示工作流表单
    return render_template('workflow.html')


# 创建HTML模板
@app.template_filter('format_similarity')
def format_similarity(value):
    """格式化相似度"""
    return f"{value:.2f}%"


if __name__ == '__main__':
    # 初始化系统
    initialize_system()

    # 运行应用
    port = 5003
    print("启动Flask应用...")
    print(f"访问地址: http://127.0.0.1:{port}")
    app.run(debug=True, host='0.0.0.0', port=port)
