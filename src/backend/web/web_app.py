#!/usr/bin/env python3
"""
Flask网页应用
实现图片上传和角色分类功能
"""
import os
import sys
import tempfile
import platform
import json
import numpy as np
from flask import Flask, request, render_template, redirect, url_for, flash
from flask_cors import CORS
from PIL import Image
from loguru import logger

# 解决OpenMP冲突问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# 导入通用分类模块
from core.classification.general_classification import GeneralClassification, get_classifier
# 导入日志记录模块
from core.log_fusion.log_recorder import record_classification_log

# Core ML模型加载（仅在macOS上）
coreml_model = None
if platform.system() == 'Darwin':
    try:
        import coremltools

        coreml_model_path = os.path.join('models', 'character_classifier_best_improved.mlpackage')
        if os.path.exists(coreml_model_path):
            coreml_model = coremltools.models.MLModel(coreml_model_path)
            logger.debug(f"Core ML模型已加载: {coreml_model_path}")
    except ImportError:
        logger.debug("coremltools未安装，Core ML功能不可用")
    except Exception as e:
        logger.debug(f"Core ML模型加载失败: {e}")

# 初始化Flask应用
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = os.path.join('src', 'web', 'static', 'uploads')  # 修正上传路径
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# 启用CORS，允许跨域请求
CORS(app)
logger.debug("🌐 CORS已启用，允许跨域请求")

# 确保上传目录存在
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# 允许的文件扩展名
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
# 允许的视频扩展名
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'wmv', 'flv', 'mkv'}


def allowed_file(filename):
    """检查文件扩展名是否允许（图片文件）"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def allowed_video_file(filename):
    """检查文件扩展名是否允许（视频文件）"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS


def allowed_media_file(filename):
    """检查文件扩展名是否允许（媒体文件，包括图片和视频）"""
    return allowed_file(filename) or allowed_video_file(filename)


def process_video(video_path, frame_skip=5, model=''):
    """处理视频文件，提取帧并进行分类
    
    Args:
        video_path: 视频路径
        frame_skip: 帧跳过间隔，用于减少处理帧数
    
    Returns:
        (video_results, overall_role, overall_similarity): 视频帧检测结果、整体角色、整体相似度
    """
    import cv2
    import tempfile
    import os
    
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("无法打开视频文件")
    
    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    video_results = []
    frame_count = 0
    processed_frames = 0
    role_counts = {}
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # 跳过一些帧以提高处理速度
            if frame_count % frame_skip != 0:
                frame_count += 1
                continue
            
            # 将帧保存为临时图像文件
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_frame:
                temp_frame_path = temp_frame.name
            
            # 保存帧
            cv2.imwrite(temp_frame_path, frame)
            
            try:
                # 对帧进行分类
                classifier = get_classifier(model=model)
                role, similarity, boxes = classifier.classify_image(temp_frame_path)
                
                # 计算时间戳
                timestamp = frame_count / fps
                
                # 添加结果
                video_results.append({
                    'frame': frame_count,
                    'role': role,
                    'similarity': similarity,
                    'timestamp': timestamp,
                    'boxes': boxes
                })
                
                # 统计角色出现次数
                if role not in role_counts:
                    role_counts[role] = 0
                role_counts[role] += similarity
                
                processed_frames += 1
                
                # 限制处理的帧数，避免处理时间过长
                if processed_frames >= 50:
                    break
                    
            except Exception as e:
                logger.debug(f"处理帧 {frame_count} 时出错: {e}")
            finally:
                # 清理临时文件
                if os.path.exists(temp_frame_path):
                    os.remove(temp_frame_path)
            
            frame_count += 1
    
    finally:
        cap.release()
    
    # 确定整体角色
    overall_role = "未知"
    overall_similarity = 0.0
    
    if role_counts:
        # 选择出现次数最多的角色
        overall_role = max(role_counts, key=role_counts.get)
        overall_similarity = role_counts[overall_role] / processed_frames if processed_frames > 0 else 0
    
    return video_results, overall_role, overall_similarity


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
                # 提取角色名，去掉前缀（如 sdv50_ 或 原神_）
                if '_' in role:
                    role = role.split('_')[-1]
            else:
                role = "未知"
        else:
            role = "未知"

        # Core ML模型不提供边界框信息
        boxes = []

        return role, similarity, boxes

    except Exception as e:
        raise ValueError(f"Core ML推理失败: {e}")


def initialize_system():
    """初始化分类系统"""
    logger.debug("初始化分类系统...")
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
    frame_skip = int(request.form.get('frame_skip', '5'))
    model = request.form.get('model', '')
    # 同时支持model_name参数，保持与前端的兼容性
    model_name = request.form.get('model_name', '')
    if model_name:
        model = model_name
    
    logger.debug("📋 参数:", {
        'use_model': use_model,
        'use_coreml': use_coreml,
        'frame_skip': frame_skip,
        'model': model,
        'model_name': model_name
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
                role, similarity, boxes = classify_with_coreml(temp_path)
                mode = 'Core ML'
                # 记录分类日志
                record_classification_log(
                    image_path=temp_path,
                    role=role,
                    similarity=similarity,
                    feature=[],
                    boxes=boxes,
                    metadata={'mode': mode, 'use_coreml': True, 'api': True, 'fileType': 'image'}
                )
            else:
                # 使用默认模型
                logger.debug("🤖 使用默认模型")
                classifier = get_classifier(model=model)
                
                # 首先尝试使用集成方法，提高识别精度
                try:
                    logger.debug("🔄 尝试使用集成分类方法")
                    role, similarity, boxes, attributes = classifier.classify_image_with_deepdanbooru(temp_path, use_attributes=True)
                    mode = 'Ensemble (CLIP + EfficientNet + DeepDanbooru)'
                except Exception as e:
                    logger.warning(f"集成分类失败，回退到单一模型: {e}")
                    # 回退到使用指定模型或默认模型
                    if use_model:
                        logger.debug("🤖 使用专用模型")
                        role, similarity, boxes, attributes = classifier.classify_image(temp_path, use_model=True, use_attributes=True)
                        mode = 'EfficientNet'
                    else:
                        # 尝试使用CLIP模型
                        try:
                            logger.debug("🤖 尝试使用CLIP模型")
                            role, similarity, boxes, attributes = classifier.classify_image(temp_path, use_model=False, use_attributes=True)
                            mode = 'CLIP'
                        except Exception as clip_error:
                            # 如果CLIP模型失败（例如索引不存在），回退到使用专用模型
                            logger.warning(f"CLIP模型分类失败，回退到专用模型: {clip_error}")
                            logger.debug("🤖 回退到使用专用模型")
                            role, similarity, boxes, attributes = classifier.classify_image(temp_path, use_model=True, use_attributes=True)
                            mode = 'EfficientNet'
                
                # 记录分类日志
                record_classification_log(
                    image_path=temp_path,
                    role=role,
                    similarity=similarity,
                    feature=[],
                    boxes=boxes,
                    metadata={'mode': mode, 'use_model': use_model, 'api': True, 'fileType': 'image'}
                )

            # 准备结果
            result = {
                'filename': file.filename,
                'role': role if role else '未知',
                'similarity': float(similarity),
                'boxes': boxes,
                'fileType': 'image',
                'mode': mode,
                'attributes': attributes
            }
            logger.debug("✅ 图像处理完成，结果:", result)
        else:
            # 处理视频文件
            logger.debug("🎬 开始处理视频文件")
            logger.debug("🎬 调用process_video函数，frame_skip:", frame_skip)
            video_results, overall_role, overall_similarity = process_video(temp_path, frame_skip, model)
            mode = 'Video Processing'
            
            logger.debug("🎬 视频处理完成，处理了", len(video_results), "帧")
            logger.debug("🎬 整体角色:", overall_role, "相似度:", overall_similarity)
            
            # 记录分类日志
            record_classification_log(
                image_path=temp_path,
                role=overall_role,
                similarity=overall_similarity,
                feature=[],
                boxes=[],
                metadata={'mode': mode, 'api': True, 'fileType': 'video', 'processed_frames': len(video_results)}
            )

            # 准备结果
            result = {
                'filename': file.filename,
                'role': overall_role if overall_role else '未知',
                'similarity': float(overall_similarity),
                'boxes': [],
                'fileType': 'video',
                'videoResults': video_results,
                'mode': mode
            }
            logger.debug("✅ 视频处理完成，结果:", result)

        logger.debug("📡 返回结果:", result)
        return json.dumps(result), 200, {'Content-Type': 'application/json'}
    except Exception as e:
        logger.debug(f"❌ 处理文件时出错: {str(e)}")
        import traceback
        error_stack = traceback.format_exc()
        logger.debug(f"📋 错误堆栈: {error_stack}")
        return json.dumps({'error': str(e), 'stack': error_stack}), 500

    finally:
        # 清理临时文件
        if os.path.exists(temp_path):
            logger.debug("🗑️ 清理临时文件:", temp_path)
            os.remove(temp_path)
            logger.debug("✅ 临时文件清理成功")
        logger.debug("="*80)
        logger.debug("🔚 API分类请求处理完成")
        logger.debug("="*80)


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


@app.route('/api/models', methods=['GET'])
def api_get_models():
    """获取可用模型列表"""
    try:
        # 定义模型目录
        models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
        
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
                logger.debug(f"原始角色信息: '{characters}'")
                
                # 尝试自动修复常见的JSON格式问题
                # 1. 替换单引号为双引号
                characters = characters.replace("'", '"')
                # 2. 去除可能的首尾引号
                if characters.startswith('"') and characters.endswith('"'):
                    characters = characters[1:-1]
                # 3. 再次清理
                characters = characters.strip()
                
                logger.debug(f"修复后角色信息: '{characters}'")
                
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
                logger.debug(f"最终JSON: '{characters_json}'")
            except json.JSONDecodeError as e:
                flash(f'角色信息JSON格式错误: {str(e)}')
                logger.debug(f'JSON解析错误: {e}，输入: {characters}')
                return redirect(request.url)
            except Exception as e:
                flash(f'处理角色信息时出错: {str(e)}')
                logger.debug(f'处理角色信息错误: {e}')
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
            
            logger.debug(f"执行命令: {cmd_list}")
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

    # 设置文档路由
    try:
        from routes.docs_routes import setup_docs_routes
        setup_docs_routes(app)
        logger.debug("📚 API文档路由已设置，访问路径: http://127.0.0.1:5001/docs")
    except Exception as e:
        logger.warning(f"⚠️ API文档路由设置失败: {e}")

    # 运行应用
    port = 5001
    logger.debug("启动Flask应用...")
    logger.debug(f"访问地址: http://127.0.0.1:{port}")
    app.run(debug=True, host='0.0.0.0', port=port)
