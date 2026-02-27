import os
import platform

# 项目根目录
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

# Flask 应用配置
SECRET_KEY = 'your-secret-key-here'
UPLOAD_FOLDER = os.path.join('src', 'web', 'static', 'uploads')  # 修正上传路径
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB

# 允许的文件扩展名
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
# 允许的视频扩展名
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'wmv', 'flv', 'mkv'}

# 视频处理配置
MAX_PROCESSED_FRAMES = 50
DEFAULT_FRAME_SKIP = 5

# 分类配置
DEFAULT_INDEX_PATH = "role_index"

# Core ML 模型配置
COREML_MODEL_PATH = os.path.join('models', 'character_classifier_best_improved.mlpackage')

# 平台检测
IS_DARWIN = platform.system() == 'Darwin'

# 确保上传目录存在
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
