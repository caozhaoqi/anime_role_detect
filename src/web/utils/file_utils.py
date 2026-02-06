from src.web.config.config import ALLOWED_EXTENSIONS, ALLOWED_VIDEO_EXTENSIONS

def allowed_file(filename):
    """检查文件扩展名是否允许（图片文件）"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def allowed_video_file(filename):
    """检查文件扩展名是否允许（视频文件）"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS


def allowed_media_file(filename):
    """检查文件扩展名是否允许（媒体文件，包括图片和视频）"""
    return allowed_file(filename) or allowed_video_file(filename)
