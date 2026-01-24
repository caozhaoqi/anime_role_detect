import os
import logging
import traceback
from datetime import datetime

class ExceptionHandling:
    def __init__(self, log_dir="logs"):
        """初始化异常处理模块"""
        self.log_dir = log_dir
        
        # 创建日志目录
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            print(f"创建日志目录: {self.log_dir}")
        
        # 配置日志
        self._configure_logging()
    
    def _configure_logging(self):
        """配置日志系统"""
        # 生成日志文件名
        log_filename = f"error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        log_path = os.path.join(self.log_dir, log_filename)
        
        # 配置日志
        logging.basicConfig(
            filename=log_path,
            level=logging.ERROR,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 添加控制台输出
        console = logging.StreamHandler()
        console.setLevel(logging.ERROR)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)
        
        print(f"日志已配置，错误将记录到: {log_path}")
    
    def handle_exception(self, e, context=""):
        """处理异常"""
        # 记录异常信息
        error_message = f"{context}: {str(e)}"
        logging.error(error_message)
        logging.error(traceback.format_exc())
        
        # 返回错误信息
        return error_message
    
    def analyze_error(self, error_message):
        """分析错误类型"""
        # 常见错误类型分析
        error_types = {
            "文件不存在": ["No such file", "does not exist", "FileNotFound"],
            "图像格式错误": ["cannot identify image file", "Unsupported image format", "invalid image"],
            "模型加载失败": ["model not found", "failed to load model", "ModelError"],
            "内存不足": ["out of memory", "OOM", "memory error"],
            "权限错误": ["permission denied", "access denied", "PermissionError"]
        }
        
        # 分析错误类型
        for error_type, keywords in error_types.items():
            if any(keyword.lower() in error_message.lower() for keyword in keywords):
                return error_type
        
        # 未知错误类型
        return "未知错误"
    
    def log_image_error(self, image_path, error_message):
        """记录图片处理错误"""
        # 生成错误日志
        error_type = self.analyze_error(error_message)
        log_message = f"图片 {image_path} 处理失败: {error_message} (错误类型: {error_type})"
        
        # 记录错误
        logging.error(log_message)
        
        print(f"[错误] {log_message}")
        
        return error_type
    
    def create_error_report(self, report_path="error_report.txt"):
        """创建错误报告"""
        # 收集所有错误日志
        error_logs = []
        for log_file in os.listdir(self.log_dir):
            if log_file.startswith("error_") and log_file.endswith(".log"):
                log_path = os.path.join(self.log_dir, log_file)
                with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                    error_logs.extend(f.readlines())
        
        # 生成错误报告
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"错误报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"总错误数: {len(error_logs)}\n")
            f.write("\n错误详情:\n")
            f.write("-" * 80 + "\n")
            for log in error_logs:
                f.write(log)
        
        print(f"错误报告已生成: {report_path}")
        return report_path

if __name__ == "__main__":
    # 测试异常处理模块
    error_handler = ExceptionHandling()
    
    # 测试异常处理
    try:
        # 模拟一个文件不存在的错误
        with open("non_existent_file.txt", "r") as f:
            pass
    except Exception as e:
        error_message = error_handler.handle_exception(e, "测试文件读取")
        error_type = error_handler.analyze_error(error_message)
        print(f"错误类型: {error_type}")
    
    # 测试图片错误日志
    test_image_path = "test.jpg"
    test_error_message = "cannot identify image file"
    error_type = error_handler.log_image_error(test_image_path, test_error_message)
    print(f"图片错误类型: {error_type}")
    
    # 生成错误报告
    report_path = error_handler.create_error_report()
    print(f"错误报告路径: {report_path}")
