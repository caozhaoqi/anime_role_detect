import argparse
import os
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.scripts.data_preparation.data_preparation import DataPreparation
from src.scripts.classification_script.classification_script import ClassificationScript
from src.web.web_ui import WebUI

class AnimeRoleDetect:
    def __init__(self):
        """初始化二次元角色识别系统"""
        pass
    
    def build_index(self, data_dir="dataset", index_path="role_index"):
        """构建向量索引"""
        print("=== 构建向量索引 ===")
        data_prep = DataPreparation(data_dir, index_path)
        
        # 组织数据集目录结构
        data_prep.organize_dataset()
        
        # 验证数据集
        try:
            data_prep.validate_dataset()
        except Exception as e:
            print(f"数据集验证失败: {e}")
            return False
        
        # 构建索引
        try:
            features, role_names = data_prep.build_index_from_dataset()
            print("索引构建成功!")
            return True
        except Exception as e:
            print(f"索引构建失败: {e}")
            return False
    
    def classify_images(self, input_dir="input", output_dir="output", index_path="role_index", threshold=0.7):
        """分类图片"""
        print("=== 分类图片 ===")
        
        # 验证输入目录
        if not os.path.exists(input_dir):
            print(f"输入目录不存在: {input_dir}")
            print("请创建输入目录并放入需要分类的图片")
            return False
        
        # 验证索引文件
        if not os.path.exists(f"{index_path}.faiss") or not os.path.exists(f"{index_path}_mapping.json"):
            print(f"索引文件不存在: {index_path}.faiss 或 {index_path}_mapping.json")
            print("请先运行 build_index 命令构建索引")
            return False
        
        # 创建分类脚本实例并运行
        script = ClassificationScript(input_dir, output_dir, index_path, threshold)
        script.classify_and_archive()
        return True
    
    def run_web_ui(self, index_path="role_index", threshold=0.7, share=False, server_port=7860):
        """运行Web UI"""
        print("=== 运行Web UI ===")
        web_ui = WebUI(index_path, threshold)
        web_ui.launch(share=share, server_port=server_port)
    
    def correct_classification(self, image_path, correct_role, index_path="role_index"):
        """修正分类结果并更新索引"""
        print("=== 修正分类结果 ===")
        
        # 验证图像文件是否存在
        if not os.path.exists(image_path):
            print(f"图像文件不存在: {image_path}")
            return False
        
        # 验证索引文件
        if not os.path.exists(f"{index_path}.faiss") or not os.path.exists(f"{index_path}_mapping.json"):
            print(f"索引文件不存在: {index_path}.faiss 或 {index_path}_mapping.json")
            print("请先运行 build_index 命令构建索引")
            return False
        
        # 初始化分类器
        from src.core.classification.classification import Classification
        classifier = Classification(index_path)
        
        # 执行增量学习
        success = classifier.incremental_learning(image_path, correct_role)
        
        if success:
            # 重新保存索引
            classifier.save_index(index_path)
            print("分类结果已修正，索引已更新!")
        else:
            print("修正分类结果失败!")
        
        return success
    
    def main(self):
        """主函数"""
        parser = argparse.ArgumentParser(description="二次元角色识别与分类系统")
        parser.add_argument("command", choices=["build_index", "classify", "web_ui", "correct"], help="执行的命令")
        parser.add_argument("--data_dir", default="dataset", help="数据集目录")
        parser.add_argument("--index_path", default="role_index", help="向量索引路径")
        parser.add_argument("--input_dir", default="input", help="输入图片目录")
        parser.add_argument("--output_dir", default="output", help="输出结果目录")
        parser.add_argument("--threshold", type=float, default=0.7, help="相似度阈值")
        parser.add_argument("--share", action="store_true", help="是否分享Web UI")
        parser.add_argument("--server_port", type=int, default=7860, help="Web UI服务器端口")
        parser.add_argument("--image_path", help="需要修正分类的图像路径")
        parser.add_argument("--correct_role", help="正确的角色名称")
        
        args = parser.parse_args()
        
        if args.command == "build_index":
            self.build_index(args.data_dir, args.index_path)
        elif args.command == "classify":
            self.classify_images(args.input_dir, args.output_dir, args.index_path, args.threshold)
        elif args.command == "web_ui":
            self.run_web_ui(args.index_path, args.threshold, args.share, args.server_port)
        elif args.command == "correct":
            if not args.image_path or not args.correct_role:
                print("错误: 请提供 --image_path 和 --correct_role 参数")
                parser.print_help()
                return
            self.correct_classification(args.image_path, args.correct_role, args.index_path)

if __name__ == "__main__":
    anime_role_detect = AnimeRoleDetect()
    anime_role_detect.main()
