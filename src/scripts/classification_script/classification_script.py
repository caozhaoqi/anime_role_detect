import os
import shutil
import argparse
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.core.preprocessing.preprocessing import Preprocessing
from src.core.feature_extraction.feature_extraction import FeatureExtraction
from src.core.classification.classification import Classification


class ClassificationScript:
    def __init__(self, input_dir, output_dir, index_path, threshold=0.7):
        """初始化分类脚本"""
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.index_path = index_path
        self.threshold = threshold
        
        # 初始化各个模块
        self.preprocessor = Preprocessing()
        self.extractor = FeatureExtraction()
        self.classifier = Classification(index_path, threshold)
        
        # 创建输出目录
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"创建输出目录: {self.output_dir}")
        
        # 创建Unknown目录
        self.unknown_dir = os.path.join(self.output_dir, "Unknown")
        if not os.path.exists(self.unknown_dir):
            os.makedirs(self.unknown_dir)
            print(f"创建Unknown目录: {self.unknown_dir}")
    
    def get_image_files(self):
        """获取输入目录中的所有图片文件"""
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
        image_files = []
        
        for root, _, files in os.walk(self.input_dir):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_files.append(os.path.join(root, file))
        
        return image_files
    
    def process_image(self, image_path):
        """处理单个图片"""
        try:
            # 预处理图像
            normalized_img, _ = self.preprocessor.process(image_path)
            
            # 提取特征
            feature = self.extractor.extract_features(normalized_img)
            
            # 分类
            role, similarity = self.classifier.classify(feature)
            
            return role, similarity
        except Exception as e:
            print(f"处理图像 {image_path} 失败: {e}")
            return "unknown", 0.0
    
    def process_multiple_characters(self, image_path):
        """处理图片中的多个角色"""
        try:
            # 处理多个角色
            processed_characters = self.preprocessor.process_multiple_characters(image_path)
            
            if not processed_characters:
                return []
            
            # 提取特征
            characters_with_features = self.extractor.extract_features_from_multiple_characters(processed_characters)
            
            # 分类
            classified_characters = self.classifier.classify_multiple_characters(characters_with_features)
            
            return classified_characters
        except Exception as e:
            print(f"处理多角色图像 {image_path} 失败: {e}")
            return []
    
    def classify_and_archive(self):
        """分类并归档所有图片"""
        # 获取所有图片文件
        image_files = self.get_image_files()
        if not image_files:
            print(f"输入目录中没有找到图片文件: {self.input_dir}")
            return
        
        print(f"找到 {len(image_files)} 张图片，开始处理...")
        
        # 处理每张图片
        for i, image_path in enumerate(image_files):
            print(f"处理第 {i+1}/{len(image_files)} 张图片: {os.path.basename(image_path)}")
            
            # 处理多角色
            characters = self.process_multiple_characters(image_path)
            
            if characters:
                # 选择置信度最高的角色
                best_character = max(characters, key=lambda x: x.get('similarity', 0.0))
                role = best_character.get('role', 'unknown')
                similarity = best_character.get('similarity', 0.0)
                
                print(f"  检测到 {len(characters)} 个角色，选择置信度最高的: {role} (相似度: {similarity:.4f})")
            else:
                # 回退到单角色处理
                role, similarity = self.process_image(image_path)
            
            # 确定目标目录
            if role == "unknown" or similarity < self.threshold:
                target_dir = self.unknown_dir
                print(f"  无法识别，放入Unknown目录 (相似度: {similarity:.4f})")
            else:
                target_dir = os.path.join(self.output_dir, role)
                # 创建角色目录（如果不存在）
                if not os.path.exists(target_dir):
                    os.makedirs(target_dir)
                    print(f"  创建角色目录: {target_dir}")
                print(f"  识别为: {role} (相似度: {similarity:.4f})")
            
            # 复制图片到目标目录
            try:
                # 获取文件名
                filename = os.path.basename(image_path)
                target_path = os.path.join(target_dir, filename)
                
                # 避免文件名冲突
                counter = 1
                while os.path.exists(target_path):
                    name, ext = os.path.splitext(filename)
                    target_path = os.path.join(target_dir, f"{name}_{counter}{ext}")
                    counter += 1
                
                # 复制文件
                shutil.copy2(image_path, target_path)
                print(f"  已复制到: {target_path}")
            except Exception as e:
                print(f"  复制文件失败: {e}")
        
        print("\n处理完成!")
        print(f"总计处理 {len(image_files)} 张图片")
    
    def correct_classification(self, image_path, correct_role):
        """修正分类结果并更新索引"""
        # 执行增量学习
        success = self.classifier.incremental_learning(image_path, correct_role)
        
        if success:
            # 重新保存索引
            self.classifier.save_index(self.index_path)
            print(f"分类结果已修正，索引已更新: 图像 {image_path} 已添加到角色 {correct_role} 的索引中")
        else:
            print(f"修正分类结果失败: 图像 {image_path}")
        
        return success


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="二次元角色识别与分类脚本")
    parser.add_argument("--input", type=str, default="input", help="输入图片目录")
    parser.add_argument("--output", type=str, default="output", help="输出结果目录")
    parser.add_argument("--index", type=str, default="role_index", help="向量索引路径")
    parser.add_argument("--threshold", type=float, default=0.7, help="相似度阈值")
    
    args = parser.parse_args()
    
    # 验证输入目录
    if not os.path.exists(args.input):
        print(f"输入目录不存在: {args.input}")
        print("请创建输入目录并放入需要分类的图片")
        sys.exit(1)
    
    # 验证索引文件
    if not os.path.exists(f"{args.index}.faiss") or not os.path.exists(f"{args.index}_mapping.json"):
        print(f"索引文件不存在: {args.index}.faiss 或 {args.index}_mapping.json")
        print("请先运行 data_preparation.py 构建索引")
        sys.exit(1)
    
    # 创建分类脚本实例并运行
    script = ClassificationScript(args.input, args.output, args.index, args.threshold)
    script.classify_and_archive()
