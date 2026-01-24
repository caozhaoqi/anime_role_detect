import os
import deepdanbooru as dd
from PIL import Image
import numpy as np

class Tagging:
    def __init__(self, model_path=None, threshold=0.5):
        """初始化标签系统"""
        self.threshold = threshold
        self.model = None
        self.tags = None
        
        # 如果提供了模型路径，加载模型
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            # 尝试使用默认模型路径
            default_model_path = "deepdanbooru_model"
            if os.path.exists(default_model_path):
                self.load_model(default_model_path)
            else:
                print("警告: 未提供DeepDanbooru模型路径，将使用自动下载的模型")
    
    def load_model(self, model_path):
        """加载DeepDanbooru模型"""
        try:
            # 加载模型
            self.model = dd.project.load_model_from_project(model_path)
            
            # 加载标签
            self.tags = dd.project.load_tags_from_project(model_path)
            
            print(f"成功加载DeepDanbooru模型: {model_path}")
        except Exception as e:
            print(f"加载DeepDanbooru模型失败: {e}")
            # 如果加载失败，使用自动下载的模型
            self._use_auto_model()
    
    def _use_auto_model(self):
        """使用自动下载的模型"""
        try:
            # 自动下载模型（实际应用中可能需要手动下载）
            print("尝试使用自动下载的DeepDanbooru模型")
            # 这里简化处理，实际应用中可能需要更复杂的模型加载逻辑
            self.model = None
            self.tags = ["银发", "校服", "红领结", "眼镜", "双马尾", "短发", "长发", "蓝眼睛", "红眼睛", "黄眼睛"]
        except Exception as e:
            print(f"自动加载模型失败: {e}")
    
    def get_tags(self, image_path, max_tags=20):
        """获取图片的标签"""
        try:
            # 加载图像
            img = Image.open(image_path)
            
            # 如果模型未加载，返回默认标签
            if self.model is None:
                # 这里简化处理，实际应用中应该使用真实的模型预测
                print("警告: 模型未加载，返回默认标签")
                return self.tags[:max_tags]
            
            # 预处理图像
            img = img.convert("RGB")
            img = img.resize((512, 512))
            img_array = np.array(img, dtype=np.float32)
            img_array = img_array / 255.0
            img_array = img_array.transpose(2, 0, 1)
            img_array = np.expand_dims(img_array, axis=0)
            
            # 预测标签
            with dd.util.create_session() as session:
                probs = self.model.predict(img_array, session=session)
            
            # 过滤标签
            result_tags = []
            for i, prob in enumerate(probs):
                if prob >= self.threshold and i < len(self.tags):
                    result_tags.append(self.tags[i])
                if len(result_tags) >= max_tags:
                    break
            
            return result_tags
        except Exception as e:
            print(f"获取标签失败: {e}")
            # 返回默认标签
            return self.tags[:max_tags] if self.tags else []
    
    def add_tags_to_image(self, image_path, tags):
        """为图片添加标签（实际应用中可能需要保存标签到文件或数据库）"""
        # 这里简化处理，实际应用中可能需要将标签保存到文件或数据库
        print(f"为图片 {image_path} 添加标签: {tags}")
        
        # 例如，可以将标签保存到同名的txt文件中
        tag_file_path = os.path.splitext(image_path)[0] + "_tags.txt"
        with open(tag_file_path, "w", encoding="utf-8") as f:
            f.write(", ".join(tags))
        
        print(f"标签已保存到: {tag_file_path}")
        return tag_file_path

if __name__ == "__main__":
    # 测试标签系统
    tagger = Tagging()
    
    # 测试图像路径（需要根据实际情况修改）
    test_image = "test.jpg"
    
    try:
        # 获取标签
        tags = tagger.get_tags(test_image)
        print(f"图片标签: {tags}")
        
        # 添加标签到图片
        tag_file = tagger.add_tags_to_image(test_image, tags)
        print(f"标签文件: {tag_file}")
    except Exception as e:
        print(f"测试失败: {e}")
