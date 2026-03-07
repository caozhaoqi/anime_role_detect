import numpy as np
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.core.general_classification import GeneralClassification

def test_infinity_handling():
    """测试无穷大值处理"""
    print("开始测试无穷大值处理...")
    
    try:
        # 初始化分类器
        classifier = GeneralClassification()
        classifier.initialize()
        
        # 创建一个测试图像（使用项目中的示例图像）
        test_image = "src/web/static/uploads/test_image.png"
        
        # 如果测试图像不存在，使用第一个可用图像
        if not os.path.exists(test_image):
            # 查找第一个PNG或JPG文件
            for root, dirs, files in os.walk(".."):
                for file in files:
                    if file.lower().endswith((".png", ".jpg", ".jpeg")):
                        test_image = os.path.join(root, file)
                        print(f"使用测试图像: {test_image}")
                        break
                if 'test_image' in locals() and os.path.exists(test_image):
                    break
        
        if not os.path.exists(test_image):
            print("错误: 找不到测试图像")
            return False
        
        # 测试分类 - 使用EfficientNet模型，不需要索引
        print("测试分类功能...")
        role, similarity, boxes = classifier.classify_image(test_image, use_model=True)
        
        print(f"分类结果: {role}")
        print(f"相似度: {similarity}")
        print(f"边界框: {boxes}")
        
        # 检查相似度是否为有效值
        if similarity is not None and isinstance(similarity, (int, float)):
            if not np.isinf(similarity) and not np.isnan(similarity):
                print("✓ 相似度值有效，没有无穷大或NaN")
                return True
            else:
                print(f"✗ 相似度值无效: {similarity}")
                return False
        else:
            print(f"✗ 相似度值无效: {similarity}")
            return False
            
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_infinity_handling()
    if success:
        print("\n🎉 测试通过！无穷大值处理修复成功！")
    else:
        print("\n❌ 测试失败！请检查错误信息。")
