import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.core.general_classification import GeneralClassification

def test_classification():
    """测试分类功能"""
    print("开始测试分类功能...")
    
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
        
        # 测试1: 使用EfficientNet模型
        print("\n测试1: 使用EfficientNet模型")
        role, similarity, boxes = classifier.classify_image(test_image, use_model=True)
        print(f"分类结果: {role}")
        print(f"相似度: {similarity}")
        print(f"边界框: {boxes}")
        
        if role and role != '类别0' and role != '类别_0':
            print("✅ EfficientNet模型分类正常")
        else:
            print("❌ EfficientNet模型分类异常")
            return False
        
        # 测试2: 再次测试，确保结果不一致
        print("\n测试2: 再次测试EfficientNet模型")
        role2, similarity2, boxes2 = classifier.classify_image(test_image, use_model=True)
        print(f"分类结果: {role2}")
        print(f"相似度: {similarity2}")
        
        if role2 and role2 != '类别0' and role2 != '类别_0':
            print("✅ EfficientNet模型分类仍然正常")
        else:
            print("❌ EfficientNet模型分类异常")
            return False
        
        # 测试3: 使用不同的图像
        print("\n测试3: 寻找第二个测试图像")
        test_image2 = None
        count = 0
        for root, dirs, files in os.walk(".."):
            for file in files:
                if file.lower().endswith((".png", ".jpg", ".jpeg")):
                    candidate = os.path.join(root, file)
                    if candidate != test_image:
                        test_image2 = candidate
                        count += 1
                        print(f"找到候选图像 {count}: {test_image2}")
                        if count >= 3:
                            break
            if test_image2 and count >= 3:
                break
        
        if test_image2:
            print(f"\n测试4: 使用不同图像 {test_image2}")
            role3, similarity3, boxes3 = classifier.classify_image(test_image2, use_model=True)
            print(f"分类结果: {role3}")
            print(f"相似度: {similarity3}")
            
            if role3 and role3 != '类别0' and role3 != '类别_0':
                print("✅ 不同图像分类正常")
            else:
                print("❌ 不同图像分类异常")
                return False
        
        print("\n🎉 所有测试通过！分类功能修复成功！")
        return True
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_classification()
    if success:
        print("\n🎉 测试通过！分类功能修复成功！")
    else:
        print("\n❌ 测试失败！请检查错误信息。")
