import os
import sys
import requests

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath('.'))

def test_classification_accuracy():
    """测试训练模型的分类准确性"""
    # 数据目录
    data_dir = 'data/train'
    
    # 检查数据目录是否存在
    if not os.path.exists(data_dir):
        print(f"数据目录不存在: {data_dir}")
        return
    
    # 获取角色目录列表
    role_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    print(f"发现 {len(role_dirs)} 个角色目录")
    
    # 测试结果
    total_tests = 0
    correct_predictions = 0
    
    # 遍历每个角色目录
    for role_name in role_dirs:
        role_dir = os.path.join(data_dir, role_name)
        # 获取目录下的图片文件
        image_files = [f for f in os.listdir(role_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))]
        
        # 跳过没有图片的目录
        if len(image_files) == 0:
            print(f"跳过角色 '{role_name}'，没有图片")
            continue
        
        print(f"\n测试角色 '{role_name}'，发现 {len(image_files)} 张图片")
        
        # 测试每张图片（只测试前5张）
        for i, img_file in enumerate(image_files[:5]):
            img_path = os.path.join(role_dir, img_file)
            total_tests += 1
            
            try:
                # 发送分类请求
                with open(img_path, 'rb') as f:
                    # 根据文件扩展名设置正确的Content-Type
                    ext = os.path.splitext(img_path)[1].lower()
                    content_type = 'image/jpeg'
                    if ext in ['.png']:
                        content_type = 'image/png'
                    elif ext in ['.gif']:
                        content_type = 'image/gif'
                    elif ext in ['.bmp']:
                        content_type = 'image/bmp'
                    
                    response = requests.post(
                        'http://127.0.0.1:8000/api/classify',
                        files={'file': (os.path.basename(img_path), f, content_type)},
                        data={'model_name': 'arona_plana'}
                    )
                
                # 解析响应
                if response.status_code == 200:
                    result = response.json()
                    predicted_role = result.get('role', 'unknown')
                    similarity = result.get('similarity', 0.0)
                    
                    # 检查分类是否正确
                    if predicted_role == role_name:
                        correct_predictions += 1
                        print(f"✓ {img_file}: 正确分类为 '{predicted_role}'，相似度: {similarity:.4f}")
                    else:
                        print(f"✗ {img_file}: 错误分类为 '{predicted_role}' (应为 '{role_name}')，相似度: {similarity:.4f}")
                else:
                    print(f"✗ {img_file}: API请求失败，状态码: {response.status_code}")
                    
            except Exception as e:
                print(f"✗ {img_file}: 测试失败: {e}")
    
    # 计算准确率
    if total_tests > 0:
        accuracy = (correct_predictions / total_tests) * 100
        print(f"\n测试完成: 共测试 {total_tests} 张图片，正确 {correct_predictions} 张，准确率: {accuracy:.2f}%")
    else:
        print("\n没有测试任何图片")

if __name__ == "__main__":
    test_classification_accuracy()
