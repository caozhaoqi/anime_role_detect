import os
import requests

def test_classification_with_timeout():
    """测试分类API，带有超时参数"""
    # 测试图片路径
    test_images = [
        ('data/train/日奈/日奈_1.jpg', '日奈'),
        ('data/train/伊织/伊织_1.jpg', '伊织'),
    ]
    
    for img_path, expected_role in test_images:
        if not os.path.exists(img_path):
            print(f"图片不存在: {img_path}")
            continue
        
        try:
            # 发送分类请求，设置超时为30秒
            with open(img_path, 'rb') as f:
                response = requests.post(
                    'http://localhost:8000/api/classify',
                    files={'file': (os.path.basename(img_path), f, 'image/jpeg')},
                    data={'model_name': 'arona_plana'},
                    timeout=30
                )
            
            # 解析响应
            if response.status_code == 200:
                result = response.json()
                predicted_role = result.get('role', 'unknown')
                similarity = result.get('similarity', 0.0)
                
                print(f"✓ {os.path.basename(img_path)}: 分类为 '{predicted_role}'，相似度: {similarity:.4f}")
            else:
                print(f"✗ {os.path.basename(img_path)}: API请求失败，状态码: {response.status_code}")
                
        except requests.exceptions.Timeout:
            print(f"✗ {os.path.basename(img_path)}: 请求超时")
        except Exception as e:
            print(f"✗ {os.path.basename(img_path)}: 测试失败: {e}")

if __name__ == "__main__":
    test_classification_with_timeout()
