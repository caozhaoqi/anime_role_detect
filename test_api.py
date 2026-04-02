import requests

# 测试图像路径
TEST_IMAGE_PATH = "test_images/test1.jpg"

# API端点
API_URL = "http://localhost:8000/api/classify"

def test_api():
    print("测试API服务...")
    
    try:
        # 准备文件数据
        f = open(TEST_IMAGE_PATH, "rb")
        try:
            files = {
                'file': ('test1.jpg', f, 'image/jpeg')
            }
        
            # 准备表单数据
            data = {
                'use_model': 'false',
                'use_attributes': 'true',
                'model_name': 'default',
                'cache_bypass': 'false'
            }
            
            print("发送分类请求...")
            response = requests.post(API_URL, files=files, data=data, timeout=300)
        finally:
            f.close()
        
        if response.status_code == 200:
            result = response.json()
            print("API响应成功!")
            print(f"使用的检测模式: {result.get('detection_mode', 'unknown')}")
            
            if result.get('detection_mode') == 'multi_role':
                roles = result.get('roles', [])
                print(f"检测到的角色数: {len(roles)}")
                for i, role in enumerate(roles):
                    print(f"角色 {i+1}:")
                    print(f"  名称: {role.get('name', 'Unknown')}")
                    print(f"  置信度: {role.get('confidence', 0):.2f}")
            else:
                print(f"角色名称: {result.get('role', 'Unknown')}")
                print(f"置信度: {result.get('similarity', 0):.2f}")
            
            print(f"处理时间: {result.get('processing_time', 0):.2f}秒")
        else:
            print(f"API请求失败，状态码: {response.status_code}")
            print(f"错误信息: {response.text}")
            
    except Exception as e:
        print(f"测试过程中发生错误: {str(e)}")

if __name__ == "__main__":
    test_api()
